include("wrapper.jl")
include("metrics.jl")
using Random
using StatsBase

function init_model(config::WrapperConfig)
	@assert config.pattern_id!=nothing "Config must have a pattern id"
	TransformerWrapper(config)
end


# per_gpu_train_batch_size: the number of labeled training examples per batch and gpu
# per_gpu_unlabeled_batch_size: the number of unlabeled examples per batch and gpu
# n_gpu: the number of gpus to use
# num_train_epochs: the number of epochs to train for
# max_steps: the maximum number of steps to train for (overrides ``num_train_epochs``)
# gradient_accumulation_steps: the number of steps to accumulate gradients for before performing an update
# weight_decay: the weight decay to use
# learning_rate: the maximum learning rate to use
# adam_epsilon: the epsilon value for Adam
# warmup_steps: the number of warmup steps to perform before reaching the maximum learning rate
# max_grad_norm: the maximum norm for the gradient
# lm_training: whether to perform auxiliary language modeling (only for MLMs)
# use_logits: whether to use each training example's logits instead of its label (used for distillation)
# alpha: the alpha parameter for auxiliary language modeling
# temperature: the temperature for distillation
# atype: KnetArray{Float32} or Array{Float32}
mutable struct TrainConfig
	batch_size
	unlabeled_batch_size
	num_train_epochs
	max_steps
	gradient_accumulation_steps
	weight_decay
	learning_rate
	adam_epsilon
	warmup_steps
	max_grad_norm
	lm_training
	use_logits
	alpha
	temperature
	atype
end

function TrainConfig(;
	batch_size=4,
	unlabeled_batch_size=8,
	num_train_epochs=3,
	max_steps=-1,
	gradient_accumulation_steps=1,
	weight_decay=0,
	learning_rate=5e-5,
	adam_epsilon=1e-8,
	warmup_steps=0,
	max_grad_norm=1,
	lm_training=false,
	use_logits=false, 
	alpha=0.9999,
	temperature=1,
	atype=atype())

	TrainConfig(batch_size,unlabeled_batch_size,num_train_epochs,max_steps,gradient_accumulation_steps,weight_decay,learning_rate,adam_epsilon,warmup_steps,max_grad_norm,lm_training,use_logits,alpha,temperature,atype)
end


struct EvalConfig
	batch_size
    metrics
    atype
end

function EvalConfig(;
	batch_size=8,
    metrics=nothing,
    atype=atype()
	)
	EvalConfig(batch_size, metrics, atype)
end	


# generations: the number of generations to train
# logits_percentage: the percentage of models to use for annotating training sets for the next generation
# scale_factor: the factor by which the training set is increased for each generation
# n_most_likely: If >0, in the first generation the n_most_likely examples per label are chosen even
#                      if their predicted label is different
struct IPetConfig
	generations
    logits_percentage
    scale_factor
    n_most_likely
end

function IPetConfig(;
	generations=3,
	logits_percentage=0.25,
	scale_factor=5,
	n_most_likely=-1)
	
	IPetConfig(generations, logits_percentage, scale_factor, n_most_likely)
end


function train_ipet(;
	ensemble_model_config::WrapperConfig, 
	ensemble_train_config::TrainConfig, 
	ensemble_eval_config::EvalConfig,
	ipet_config::IPetConfig,
	final_model_config::WrapperConfig,
	final_train_config::TrainConfig, 
	final_eval_config::EvalConfig,
	pattern_ids,
	output_dir,
	ensemble_repetitions=3,
	final_repetitions=1, 
	reduction="wmean",
	train_data=nothing,
	unlabeled_data=nothing,
	eval_data=nothing,
	do_train=true,
	do_eval=true,
	no_distillation=false,
	seed=42
	)
	for gen in 1:ipet_config.generations

		gen_output_dir = joinpath(output_dir, "g$gen")


		# Step 1: train an ensemble of models for each of the patterns
		ipet_data_dir = gen > 1 ? joinpath(output_dir, "g$(gen-1)", "next-gen-train-data") : nothing
		train_pet_ensemble(
			model_config=ensemble_model_config,
			train_config=ensemble_train_config,
			eval_config=ensemble_eval_config,
			pattern_ids=pattern_ids,
			output_dir=gen_output_dir,
			ipet_data_dir=ipet_data_dir,
			repetitions=ensemble_repetitions,
			train_data=train_data,
			unlabeled_data=unlabeled_data,
			eval_data=eval_data,
			do_train=do_train,
			do_eval=do_eval,
			save_unlabeled_logits=!no_distillation,
			seed=seed
			)

		# Step 2: Use the model to annotate examples for the next generation
		original_data_size = length(train_data)
		num_new_examples = Int(original_data_size*(ipet_config.scale_factor^(gen))-original_data_size)
		generate_ipet_train_sets(train_data=train_data, unlabeled_data=unlabeled_data, labels=ensemble_model_config.label_list, 
			logits_dir=gen_output_dir, output_dir=joinpath(gen_output_dir, "next-gen-train-data"), reduction=reduction,
			num_new_examples=num_new_examples, logits_percentage=ipet_config.logits_percentage, n_most_likely= (gen==1 ? ipet_config.n_most_likely : -1),
			seed=seed)
	end


	# Step 3: Merge annotations of each model
	logits_dir = joinpath(output_dir, "g$(ipet_config.generations)")
	logits_file = joinpath(logits_dir,"unlabeled_logits.txt")
	merge_logits(logits_dir, logits_file, reduction)
	logits = load_logits(logits_file)
	@assert length(logits) == length(unlabeled_data)

	for (example, example_logits) in zip(unlabeled_data, logits)
		example.logits=example_logits
	end

	# Step 3: Train the final sequence classifier model
	final_model_config.wrapper_type = "sc"
	final_train_config.use_logits = true

	train_classifier(final_model_config, final_train_config, final_eval_config, joinpath(output_dir,"final"), 
		repetitions=final_repetitions, train_data=train_data, unlabeled_data=unlabeled_data, eval_data=eval_data)
end



function train_pet(;
	ensemble_model_config::WrapperConfig, 
	ensemble_train_config::TrainConfig, 
	ensemble_eval_config::EvalConfig,
	final_model_config::WrapperConfig,
	final_train_config::TrainConfig, 
	final_eval_config::EvalConfig,
	pattern_ids,
	output_dir,
	ensemble_repetitions=3,
	final_repetitions=1, 
	reduction="wmean",
	train_data=nothing,
	unlabeled_data=nothing,
	eval_data=nothing,
	do_train=true,
	do_eval=true,
	no_distillation=false,
	seed=42
	)
	# Step 1: train an ensemble of models for each of the patterns
	train_pet_ensemble(
		model_config=ensemble_model_config,
		train_config=ensemble_train_config,
		eval_config=ensemble_eval_config,
		pattern_ids=pattern_ids,
		output_dir=output_dir,
		repetitions=ensemble_repetitions,
		train_data=train_data,
		unlabeled_data=unlabeled_data,
		eval_data=eval_data,
		do_train=do_train,
		do_eval=do_eval,
		save_unlabeled_logits=!no_distillation,
		seed=seed
		)

	# Step 2: Merge annotations of each model
	logits_file = joinpath(output_dir,"unlabeled_logits.txt")
	merge_logits(output_dir, logits_file, reduction)
	logits = load_logits(logits_file)
	@assert length(logits) == length(unlabeled_data)

	for (example, example_logits) in zip(unlabeled_data, logits)
		example.logits=example_logits
	end

	# Step 3: Train the final sequence classifier model
	final_model_config.wrapper_type = "sc"
	final_train_config.use_logits = true

	train_classifier(final_model_config, final_train_config, final_eval_config, joinpath(output_dir,"final"),
		repetitions=final_repetitions, train_data=train_data, unlabeled_data=unlabeled_data, eval_data=eval_data)
end

function train_classifier(
		model_config::WrapperConfig,
		train_config::TrainConfig,
		eval_config::EvalConfig,
		output_dir;
		repetitions=3,
		train_data=nothing,
		unlabeled_data=nothing,
		eval_data=nothing,
		do_train=true,
		do_eval=true,
		seed=42
	)
	train_pet_ensemble(
		model_config=model_config,
		train_config=train_config,
		eval_config=eval_config,
		pattern_ids=[1],
		output_dir=output_dir,
		repetitions=repetitions,
		train_data=train_data,
		unlabeled_data=unlabeled_data,
		eval_data=eval_data,
		do_train=do_train,
		do_eval=do_eval,
		seed=seed
		)
end

function train_pet_ensemble(;
		model_config::WrapperConfig,
		train_config::TrainConfig,
		eval_config::EvalConfig,
		pattern_ids,
		output_dir,
		ipet_data_dir=nothing,
		repetitions=3,
		train_data=nothing,
		unlabeled_data=nothing,
		eval_data=nothing,
		do_train=true,
		do_eval=true,
		save_unlabeled_logits=false,
		seed=42
		)
	
	results = Dict()

	wrapper = nothing

	for pattern_id in pattern_ids
		for iteration in 1:repetitions

			model_config.pattern_id=pattern_id
			results_dict = Dict()

			pattern_iter_output_dir = joinpath(output_dir,"p$pattern_id-i$iteration")
			
			if ispath(pattern_iter_output_dir) && isfile(joinpath(pattern_iter_output_dir, "logits.txt"))
				@warn "Path $pattern_iter_output_dir already exists. Skipping..." maxlog=10
				continue
			end
		
			mkpath(pattern_iter_output_dir)

			wrapper = init_model(model_config)

			# Training
			if do_train
				
				if ipet_data_dir != nothing
					p = joinpath(ipet_data_dir, "p$pattern_id-i$iteration-train.jld2")
				
					ipet_train_data = Knet.load(p, "ipet_train_data")
				
					for example in ipet_train_data
						example.logits = nothing
					end
				else
					ipet_train_data = nothing
				end

				# Temporary evaluation set accuracy logging
				callback = function()
					return evaluate(wrapper, eval_data, eval_config)["scores"]
				end
				
				training_result = train_single_model(wrapper, train_data, train_config, eval_config,ipet_train_data=ipet_train_data, unlabeled_data=unlabeled_data, logging_callback=callback, logging_steps=500)

				for k in keys(training_result)
					results_dict[k] = training_result[k]
				end

				JSON.print(open(joinpath(pattern_iter_output_dir,"results.txt"), "w"), results_dict)
				
				println("Saved trained model at $pattern_iter_output_dir...")

				save(joinpath(pattern_iter_output_dir,"model.jld2"), wrapper)
				Knet.save(joinpath(pattern_iter_output_dir,"train_config.jld2"), "train_config", train_config)
				Knet.save(joinpath(pattern_iter_output_dir,"eval_config.jld2"), "eval_config", eval_config)

				println("Saving complete.")

				if save_unlabeled_logits
					logits = evaluate(wrapper, unlabeled_data, eval_config)["logits"]
					save_logits(joinpath(pattern_iter_output_dir,"logits.txt"), logits)
				end

			end

			# Evaluation
			if do_eval
				println("Starting evaluation...")

				if wrapper == nothing
					wrapper = Knet.load(joinpath(pattern_iter_output_dir,"model.jld2"), "model_wrapper")
				end

				eval_result = evaluate(wrapper, eval_data, eval_config)

				save_predictions(joinpath(pattern_iter_output_dir,"predictions.jsonl"), wrapper, eval_result)
				save_logits(joinpath(pattern_iter_output_dir,"eval_logits.txt"), eval_result["logits"])

				scores = eval_result["scores"]

				println("--- RESULT (pattern_id=$pattern_id, iteration=$iteration ---")
				println(scores)

				results_dict["test_set_after_training"] = scores

				JSON.print(open(joinpath(pattern_iter_output_dir,"results.json"), "w"), results_dict)
				
				for metric in keys(scores)
					if !haskey(results, metric)
						results[metric]=Dict()
					end
					if !haskey(results[metric], pattern_id)
						results[metric][pattern_id] = []
					end
					push!(results[metric][pattern_id], scores[metric])
				end

			end
		end
		# To get GC to flush models from memory
		if wrapper != nothing
			wrapper.model=nothing
		end
		wrapper=nothing
	end

	if do_eval
		println("=== OVERALL RESULTS ===")
		write_results(joinpath(output_dir,"results_test.txt"), results)
	end
	println("=== Ensemble Training Complete ===")
end


function train_single_model(model::TransformerWrapper, train_data, config, eval_config; ipet_train_data=nothing, unlabeled_data=nothing, return_train_set_results=true, logging_callback=nothing, logging_steps=500)
	if ipet_train_data==nothing
		ipet_train_data=[]
	end

	results_dict = Dict()

	if train_data!=nothing && return_train_set_results
		# results_dict["train_set_before_training"] = evaluate(model, train_data, eval_config)["scores"]
		# @show results_dict["train_set_before_training"]
	end

	all_train_data = [train_data..., ipet_train_data...]

	if length(all_train_data)==0 && !config.use_logits
		println("Training method was called without training examples")
	else
		global_step, tr_loss = train(model, all_train_data, config.batch_size,
			num_train_epochs=config.num_train_epochs,
			weight_decay=config.weight_decay,
			learning_rate=config.learning_rate,
			adam_epsilon=config.adam_epsilon,
			warmup_steps=config.warmup_steps,
			max_grad_norm=config.max_grad_norm,
			unlabeled_batch_size=config.unlabeled_batch_size,
			unlabeled_data=unlabeled_data,
			lm_training=config.lm_training,
			use_logits=config.use_logits,
			alpha=config.alpha,
			temperature=config.temperature,
			max_steps=config.max_steps,
			gradient_accumulation_steps=config.gradient_accumulation_steps,
			logging_callback=logging_callback, 
			logging_steps=logging_steps,
			atype=config.atype)

		results_dict["global_step"] = global_step
		results_dict["average_loss"] = tr_loss
	end

	if train_data!=nothing && return_train_set_results
		results_dict["train_set_after_training"] = evaluate(model, train_data, eval_config)["scores"]
	end

	return results_dict
end

function evaluate(model::TransformerWrapper, eval_data, config; priming_data=nothing)
	metrics = config.metrics != nothing ? ["acc"] : config.metrics

	results = eval(model, eval_data, config.batch_size, atype=config.atype)

	predictions = vec((x->x[1]).(argmax(results["logits"], dims=1)))

	# print("preds: $predictions, labels: $(results["labels"]), logits=$(results["logits"])")

	scores = Dict()

	for metric in metrics
		if !haskey(registeredmetrics, metric)
			@error "Metric $metric not implemented"
		else
			scores[metric] = registeredmetrics[metric]()(results["labels"], predictions)
		end
	end

	results["scores"] = scores
	results["predictions"] = predictions
	return results
end

function generate_ipet_train_sets(;train_data, unlabeled_data, labels, logits_dir, output_dir, reduction, num_new_examples, 
	logits_percentage, n_most_likely, seed)
	
	subdirs = iterate(walkdir(logits_dir))[1][2]
	
	mkpath(output_dir)

	@info "Found the following $(length(subdirs)) subdirectories: $(subdirs)"

	@assert train_data!=nothing "No training data provided"
# 
	# TODO: Does this work for ReCoRD?
	# No, but ReCoRD doesn't support iPET in the first place.
	train_examples_per_label = [(label, sum([getLabel(x) == label for x in train_data])) for label in labels]

	multiplier = num_new_examples/length(train_data)

	examples_per_label = [(lbl, Int(epl * multiplier)) for (lbl, epl) in train_examples_per_label]

	@info "Example distribution in the original dataset: $train_examples_per_label"
	@info "Target distribution for the new dataset: $examples_per_label"

	for example in unlabeled_data
		example.logits = nothing
		removeLabel(example)
	end

	logits_lists = Dict()
	rng = MersenneTwister(seed);
	for subdir in subdirs
		results_file = joinpath(logits_dir, subdir, "results.txt")
		logits_file = joinpath(logits_dir, subdir, "logits.txt")
		logits = []

		if !isfile(results_file) || !isfile(logits_file)
			@warn "Skipping subdir $subdir because either results.txt or logits.txt not found."
			continue
		end

		results = JSON.parse(read(open(results_file), String))
		# After training it's almost always 1
		results_train = results["train_set_before_training"]["acc"]
		println("Logits file: $logits_file")
		logits = load_logits(logits_file)

		# println("File $results_file: Score: $results_train, #logits $(length(logits)), #labels = $(length(logits[1]))")
		print("File $results_file: Score: $results_train, #logits $(length(logits)), #labels = ")
		println("$(length(logits[1]))")


		logits = hcat(logits...)

		logits_lists[subdir]=(results_train, logits)
	end

	for subdir in subdirs
		other_logits_lists = []

		[(sd!=subdir ? push!(other_logits_lists, logits_lists[sd]) : nothing) for sd in keys(logits_lists)]

		subdir_train_set = generate_ipet_train_set(
			other_logits_lists, labels=labels, original_data=unlabeled_data, examples_per_label=examples_per_label,
			logits_percentage=logits_percentage, reduction=reduction, n_most_likely=n_most_likely, rng=rng
			)

		@assert all([x.labeled for x in subdir_train_set]) "Found unlabeled instances in the ipet training set."

		Knet.save(joinpath(output_dir, "$subdir-train.jld2"), "ipet_train_data", subdir_train_set)
	end
end

function generate_ipet_train_set(logits_lists; labels, original_data, examples_per_label, logits_percentage, reduction,
	n_most_likely, rng)

	num_logits_lists = Int(round(length(logits_lists) * logits_percentage))

	# Sample logits_percentage percentage of logits_lists
	logits_lists = logits_lists[randperm(rng, length(logits_lists))[1:num_logits_lists]]

	weights = (x->x[1]).(logits_lists)
	logits = cat([reshape(x[2], 1, size(x[2])...) for x in (logits_lists)]..., dims=1)

	if reduction == "mean"
		logits = mean(logits, dims=1)
	else
		logits = sum(weights.*logits, dims=1)./sum(weights)
	end

	logits = reshape(logits, size(logits)[2:end]...)
	logits = softmax(logits, dims=1)

	@assert size(logits, 2) == length(original_data)

	original_data = deepcopy.(original_data)

	for i in 1:size(logits, 2)
		example = original_data[i]
		example.logits = logits[:, i]
		setLabel(example, argmax(example.logits))
	end

	test_set = []

	for (idx, label) in enumerate(labels)
		if n_most_likely <= 0
			examples = [ex for ex in original_data if getLabel(ex) == label]

			while length(examples) < examples_per_label[idx][2]
				examples = [examples..., examples...]
			end
		else
			examples = [(ex.logits[idx], ex_idx, ex) for (ex_idx, ex) in enumerate(original_data)]
			examples = sort(examples, lt = (x, y)->(x[1]>y[1]))
			examples = [ex for (score, ex_idx, ex) in examples[1:n_most_likely]]
		end

		probs = Weights([ex.logits[idx] for ex in examples])

		label_examples = sample(rng, examples, probs, examples_per_label[idx][2]; replace=false)

		test_set = [test_set..., label_examples...]
	end

	return test_set
end





function write_results(path, results)
	f = open(path, "w")

	for metric in keys(results)
		for pattern_id in keys(results[metric])
			avg = mean(results[metric][pattern_id])
			stdv = std(results[metric][pattern_id])

			result_str = "$metric-p$pattern_id: $avg +- $stdv\n"
			print(result_str)
			write(f, result_str)
		end
	end

	for metric in keys(results)
		all_results = [result for pattern_results in values(results[metric]) for result in pattern_results]
		all_mean = mean(all_results)
		all_stdev = std(all_results)
		result_str = "$metric-all-p: $all_mean +- $all_stdev\n"
		print(result_str)
		write(f, result_str)
	end
end

function merge_logits(logits_dir, output_file, reduction)

	subdirs = iterate(walkdir(logits_dir))[1][2]
	all_logits_list = []

	for subdir in subdirs
		results_file = "$logits_dir/$subdir/results.txt"
		logits_file = "$logits_dir/$subdir/logits.txt"
		logits = []

		if !isfile(results_file) || !isfile(logits_file)
			@warn "Skipping subdir $subdir because either results.txt or logits.txt not found."
			continue
		end

		results = JSON.parse(read(open(results_file), String))
		# After training it's almost always 1
		results_train = results["train_set_before_training"]["acc"]

		logits = load_logits(logits_file)

		println("File $results_file: Score: $results_train, #logits $(length(logits)), #labels = $(length(logits[1]))")

		logits = hcat(logits...)

		push!(all_logits_list, (results_train, logits))
	end

	merged_logits = merge_logits_lists(all_logits_list, reduction)
	save_logits(output_file, merged_logits)
end

function merge_logits_lists(logits_lists, reduction="mean")
	weights = (x->x[1]).(logits_lists)
	logits = cat([reshape(x[2], 1, size(x[2])...) for x in (logits_lists)]..., dims=1)

	if reduction == "mean"
		logits = mean(logits, dims=1)
	else
		logits = sum(weights.*logits, dims=1)./sum(weights)
	end

	return reshape(logits, size(logits)[2:end]...)
end
		
function save_predictions(path, wrapper, results)
	predictions_with_idx = []

	if length(wrapper.outputs) > 0
		predictions_with_idx=wrapper.outputs
	else
		inv_label_map = Dict()
		for (a, b) in wrapper.prep.label_map
			inv_label_map[b]=a
		end

		for (idx, prediction_idx) in zip(results["indices"], results["predictions"])
			prediction = inv_label_map[prediction_idx]
			push!(predictions_with_idx, Dict("idx"=>idx, "label"=>prediction))
		end
	end

	f = open(path, "w")
	for line in predictions_with_idx
		JSON.print(f, line)
		write(f, "\n")
	end
	close(f)	
end

function save_logits(path, logits, score=-1)
	f = open(path, "w")
	# To not iterate on the GPU
	logits = Array{Float32}(logits)
	write(f, "$score\n")
	for i in 1:size(logits, 2)
		line = join(logits[:, i], " ")
		write(f, "$line\n")
	end
	close(f)
end

function load_logits(path, getscore=false)
	lines = readlines(path)

	logits = [parse.(Float32, x) for x in split.(lines[2:end])]
	
	if !getscore
		return logits
	else
		score = parse(Float32, lines[1])
		return score, logits
	end
end



