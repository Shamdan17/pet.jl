include("wrapper.jl")
include("metrics.jl")

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
	batch_size=8,
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
	logits_file = "$output_dir/unlabeled_logits.txt"
	merge_logits(output_dir, logits_file, reduction)
	logits = load_logits(logits_file)
	@assert length(logits) == length(unlabeled_data)

	for (example, example_logits) in zip(unlabeled_data, logits)
		example.logits=example_logits
	end

	# Step 3: Train the final sequence classifier model
	final_model_config.wrapper_type = "sc"
	final_train_config.use_logits = true

	train_classifier(final_model_config, final_train_config, final_eval_config, "$output_dir/final", 
		repetitions=final_repetitions, train_data=train_data, unlabeled_data=unlabeled_data)
end

function train_classifier(;
		model_config::WrapperConfig,
		train_config::TrainConfig,
		eval_config::EvalConfig,
		output_dir,
		repetitions=3,
		train_data=nothing,
		unlabeled_data=nothing,
		eval_data=nothing,
		do_train=true,
		do_eval=true,
		seed=42
	)
	train_pet_ensemble(
		model_config=ensemble_model_config,
		train_config=ensemble_train_config,
		eval_config=ensemble_eval_config,
		pattern_ids=[1],
		output_dir=output_dir,
		repetitions=ensemble_repetitions,
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
	Knet.seed!(seed)

	for pattern_id in pattern_ids
		for iteration in 1:repetitions
			model_config.pattern_id=pattern_id
			results_dict = Dict()

			# TODO: Replace with path join function
			pattern_iter_output_dir = "$output_dir/p$pattern_id-i$iteration"
			
			if ispath(pattern_iter_output_dir)
				@warn "Path $pattern_iter_output_dir already exists. Skipping..." maxlog=2
			end
		
			mkpath(pattern_iter_output_dir)

			wrapper = init_model(model_config)

			# Training
			if do_train
				if ipet_data_dir != nothing
					@warn "Not implemented"
				end
				ipet_train_data = nothing

				training_result = train_single_model(wrapper, train_data, train_config, eval_config,ipet_train_data=ipet_train_data, unlabeled_data=unlabeled_data)

				for k in keys(training_result)
					results_dict[k] = training_result[k]
				end

				JSON.print(open("$pattern_iter_output_dir/results.txt", "w"), results_dict)
				
				println("Saved trained model at $pattern_iter_output_dir...")

				save("$pattern_iter_output_dir/model.jld2", wrapper)
				Knet.save("$pattern_iter_output_dir/train_config.jld2", "train_config", train_config)
				Knet.save("$pattern_iter_output_dir/eval_config.jld2", "eval_config", eval_config)

				println("Saving complete.")

				if save_unlabeled_logits
					logits = evaluate(wrapper, unlabeled_data, eval_config)["logits"]
					save_logits("$pattern_iter_output_dir/logits.txt", logits)
				end

			end

			# Evaluation
			if do_eval
				println("Starting evaluation...")

				if wrapper == nothing
					wrapper = load("$pattern_iter_output_dir/model.jld2", "model_wrapper")
				end

				eval_result = evaluate(wrapper, eval_data, eval_config)

				save_predictions("$pattern_iter_output_dir/predictions.jsonl", wrapper, eval_result)
				save_logits("$pattern_iter_output_dir/eval_logits.txt", eval_result["logits"])

				scores = eval_result["scores"]

				println("--- RESULT (pattern_id=$pattern_id, iteration=$iteration ---")
				println(scores)

				results_dict["test_set_after_training"] = scores

				JSON.print(open("$pattern_iter_output_dir/results.json", "w"), results_dict)
				
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
		wrapper.model=nothing
		wrapper=nothing
	end

	if do_eval
		println("=== OVERALL RESULTS ===")
		write_results("$output_dir/results_test.txt", results)
	end
	println("=== Ensemble Training Complete ===")
end


function train_single_model(model::TransformerWrapper, train_data, config, eval_config; ipet_train_data=nothing, unlabeled_data=nothing, return_train_set_results=true)
	if ipet_train_data==nothing
		ipet_train_data=[]
	end

	results_dict = Dict()

	if train_data!=nothing && return_train_set_results
		results_dict["train_set_before_training"] = evaluate(model, train_data, eval_config)["scores"]
	end

	all_train_data = [train_data..., ipet_train_data...]

	if length(all_train_data)==0 && !config.use_logits
		println("Training method was called without training examples")
	else
		global_step, tr_loss = train(model, train_data, config.batch_size,
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
		logits = mean(weights.*logits, dims=1)./sum(weights)
	end

	return reshape(logits, size(logits)[2:end]...)
end
		
function save_predictions(path, wrapper, results)
	predictions_with_idx = []

	inv_label_map = Dict()
	for (a, b) in wrapper.prep.label_map
		inv_label_map[b]=a
	end

	for (idx, prediction_idx) in zip(results["indices"], results["predictions"])
		prediction = inv_label_map[prediction_idx]
		push!(predictions_with_idx, Dict("idx"=>idx, "label"=>prediction))
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



