using ProgressBars: tqdm, set_description, set_postfix
import Knet.Train20: full
import Printf.@sprintf
include("albert/albert_config.jl")
include("albert/albert_model.jl")
include("albert/albert_tokenizer.jl")
include("preprocessors.jl")
include("optimizer.jl")

# This defines a wrapper for transformer models to make training and inference more convenient.
preprocessors_map = Dict(
	"mlm"=>MLMPreprocessor,
	"sc"=>SCPreprocessor,
	)


mutable struct WrapperConfig
	# model_type
	model_config
	model_name_or_path
	wrapper_type
	task_name
	max_seq_length
	label_list
	pattern_id
	atype
	# cache_dir
end

function WrapperConfig(model_config, model_name_or_path::AbstractString, wrapper_type::AbstractString, task_name, max_seq_length, label_list, pattern_id)
	@assert wrapper_type == "mlm" || wrapper_type == "sc"
	WrapperConfig(model_config, model_name_or_path, wrapper_type, task_name, max_seq_length, label_list, pattern_id, atype=atype())
end


mutable struct TransformerWrapper
	wrapper_config
	tokenizer
	model_config
	model
	prep
	# The following field is a very hacky way to handle record outputs
	# I don't like how this works, but for the interest of time
	# I will keep it as is.
	outputs
	TransformerWrapper(wrapper_config,tokenizer,model_config,model,prep)=new(wrapper_config,tokenizer,model_config,model,prep,[])
end

pretrained_initializers = Dict(
	"mlm"=>pretrainedAlbertForMLM,
	"sc"=>pretrainedAlbertForSC
	)


function save(save_dir::AbstractString, wrapper::TransformerWrapper)
	# The tokenizer is a pyobject at it's core
	tmp = wrapper.tokenizer
	wrapper.tokenizer = nothing
	wrapper.prep.tokenizer = nothing
	wrapper.prep.pvp.tokenizer = nothing
	# Knet.save(save_dir, "model_wrapper", wrapper)	
	wrapper.tokenizer = tmp
	wrapper.prep.tokenizer = tmp
	wrapper.prep.pvp.tokenizer = tmp
end

function load(save_dir::AbstractString)
	wrapper = Knet.load(save_dir, "model_wrapper")
	wrapper.tokenizer = AlbertTokenizer("albert-xxlarge-v2")
	wrapper.prep.tokenizer = wrapper.tokenizer
	wrapper.prep.pvp.tokenizer = wrapper.tokenizer
	return wrapper
end

function TransformerWrapper(wrapper_config::WrapperConfig)
	tokenizer = AlbertTokenizer("albert-xxlarge-v2")#wrapper_config.model_name_or_path)
	model = pretrained_initializers[wrapper_config.wrapper_type](wrapper_config.model_name_or_path, wrapper_config.model_config,dtype=wrapper_config.atype, num_labels=length(wrapper_config.label_list))

	prep = preprocessors_map[wrapper_config.wrapper_type](tokenizer, wrapper_config.model_config, wrapper_config.max_seq_length, pvp_map[lowercase(wrapper_config.task_name)](tokenizer, wrapper_config.pattern_id), wrapper_config.label_list)

	TransformerWrapper(wrapper_config, tokenizer, wrapper_config.model_config, model, prep)
end

function train(
	model::TransformerWrapper, 
	task_train_data,
	batch_size;
	num_train_epochs=3,
	weight_decay=0,
	learning_rate=5e-5,
	adam_epsilon=1e-8,
	warmup_steps=0,
	max_grad_norm=1,
	unlabeled_batch_size=8,
	unlabeled_data=nothing,
	lm_training=false,
	use_logits=false,
	alpha=0.8,
	temperature=1,
	max_steps=-1,
	gradient_accumulation_steps=1,
	logging_callback=nothing,
	logging_steps=500,
	)
	
	train_dataset = generate_dataset(model.prep, task_train_data)

	train_idxes = [1:length(train_dataset["idx"])...]

	# Here we create minibatches of indices of batches
	train_batches = minibatch(train_idxes, batch_size, shuffle=true, partial=true)
	# train_batches = minibatch(train_idxes, batch_size, shuffle=false, partial=true)

	unlabeled_dataset = nothing

	if lm_training || use_logits
		@assert unlabeled_data != nothing

		unlabeled_dataset = generate_dataset(model.prep, unlabeled_data)
	end

	if use_logits
		train_dataset=unlabeled_dataset
		train_idxes = [1:length(train_dataset["idx"])...]
		@show batch_size
		train_batches = minibatch(train_idxes, batch_size, shuffle=false, partial=true)
	end

	if max_steps > 0
		t_total = max_steps
		num_train_epochs = Int(ceil(gradient_accumulation_steps*t_total/length(train_batches))) 
	else
		t_total = length(train_batches) * num_train_epochs ÷ gradient_accumulation_steps
	end


	global_step = 0
	tr_loss, logging_loss = 0, 0
	scheduler = get_linear_scheduler_with_warmup(t_total, warmup_steps)

	# No decay for lnorm and biases. These are the only parameters that are one dimensional.
	wdecay_func = function(w)
		length(size(w))==1 ? 0 : weight_decay
	end

	# Used for accumulation of gradients
	gradients = Dict()
	if gradient_accumulation_steps < 1
		gradient_accumulation_steps=1
	end

	# Set optimizer of parameters to AdamW (Adam with weight decay)
	for x in Knet.params(model)
		x.opt = AdamW(; lr=learning_rate, eps=adam_epsilon, wdecayfunc=wdecay_func, scheduler=scheduler)
		if gradient_accumulation_steps>1
			gradients[x] = zero(x)
		end
	end

	global_step = 0
	gradient_accumulation_counter=1

	train_iterator = tqdm(1:num_train_epochs)
	for (epoch_num, _) in enumerate(train_iterator)
		epoch_iterator = tqdm(train_batches, leave=false)
		set_description(epoch_iterator,"Epoch $epoch_num")

		for (step, batch) in enumerate(epoch_iterator)
			cur_batch = Dict()

			cur_batch["input_ids"] = train_dataset["input_ids"][:, batch]
			cur_batch["attention_mask"] = train_dataset["attention_mask"][:, batch]
			cur_batch["token_type_ids"] = train_dataset["token_type_ids"][:, batch]
			cur_batch["labels"] = train_dataset["labels"][batch]
			cur_batch["mlm_labels"] = train_dataset["mlm_labels"][:, batch]
			cur_batch["logits"] = train_dataset["logits"][:, batch]
			cur_batch["idx"] = train_dataset["idx"][batch]
			
			add_input_features_to_batch!(train_dataset, cur_batch, batch, model.wrapper_config.task_name)

			# Switch logits to model atype if not so
			cur_batch["logits"] = cur_batch["logits"] == nothing ? nothing : model.wrapper_config.atype(cur_batch["logits"])


			if model.wrapper_config.wrapper_type=="mlm" && haskey(train_step_helpers, model.wrapper_config.task_name)
				L = @diff train_step_helper(cur_batch, model)
			else
				L = @diff loss_func_map[model.wrapper_config.wrapper_type](model, cur_batch, use_logits=use_logits, temperature=temperature)
			end

			# println("Loss: ", value(L))
			tr_loss += value(L)
			set_postfix(epoch_iterator, Loss=@sprintf("%.3g", value(L)), lr=@sprintf("%.3g", get_last_lr(first(Knet.params(model)).opt)))

			if (global_step+1)%logging_steps== 0 && (step)%gradient_accumulation_steps == 0
				println("Callback: ", logging_callback())
			end

			# If enough gradients accumulated, update, otherwise continue aggregating the gradients
			if (gradient_accumulation_counter)%gradient_accumulation_steps == 0
				for x in Knet.params(model)
					update!(x, gradient_accumulation_steps == 1 ? grad(L, x) : gradients[x]./gradient_accumulation_steps)
					if gradient_accumulation_steps > 1
						gradients[x].=0
					end
				end
				global_step += 1
			else
				for x in Knet.params(model)
					g = full(grad(L, x))
					if g == nothing
						continue
					end
					gradients[x].+=g
				end
			end
			
			gradient_accumulation_counter+=1

			if 0 < max_steps < global_step
				break
			end
		end
		if 0 < max_steps < global_step
			break
		end
	end
	return global_step, (global_step > 0 ? tr_loss/global_step : -1)
end

function eval(
	model::TransformerWrapper, 
	eval_data,
	batch_size=8;
	)

	eval_dataset = generate_dataset(model.prep, eval_data, training=false)

	eval_idxes = [1:length(eval_dataset["idx"])...]

	# Here we create minibatches of indices of batches
	eval_batches = minibatch(eval_idxes, batch_size, shuffle=false, partial=true)

	data_iterator = length(eval_batches) == 1 ? eval_batches : tqdm(eval_batches)

	preds = []
	all_indices, out_label_ids, question_ids = [], [], []

	for batch in data_iterator
		cur_batch = Dict()

		cur_batch["input_ids"] = eval_dataset["input_ids"][:, batch]
		cur_batch["attention_mask"] = eval_dataset["attention_mask"][:, batch]
		cur_batch["token_type_ids"] = eval_dataset["token_type_ids"][:, batch]
		cur_batch["mlm_labels"] = eval_dataset["mlm_labels"][:, batch]

		add_input_features_to_batch!(eval_dataset, cur_batch, batch, lowercase(model.wrapper_config.task_name))

		labels = eval_dataset["labels"][batch]
		indices = eval_dataset["idx"][batch]

		logits = eval_step_helper(cur_batch, model)
		if logits == nothing
			logits = eval_func_map[model.wrapper_config.wrapper_type](model, cur_batch)
		end

		push!(preds,logits)
		push!(out_label_ids,labels)
		push!(all_indices, indices)
		if haskey(eval_dataset, "question_idx")
			push!(question_ids, eval_dataset["question_idx"][batch])
		end
	end

	return Dict{Any, Any}(
		"indices"=>vcat(all_indices...),
		"logits"=>cat(preds..., dims=2),
		"labels"=>vcat(out_label_ids...),
		"question_ids"=>question_ids,
		)
end


function mlm_loss(wrapper::TransformerWrapper, labeled_batch; o...)
	model_outputs = wrapper.model(
		input_ids=labeled_batch["input_ids"],
		attention_mask=labeled_batch["attention_mask"],
		# For some reason, token_type_ids are not used despite albert accepting them as an input.
		# token_type_ids=labeled_batch["token_type_ids"]
		)

	mlm_labels, labels = labeled_batch["mlm_labels"], labeled_batch["labels"]
	
	prediction_scores = convert_mlm_logits_to_cls_logits(wrapper.prep.pvp, mlm_labels, model_outputs["logits"])
	
	lss = nll(prediction_scores, labels)
end

# # Cross Entropy Loss
# function CELoss(preds, labels)
# 	oh = oftype(preds, [i==j ? 1 : 0 for i in 1:size(preds, 1), j in 1:size(preds, 1)][:, labels])
# 	# for i in 1:size(preds, 1)
# 	# 	oh[i, i].+=1
# 	# end
# 	mean(.-sum(oh .* logsoftmax(preds; dims=1); dims=1))
# end


function sc_loss(wrapper::TransformerWrapper, labeled_batch; use_logits=false, temperature=1)
	model_outputs = wrapper.model(
		input_ids=labeled_batch["input_ids"],
		labels = use_logits ? nothing : labeled_batch["labels"], 
		attention_mask=labeled_batch["attention_mask"],
		token_type_ids=labeled_batch["token_type_ids"]
		# logits=labeled_batch[""]
		)

	if !use_logits
		return model_outputs
	else
		logits_predicted, logits_targeted = model_outputs["logits"], labeled_batch["logits"]
		return distillation_loss(logits_predicted, logits_targeted, temperature)
	end
end

loss_func_map = Dict(
	"mlm"=>mlm_loss,
	"sc"=>sc_loss
	)


# Compute the distillation loss (KL divergence between predictions and targets) as described in the PET paper
function distillation_loss(predictions, targets, temperature=1)
	p = logsoftmax(predictions ./ temperature, dims=1)
	q = softmax(targets ./ temperature, dims=1)
	# KL_DIV function 
	# println(size(q), size(p))
	return sum(q.*(log.(q).-p)) * temperature^2 / size(predictions, 2)
end


function generate_dataset(prep::preprocessor, data; verbose=false, training=true)
	# print(prep)
	# print(typeof(prep))
	features = prep.(data, training=training)
	# Add special task dependent features if any
	add_special_input_features!.(data, features, tokenizer=prep.tokenizer)

	if verbose
		for (idx, feature) in enumerate(features[1:5])
			println("=== Example $idx ===")
			println(feature)
		end
	end
	features_dict = Dict{Any,Any}(
			"input_ids"=>hcat([feature["input_ids"] for feature in features]...),
			"attention_mask"=>hcat([feature["attention_mask"] for feature in features]...),
			"token_type_ids"=>hcat([feature["token_type_ids"] for feature in features]...),
			"labels"=>[feature["label"] for feature in features],
			"mlm_labels"=>hcat([feature["mlm_labels"] for feature in features]...),
			"logits"=>hcat([feature["logits"] for feature in features]...),
			"idx"=>[feature["idx"] for feature in features],
		)
	# Add special task dependent features if any
	add_features_to_dict_helper!(features, features_dict, typeof(data[1]))
	
	return features_dict
end

function mlm_eval(wrapper::TransformerWrapper, labeled_batch)
	model_outputs = wrapper.model(
		input_ids=labeled_batch["input_ids"],
		attention_mask=labeled_batch["attention_mask"],
		token_type_ids=labeled_batch["token_type_ids"]
		)
	convert_mlm_logits_to_cls_logits(wrapper.prep.pvp, labeled_batch["mlm_labels"], model_outputs["logits"])
end

function sc_eval(wrapper::TransformerWrapper, labeled_batch)
	model_outputs = wrapper.model(
		input_ids=labeled_batch["input_ids"],
		attention_mask=labeled_batch["attention_mask"],
		token_type_ids=labeled_batch["token_type_ids"]
		)["logits"]
end

eval_func_map = Dict(
	"mlm"=>mlm_eval,
	"sc"=>sc_eval
	)

