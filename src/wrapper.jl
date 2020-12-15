using ProgressBars: tqdm, set_description
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
	# cache_dir
end

# function WrapperConfig(model_config, model_name_or_path::AbstractString, wrapper_type::AbstractString, task_name, max_seq_length, label_list, pattern_id)
# 	@assert wrapper_type == "mlm" || wrapper_type == "sc"
# 	WrapperConfig(model_config, model_name_or_path, wrapper_type, task_name, max_seq_length, label_list, pattern_id)
# end


mutable struct TransformerWrapper
	wrapper_config
	tokenizer
	model_config
	model
	prep
end

pretrained_initializers = Dict(
	"mlm"=>pretrainedAlbertForMLM,
	"sc"=>pretrainedAlbertForSC
	)


function TransformerWrapper(wrapper_config::WrapperConfig)
	tokenizer = AlbertTokenizer("albert-base-v2")#wrapper_config.model_name_or_path)
	model = pretrained_initializers[wrapper_config.wrapper_type](wrapper_config.model_name_or_path, wrapper_config.model_config, num_labels=length(wrapper_config.label_list))

	prep = preprocessors_map[wrapper_config.wrapper_type](tokenizer, wrapper_config.model_config, wrapper_config.max_seq_length, BoolQPVP(tokenizer, wrapper_config.pattern_id), wrapper_config.label_list)

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
	logging_steps = 50,
	unlabeled_batch_size=8,
	unlabeled_data=nothing,
	lm_training=false,
	use_logits=false,
	alpha=0.8,
	temperature=1,
	max_steps=-1,
	atype=atype()
	)
	
	train_dataset = generate_dataset(model.prep, task_train_data)

	train_idxes = [1:length(train_dataset["idx"])...]

	# Here we create minibatches of indices of batches
	train_batches = minibatch(train_idxes, batch_size, shuffle=true, partial=true)

	unlabeled_dataset = nothing

	if lm_training || use_logits
		@assert unlabeled_data != nothing

		unlabeled_dataset = generate_dataset(model.prep, unlabeled_data)
	end

	if max_steps > 0
		t_total = max_steps
		num_train_epochs = Int(ceil(t_total/length(train_batches))) 
	else
		t_total = length(train_batches) * num_train_epochs #÷ gradient_accumulation_steps
	end


	global_step = 0
	tr_loss, logging_loss = 0, 0
	scheduler = get_linear_scheduler_with_warmup(t_total, warmup_steps)

	# No decay for lnorm and biases. These are the only parameters that are one dimensional.
	wdecay_func = function(w)
		length(size(w))==1 ? 0 : weight_decay
	end

	# Set optimizer of parameters to AdamW (Adam with weight decay)
	for x in Knet.params(model)
		x.opt = AdamW(; lr=learning_rate, eps=adam_epsilon, wdecayfunc=wdecay_func, scheduler=scheduler)
	end

	global_step = 0
	total_loss = 0

	train_iterator = tqdm(1:num_train_epochs)
	for (epoch_num, _) in enumerate(train_iterator)
		epoch_iterator = tqdm(train_batches)
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

			# Switch to model atype
			L = @diff loss_func_map[model.wrapper_config.wrapper_type](model, cur_batch)

			total_loss += value(L)
			set_postfix(epoch_iterator, Loss=@sprintf("%.2f", value(L)))

			for x in params(model)
				update!(x, grad(L, x))
				global_step += 1
			end

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
	atype=atype()
	)
	eval_dataset = generate_dataset(model.prep, eval_data)

	eval_idxes = [1:length(eval_dataset["idx"])...]

	# Here we create minibatches of indices of batches
	eval_batches = minibatch(eval_idxes, batch_size, shuffle=false, partial=true)

	data_iterator = tqdm(eval_batches)

	preds = []
	all_indices, out_label_ids, question_ids = [], [], []

	for batch in data_iterator
		cur_batch = Dict()

		cur_batch["input_ids"] = eval_dataset["input_ids"][:, batch]
		cur_batch["attention_mask"] = eval_dataset["attention_mask"][:, batch]
		cur_batch["token_type_ids"] = eval_dataset["token_type_ids"][:, batch]
		cur_batch["mlm_labels"] = eval_dataset["mlm_labels"][:, batch]

		labels = eval_dataset["labels"][batch]
		indices = eval_dataset["idx"][batch]

		logits = eval_func_map[model.wrapper_config.wrapper_type](model, cur_batch)

		push!(preds,logits)
		push!(out_label_ids,labels)
		push!(all_indices, indices)
		# if haskey(question_ids, "question_idx")
		# 	append!(question_ids, indices)
		# end
	end

	return Dict(
		"indices"=>vcat(all_indices...),
		"logits"=>cat(preds..., dims=2),
		"labels"=>vcat(out_label_ids...),
		)
end


function mlm_loss(wrapper::TransformerWrapper, labeled_batch)

	model_outputs = wrapper.model(
		input_ids=labeled_batch["input_ids"],
		attention_mask=labeled_batch["attention_mask"],
		token_type_ids=labeled_batch["token_type_ids"]
		# logits=labeled_batch[""]
		)

	mlm_labels, labels = labeled_batch["mlm_labels"], labeled_batch["label"]

	prediction_scores = convert_mlm_logits_to_cls_logits(wrapper.prep.pvp, mlm_labels, model_outputs["logits"])

	loss = nll(prediction_scores, labels)
end


function sc_loss(wrapper::TransformerWrapper, labeled_batch, use_logits=false, temperature=1)
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
		logits_predicted, logits_targeted = outputs["logits"], labeled_batch["logits"]
		return distillation_loss(logits_predicted, logits_targeted, temperature)
	end
end

loss_func_map = Dict(
	"mlm"=>mlm_loss,
	"sc"=>sc_loss
	)


# Compute the distillation loss (KL divergence between predictions and targets) as described in the PET paper
function distillation_loss(predictions, targets, temperature=1)
	p = log.(softmax(predictions ./ temperature), dims=1)
	q = softmax(targets ./ temperature, dims=1)
	# KL_DIV function 
	# TODO 
	return sum(p.*(log.(p).-q)) * temperature^2 / size(predictions, 1)
end


function generate_dataset(prep::preprocessor, data, verbose=false)
	# print(prep)
	# print(typeof(prep))
	features = prep.(data)
	if verbose
		for (idx, feature) in enumerate(features[1:5])
			println("=== Example $idx ===")
			println(feature)
		end
	end
	features = Dict(
			"input_ids"=>hcat([feature["input_ids"] for feature in features]...),
			"attention_mask"=>hcat([feature["attention_mask"] for feature in features]...),
			"token_type_ids"=>hcat([feature["token_type_ids"] for feature in features]...),
			"labels"=>[feature["label"] for feature in features],
			"mlm_labels"=>hcat([feature["mlm_labels"] for feature in features]...),
			"logits"=>hcat([feature["logits"] for feature in features]...),
			"idx"=>[feature["idx"] for feature in features],
		)
	return features
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
		)
end

eval_func_map = Dict(
	"mlm"=>mlm_eval,
	"sc"=>sc_eval
	)

