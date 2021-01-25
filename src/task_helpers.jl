# include("wrapper.jl")
using Knet.Ops20:findindices

function requires_helper(task_name)
	haskey(train_step_helpers, task_name)
end


function train_step_helper(batch, wrapper; o...)
	if haskey(train_step_helpers, wrapper.wrapper_config.task_name)
		train_step_helpers[wrapper.wrapper_config.task_name](batch, wrapper, o...)
	else
		nothing
	end
end


function train_step_wsc(batch, wrapper; o...)
	target_token_ids = hcat(batch["target_token_ids"]...)
	loss = wrapper.model(
		input_ids=batch["input_ids"],
		labels=target_token_ids,
		attention_mask=batch["attention_mask"],
		# For some reason, token_type_ids are not used despite albert accepting them as an input.
		# token_type_ids=batch["token_type_ids"]
		)
	loss
end

function fake_size(x)
	size(x)
end

function train_step_copa(batch, wrapper; o...)
	sz = fake_size(batch["labels"])
	mask = reshape(batch["labels"], 1, sz...).-1
	
	# Should be max_seq_len, B
	choice_1_token_ids = batch["choice_1_token_ids"]
	choice_1_token_ids = hcat(choice_1_token_ids...)
	choice_1_token_ids[choice_1_token_ids.==-100].*=0

	choice_2_token_ids = batch["choice_2_token_ids"]
	choice_2_token_ids = hcat(choice_2_token_ids...)
	choice_2_token_ids[choice_2_token_ids.==-100].*=0

	correct_targets = choice_1_token_ids.*(1 .- mask) + choice_2_token_ids.*mask
	wrong_targets = choice_2_token_ids.*(1 .- mask) + choice_1_token_ids.*mask


	# V, max_len, B
	prediction_scores = wrapper.model(
		input_ids=batch["input_ids"],
		attention_mask=batch["attention_mask"],
		# For some reason, token_type_ids are not used despite albert accepting them as an input.
		# token_type_ids=batch["token_type_ids"]
		)["logits"]


	# Original hinge loss implementation
	loss_correct_labels = nll(prediction_scores, correct_targets)
	loss_wrong_labels = nll(prediction_scores, wrong_targets)
	relu(1+loss_correct_labels-loss_wrong_labels)
	# Better (More correct?) hinge loss implementation
	# loss_correct_labels = [nll(prediction_scores[:,:,b], correct_targets[:,b]) for b in 1:size(choice_1_token_ids)[end]] 
	# loss_wrong_labels = [nll(prediction_scores[:,:,b], wrong_targets[:,b]) for b in 1:size(choice_1_token_ids)[end]]
	# loss = 1 .+ loss_correct_labels .- loss_wrong_labels
	# Hinge loss -> Only consider positive values
	# loss = relu.(loss)
	# sum(loss)/size(choice_1_token_ids)[end]
end

function train_step_record(batch, wrapper; o...)
	prediction_scores = wrapper.model(
		input_ids=batch["input_ids"],
		attention_mask=batch["attention_mask"],
		# For some reason, token_type_ids are not used despite albert accepting them as an input.
		# token_type_ids=batch["token_type_ids"]
		)["logits"]

	# TODO: Make prediction_scores of shape :, VocabSize (same as original)
	#                                    or Vocabsize, : (Julia conventions)
	# prediction_scores = reshape(prediction_scores, V, :)
	
	# max_seq_len * max_num_candidates * batch_size
	all_candidate_token_ids = batch["candidate_token_ids"]
	all_candidate_token_ids = cat([hcat(x...) for x in all_candidate_token_ids]..., dims=3)

	# max_num_candidates * batch_size 
	all_candidate_labels = batch["candidate_labels"]
	all_candidate_labels = hcat(all_candidate_labels...)

	# println(size(all_candidate_labels))
	# println(all_candidate_labels)

	# Put max_num_candidates first
	# all_candidate_token_ids = permutedims(all_candidate_token_ids, [2, 1, 3])
	# all_candidate_labels = permutedims(all_candidate_labels, [2, 1])

	# First candidate = correct, rest = wrong
	total_loss = 0
	all_candidate_token_ids[all_candidate_token_ids.==-100].*=0
	for b in 1:size(all_candidate_token_ids)[end]
		loss_correct_label = nll(prediction_scores[:,:,b], reshape(all_candidate_token_ids[:,1,b], :))

		# Compute hinge loss
		# TODO: candidate_labels not used?
		# Hacky implementation, first candidate is the correct one rest are wrong.

		for idx in 2:size(all_candidate_labels, 1)
			candidate_token_ids = all_candidate_token_ids[:, idx, b]
			candidate_labels = all_candidate_labels[idx, b]

			if candidate_labels == -100
				continue
			end

			# loss_wrong_label = nllnoreduce(prediction_scores, reshape(candidate_token_ids, :), mask_val=1e7)
			loss_wrong_label = nll(prediction_scores[:,:,b], reshape(candidate_token_ids, :))

			hinge_loss = 1 .+ loss_correct_label .- loss_wrong_label
			# Hinge loss we discard anything less than 0, which should be equivalent to relu
			total_loss += sum(relu.(hinge_loss))
		end
	end

	total_loss
end

# function nllnoreduce(scores,labels::AbstractArray{<:Integer}; mask_val=-1e7)
#     indices = [CartesianIndex(x==-100 ? 1 : x, i) for (i, x) in enumerate(labels)]
#     lp = logsoftmax(scores,dims=1)[indices]
# 	mask = zero(lp)
# 	mask[labels.==-100]=mask_val
# 	lp = lp .+ mask_val
#     lp ./ (-(length(lp)-sum(mask.==mask_val)))
# end

train_step_helpers = Dict(
	"record"=>train_step_record,
	"copa"=>train_step_copa,
	"wsc"=>train_step_wsc,
)



function eval_step_helper(batch, wrapper; o...)
	if haskey(eval_step_helpers, wrapper.wrapper_config.task_name)
		eval_step_helpers[wrapper.wrapper_config.task_name](batch, wrapper, o...)
	else
		nothing
	end
end

function eval_step_wsc(batch, wrapper; o...)
	if wrapper.wrapper_config.wrapper_type != "mlm"
		return nothing
	end

	@assert size(batch["input_ids"])[end]==1 "eval_step_wsc() only implemented for batchsize=1"
	
	input_ids = deepcopy(batch["input_ids"])

	orig_mask_positions = [idx for (idx, input_id) in enumerate(input_ids[:,1]) if (input_id == wrapper.tokenizer.mask_token_id)]

	while true
		mask_positions = [idx for (idx, input_id) in enumerate(input_ids[:,1]) if (input_id == wrapper.tokenizer.mask_token_id)]

		if length(mask_positions) == 0
			output_ids = [input_ids[idx, 1] for idx in orig_mask_positions if !(input_ids[idx, 1] in wrapper.tokenizer.all_special_ids)]
			output_actual = decode(wrapper.tokenizer, output_ids)

			output_expected = batch["id_to_target"][batch["target_id"][1]]

			# Transform both outputs as described in T5 paper
			output_actual = strip(lowercase(output_actual))
			output_actual = split(output_actual, x->!isletter(x), keepempty=false)
			output_expected = strip(lowercase(output_expected))
			output_expected = split(output_expected, x->!isletter(x), keepempty=false)
			if all(x in output_actual for x in output_expected) || all(x in output_expected for x in output_actual)
				return [0, 1]
			else
				return [1, 0]
			end
		end

		ntl = wrapper.model(
			input_ids=input_ids,
			attention_mask=batch["attention_mask"],
			# For some reason, token_type_ids are not used despite albert accepting them as an input.
			# token_type_ids=batch["token_type_ids"]
		)["logits"]

		ntl = softmax(ntl, dims=1)
		
		@assert length(size(ntl))==3
		@assert size(ntl, 3)==1

		most_confident = ()
		most_confident_score = -1

		for mask_position in mask_positions
			next_token_logits = ntl[:, mask_position, 1]
			top_token_id = argmax(next_token_logits)
			top_score = next_token_logits[top_token_id]

			if top_score > most_confident_score
				most_confident_score = top_score
				most_confident = (mask_position, top_token_id)
			end
		end
		# @assert input_ids[most_confident[1], 1] == tokenizer.mask_token_id
		input_ids[most_confident[1], 1] = most_confident[2]
	end
end

function eval_step_copa(batch, wrapper; decoding_strategy="max_first")
	if wrapper.wrapper_config.wrapper_type != "mlm"
		return nothing
	end
	@assert size(batch["input_ids"])[end]==1 "eval_step_copa() only implemented for batchsize=1"
	log_probs = []
	for choice in ["choice_1", "choice_2"]
		labels = batch["$(choice)_token_ids"]
		log_prob = get_choice_log_probability(wrapper, batch, labels, decoding_strategy=decoding_strategy)
		push!(log_probs, log_prob)
	end
	log_probs
end

function get_choice_log_probability(wrapper, batch, target_sequence; decoding_strategy="max_first")
	@assert decoding_strategy == "max_first" "Only max_first decoding strategy supported. Found $decoding_strategy"
	target_sequence=target_sequence[1]
	# Adjust the number of masks

	num_masks = sum(1 for tokid in target_sequence if tokid>0)
	input_ids = trim_input_ids(input_ids=batch["input_ids"], num_masks=num_masks, pad_token_id=wrapper.tokenizer.pad_token_id, mask_token_id=wrapper.tokenizer.mask_token_id)

	log_probabilities = []

	while true
		masks = [(idx, tokid) for (idx, tokid) in enumerate(target_sequence) if tokid>0]
		if length(masks)==0
			break
		end

		outputs = wrapper.model(input_ids)["logits"]

		ntl = softmax(outputs, dims=1)

		mask_pos, masked_id = nothing, nothing
		highest_prob = -1

		for (m_pos, m_id) in masks
			m_prob = ntl[m_id,m_pos]
			if m_prob > highest_prob
				highest_prob = m_prob
				mask_pos, masked_id = m_pos, m_id
			end
		end

		push!(log_probabilities, log(highest_prob))
		input_ids[mask_pos, 1] = masked_id

		target_sequence[mask_pos] = -100
	end


	return sum(log_probabilities)
end

function eval_step_record(batch, wrapper; batchsize=16, o...)
	@assert size(batch["input_ids"])[end]==1 "eval_step_record() only implemented for batchsize=1"
	best_choice_correct, best_choice, max_prob = false, nothing, -1e9	

	question_idx = batch["question_idx"][1]
	output_line = Dict{Any, Any}(
		"idx"=>question_idx,
		"choices"=>Dict()
		)

	# max_seq_len * max_num_candidates * batch_size
	all_candidate_token_ids = batch["candidate_token_ids"]
	all_candidate_token_ids = cat([hcat(x...) for x in all_candidate_token_ids]..., dims=3)
	
	# Group choices by length
	choices_grouped_by_length = Dict()
	# @show batch["candidate_labels"]

	N = length(batch["candidate_labels"][1])
	# @show N
	for idx in 1:N
		choice_ids = all_candidate_token_ids[:, idx, 1]
		label = batch["candidate_labels"][1][idx]
		if label < 0
			continue
		end
		num_masks = sum([1 for x in choice_ids if x != -100])
		choice = batch["original_choices"][question_idx][idx]

		choices_grouped_by_length[num_masks] = push!(get(choices_grouped_by_length, num_masks, []), (choice, choice_ids, label))
	end

	input_ids = Dict()
	initial_outputs = Dict()
	# @show keys(choices_grouped_by_length)
	# @show choices_grouped_by_length
	for num_masks in keys(choices_grouped_by_length)
		# Modify the input ids to contain the correct number of masks
		input_ids[num_masks] = trim_input_ids(input_ids=batch["input_ids"], num_masks=num_masks,
			pad_token_id=wrapper.tokenizer.pad_token_id, mask_token_id=wrapper.tokenizer.mask_token_id)

		initial_outputs[num_masks] = wrapper.model(input_ids[num_masks])["logits"]
	end
	# @show choices_grouped_by_length
	# exit(0)
	for num_masks in keys(choices_grouped_by_length)
		choices_with_labels = choices_grouped_by_length[num_masks]
		# @show choices_with_labels
		N = length(choices_with_labels)		

		for batch in minibatch(choices_with_labels, batchsize, partial=true)
			batch_input_ids = reshape(input_ids[num_masks], :, 1).*ones(Int, 1, length(batch)) # -> seq_len, B
			# @show size(batch_input_ids)
			# @show size(input_ids[num_masks])
			choice_ids = [choice_id for (choice, choice_id, label) in batch]

			probs = get_choice_probabilities_batched(wrapper, choice_ids, batch_input_ids, initial_outputs[num_masks], decoding_strategy="max_first")

			for (idx, (choice, choice_ids, label)) in enumerate(batch)
				prob = probs[idx]
				output_line["choices"][choice] = prob

				# @show choice, label, max_prob
				# @show prob
				if prob > max_prob
					# @show "got in loop"
					best_choice_correct=(label==1)
					max_prob=prob
				end
			end
		end
	end


	# @show "SLOW implementation"
	# @show output_line
	# @show best_choice_correct
	# @show "SLOW implementation done"
	push!(wrapper.outputs,output_line)
	# exit(0)
	if best_choice_correct
		# @show "Correct"
		return [0, 1]
	else
		return [1, 0]
	end
end

function get_choice_probabilities_batched(wrapper, target_sequences, input_ids, initial_output; decoding_strategy="max_first")
	log_probabilities = Dict()
	first_call = true
	@assert decoding_strategy == "max_first" "Only max_first decoding strategy supported. Found $decoding_strategy"

	while true
		masks = Dict(batch_idx=>[(idx, tok) for (idx, tok) in enumerate(target_sequences[batch_idx]) if tok>=0] for batch_idx in 1:size(target_sequences, 1))

		if length(masks[1])==0
			break
		end

		if first_call
			outputs = initial_output
		else
			outputs = wrapper.model(input_ids)["logits"]
		end

		next_token_logits = softmax(outputs, dims=1)
		next_token_logits = Array{Float32}(next_token_logits)

		for batch_idx in 1:size(target_sequences, 1)
			ntl = first_call ? next_token_logits[:,:,1] : next_token_logits[:, :, batch_idx]

			mask_pos, masked_id = nothing, nothing
			highest_prob = -1

			for (m_pos, m_id) in masks[batch_idx]
				m_prob = ntl[m_id,m_pos]
				if m_prob > highest_prob
					highest_prob = m_prob
					mask_pos, masked_id = m_pos, m_id
				end
			end

			log_probabilities[batch_idx] = push!(get(log_probabilities, batch_idx, []), log(ntl[masked_id, mask_pos]))
			input_ids[mask_pos, batch_idx] = masked_id
			target_sequences[batch_idx][mask_pos] = -100
		end

		first_call = false
	end

	return Dict(batch_idx=>sum(log_prob for log_prob in log_probabilities[batch_idx]) for batch_idx in 1:size(target_sequences, 1))
end


function eval_step_record_fast(batch, wrapper; batchsize=16, o...)
	@assert size(batch["input_ids"])[end]==1 "eval_step_record() only implemented for batchsize=1"
	best_choice_correct, best_choice, max_prob = false, nothing, -1e9	

	question_idx = batch["question_idx"][1]
	output_line = Dict{Any, Any}(
		"idx"=>question_idx,
		"choices"=>Dict()
		)

	# max_seq_len * max_num_candidates * batch_size
	all_candidate_token_ids = batch["candidate_token_ids"]
	all_candidate_token_ids = cat([hcat(x...) for x in all_candidate_token_ids]..., dims=3)
	
	# # Group choices by length
	# choices_grouped_by_length = Dict()
	# @show batch["candidate_labels"]
	all_choices = []
	max_num_masks = 0

	N = length(batch["candidate_labels"][1])
	# @show N
	for idx in 1:N
		choice_ids = all_candidate_token_ids[:, idx, 1]
		label = batch["candidate_labels"][1][idx]
		if label < 0
			continue
		end
		num_masks = sum([1 for x in choice_ids if x != -100])
		choice = batch["original_choices"][question_idx][idx]
		if num_masks > max_num_masks
			max_num_masks = num_masks
		end
		push!(all_choices, (choice, choice_ids, label, num_masks))
	end

	# Sort by length, descending
	sort!(all_choices,lt=(x,y)->x[4]>y[4])

	input_ids = []
	attn_masks = []
	# Get all sizes
	all_sizes = Set([x[4] for x in all_choices])
	initial_output_index = Dict()
	# Prepare initial input
	initial_input = []
	initial_attn = []
	max_len = nothing
	for size in sort(collect(all_sizes), rev=true)
		cur_inp = trim_input_ids(input_ids=batch["input_ids"], num_masks=size,
			pad_token_id=wrapper.tokenizer.pad_token_id, mask_token_id=wrapper.tokenizer.mask_token_id)
		cur_atn = [1 for i in 1:length(cur_inp)]
		pad_cnt = max_num_masks - size
		if pad_cnt>0
			cur_inp = [cur_inp..., [wrapper.tokenizer.pad_token_id for i in 1:pad_cnt]...]
			cur_atn = [cur_atn..., [0 for i in 1:pad_cnt]...]
		end
		push!(initial_input, cur_inp)
		push!(initial_attn, cur_atn)
		# Add the index of this size to the dict
		initial_output_index[size] = length(initial_input)
	end

	# Max into matrix
	initial_input = hcat(initial_input...)
	initial_attn = hcat(initial_attn...)

	# First forward pass for all sizes
	# Size: Vocab x max_len x num_unique_masks
	initial_outputs = wrapper.model(
		input_ids = initial_input,
		attention_mask = initial_attn,
		)["logits"]


	for batch in minibatch(all_choices, batchsize, shuffle=false, partial=true)
		batch_indices = [initial_output_index[x[4]] for x in batch]
		# Vocab x max_len x bs
		batch_initial_output = initial_outputs[:,:,batch_indices]
		# max_len x bs
		batch_inp_ids = deepcopy(initial_input[:, batch_indices])
		batch_attns = initial_attn[:, batch_indices]

		choice_ids = [choice_id for (choice, choice_id, label) in batch]

		probs = get_choice_probabilities_batched_fast(wrapper, choice_ids, batch_inp_ids, batch_attns, batch_initial_output, decoding_strategy="max_first")

		for (idx, (choice, choice_ids, label)) in enumerate(batch)
			prob = probs[idx]
			output_line["choices"][choice] = prob

			# @show choice, label, max_prob
			# @show prob
			if prob > max_prob
				# @show "got in loop"
				best_choice_correct=(label==1)
				max_prob=prob
			end
		end
	end

	# @show "FAST implementation"
	# @show output_line
	# @show best_choice_correct
	# @show "FAST implementation done"
	push!(wrapper.outputs,output_line)
	# exit(f0)
	if best_choice_correct
		# @show "Correct"
		return [0, 1]
	else
		return [1, 0]
	end
end

function get_choice_probabilities_batched_fast(wrapper, target_sequences, input_ids,attention_masks, initial_output; decoding_strategy="max_first")
	log_probabilities = Dict()
	@assert decoding_strategy == "max_first" "Only max_first decoding strategy supported. Found $decoding_strategy"
	
	first_call = true

	while true
		masks = [[(idx, tok) for (idx, tok) in enumerate(target_sequences[batch_idx]) if tok>=0] for batch_idx in 1:size(target_sequences, 1)]
		# Skip the ones that are done
		masks = [x for x in masks if length(x)>0]

		# If no more masks left then we're done 
		if length(masks)==0
			break
		end

		first_mask_index = minimum(x[1] for lst in masks for x in lst)
		last_mask_index = maximum(x[1] for lst in masks for x in lst)
		input_ids = input_ids[:,1:length(masks)]
		attention_masks = attention_masks[:,1:length(masks)]

		if first_call
			outputs = initial_output
		else
			outputs = wrapper.model(
				input_ids=input_ids,
				attention_mask=attention_masks
				)["logits"]
		end

		next_token_logits = softmax(outputs[:,first_mask_index:last_mask_index, :], dims=1)
		next_token_logits = Array{Float32}(next_token_logits)

		for batch_idx in 1:length(masks)
			ntl = next_token_logits[:, :, batch_idx]

			mask_pos, masked_id = nothing, nothing
			highest_prob = -1

			for (m_pos, m_id) in masks[batch_idx]
				m_prob = ntl[m_id,m_pos-first_mask_index+1]
				if m_prob > highest_prob
					highest_prob = m_prob
					mask_pos, masked_id = m_pos, m_id
				end
			end

			log_probabilities[batch_idx] = push!(get(log_probabilities, batch_idx, []), log(ntl[masked_id, mask_pos-first_mask_index+1]))
			input_ids[mask_pos, batch_idx] = masked_id
			target_sequences[batch_idx][mask_pos] = -100
		end

		first_call = false
	end

	return Dict(batch_idx=>sum(log_prob for log_prob in log_probabilities[batch_idx]) for batch_idx in 1:size(target_sequences, 1))
end


eval_step_helpers = Dict(
	# "record"=>eval_step_record,
	"record"=>eval_step_record_fast,
	"copa"=>eval_step_copa,
	"wsc"=>eval_step_wsc,
)


# For unsupported types
function add_special_input_features!(input_example::datum, input_features; o...)
	return
end

# MultiRC
function add_special_input_features!(input_example::MultiRC, input_features; o...)
	input_features["question_idx"] = input_example.questions[1].idx
end

# COPA
function add_special_input_features!(input_example::COPA, input_features; tokenizer=nothing, o...)
	mask_start = indexin(tokenizer.mask_token_id, input_features["input_ids"])[1]
	# If no mask -> Must be sc, nothing to to so return.
	mask_start == nothing && return

	choices = Dict()
    choice1 = rstrip(ispunct, "$(lowercase(input_example.choice1[1]))$(input_example.choice1[2:end])")
    choice2 = rstrip(ispunct, "$(lowercase(input_example.choice2[1]))$(input_example.choice2[2:end])")
    choices["choice_1"]=choice1
	choices["choice_2"]=choice2
	

	for choice in keys(choices)
		choice_token_ids = tokenizer(choices[choice], add_special_tokens=false)["input_ids"]
		mask_end = mask_start + length(choice_token_ids)-1
		input_features["$(choice)_token_ids"] = [-100 for i in 1:length(input_features["input_ids"])]
		input_features["$(choice)_token_ids"][mask_start:mask_end].=choice_token_ids
	end
end

# WSC
function add_special_input_features!(input_example::WSC, input_features; tokenizer=nothing, o...)
	mask_start = indexin(tokenizer.mask_token_id, input_features["input_ids"])[1]
	# If no mask -> Must be sc, nothing to to so return.
	mask_start == nothing && return 

	num_masks = sum(input_features["input_ids"].==tokenizer.mask_token_id)
	mask_end = mask_start + num_masks - 1

	target = input_example.entity
	input_features["target"] = target
	target_token_ids = tokenizer(target, add_special_tokens=false)["input_ids"]
	input_features["target_token_ids"] = [0 for i in 1:length(input_features["input_ids"])]

	# Add pad tokens to fit num_masks, so that the model predicts pad tokens for unneeded tokens
	target_token_ids = [target_token_ids..., [tokenizer.pad_token_id for i in 1:(num_masks-length(target_token_ids))]...]

	input_features["target_token_ids"][mask_start:mask_end]=target_token_ids
end

# Record
function add_special_input_features!(input_example::ReCoRD, input_features; tokenizer=nothing, o...)
	@assert length(input_example.qas)==1 "More than one question found. Please flatten input using the flatten function."
	mask_start = indexin(tokenizer.mask_token_id, input_features["input_ids"])[1]
	
    # Get unique only
	choices = [e.text for e in input_example.entities]
	question_idx = input_example.qas[1].idx

	input_features["candidate_token_ids"] = []
	input_features["candidate_labels"] = []
	input_features["question_idx"] = question_idx

	input_features["original_choices"] = []

	for (idx, choice_text) in enumerate(choices)
		choices_token_ids = tokenizer(choice_text, add_special_tokens=false)["input_ids"]
		choice_label = (!(length(input_example.qas[1].answers)==0) && choice_text in [x.text for x in input_example.qas[1].answers]) ? 1 : 0

		mask_end = mask_start + length(choices_token_ids)-1
		candidate_token_ids = [-100 for i in 1:length(input_features["input_ids"])]
		candidate_token_ids[mask_start:mask_end].=choices_token_ids

		push!(input_features["candidate_token_ids"], candidate_token_ids)  
		push!(input_features["candidate_labels"], choice_label)  
		push!(input_features["original_choices"], choice_text)
	end
end


function add_features_to_dict_helper!(features, feature_dict, datatype; o...)
	if haskey(add_features_to_dict_helpers, datatype)
		add_features_to_dict_helpers[datatype](features, feature_dict, o...)
	else
		nothing
	end
end

# MultiRC
function add_features_to_mrc_dict(features, feature_dict; o...)
	feature_dict["question_idx"] = [f["question_idx"] for f in features]
end

# COPA
function add_features_to_copa_dict(features, feature_dict; o...)
	# Return if not keys -> Incompatible model type, don't need to do anything
	!haskey(features[1], "choice_1_token_ids") && return
	feature_dict["choice_1_token_ids"] = [f["choice_1_token_ids"] for f in features]
	feature_dict["choice_2_token_ids"] = [f["choice_2_token_ids"] for f in features]
end

# WSC
function add_features_to_wsc_dict(features, feature_dict; o...)
	# Return if not has keys -> Incompatible model type, don't need to do anything
	!haskey(features[1], "target_token_ids") && return
	# feature_dict["target_id"] = [idx for (idx, f) in enumerate(features)]
	feature_dict["target"] = [f["target"] for f in features]
	feature_dict["target_token_ids"] = [f["target_token_ids"] for f in features]
end


# ReCoRD
function add_features_to_record_dict(features, feature_dict; o...)
	# Apply padding if necessary
	max_num_candidates = maximum(length(f["candidate_token_ids"]) for f in features)

	for feature in features
		while length(feature["candidate_token_ids"]) < max_num_candidates
			push!(feature["candidate_token_ids"], [-100 for i in 1:length(feature["input_ids"])])
			push!(feature["candidate_labels"], -100)
		end
	end
	feature_dict["original_choices"] = Dict()
	for f in features
		q_idx = f["question_idx"]
		original_choices = f["original_choices"]
		feature_dict["original_choices"][q_idx]=original_choices
	end
	feature_dict["question_idx"] = [f["question_idx"] for f in features]
	feature_dict["candidate_token_ids"] = [f["candidate_token_ids"] for f in features]
	feature_dict["candidate_labels"] = [f["candidate_labels"] for f in features]
end


add_features_to_dict_helpers = Dict(
	MultiRC=>add_features_to_mrc_dict,
	COPA=>add_features_to_copa_dict,
	ReCoRD=>add_features_to_record_dict,
	WSC=>add_features_to_wsc_dict,
)


# For unsupported types
function get_sc_inputs(x, sc_preprocessor)
	@warn "Unsupported type: $(typeof(x))"
	nothing, nothing
end

function get_sc_inputs(x::WiC, sc_preprocessor)
	tokenizer = sc_preprocessor.tokenizer
	text_a = "$(x.word): $(x.sentence1)"
	text_b = x.sentence2
	# tokens_a = tokenizer(text_a, add_special_tokens=false)
	# input_ids = build_inputs_with_special_tokens(pvp.tokenizer, tokens_a, tokens_b)
	output = encode_plus(tokenizer, text_a, text_b, add_special_tokens=true, max_length=sc_preprocessor.max_seq_length, truncation=true)
	return output["input_ids"], output["token_type_ids"]
end

function get_sc_inputs(x::BoolQ, sc_preprocessor)
	tokenizer = sc_preprocessor.tokenizer
	text_a = x.passage
	text_b = x.question
	output = encode_plus(tokenizer, text_a, text_b, add_special_tokens=true, max_length=sc_preprocessor.max_seq_length, truncation=true)
	return output["input_ids"], output["token_type_ids"]
end

function get_sc_inputs(x::CB, sc_preprocessor)
	tokenizer = sc_preprocessor.tokenizer
	text_a = x.premise
	text_b = x.hypothesis
	output = encode_plus(tokenizer, text_a, text_b, add_special_tokens=true, max_length=sc_preprocessor.max_seq_length, truncation=true)
	return output["input_ids"], output["token_type_ids"]
end

function get_sc_inputs(x::RTE, sc_preprocessor)
	tokenizer = sc_preprocessor.tokenizer
	text_a = x.premise
	text_b = x.hypothesis
	output = encode_plus(tokenizer, text_a, text_b, add_special_tokens=true, max_length=sc_preprocessor.max_seq_length, truncation=true)
	return output["input_ids"], output["token_type_ids"]
end

function get_sc_inputs(x::MultiRC, sc_preprocessor)
	@assert length(x.questions)==1 && length(x.questions[1].options)==1 "MultiRC instances with multiple questions/answers are not supported. Please use the flatten function in data.jl before providing the data to the model"
	tokenizer = sc_preprocessor.tokenizer
	text_a = x.passage
	text_b = join([x.questions[1].question, tokenizer.sep_token, x.questions[1].options[1].text], " ")
	output = encode_plus(tokenizer, text_a, text_b, add_special_tokens=true, max_length=sc_preprocessor.max_seq_length, truncation=true)
	return output["input_ids"], output["token_type_ids"]
end

function get_sc_inputs(x::ReCoRD, sc_preprocessor)
	@error "ReCoRD does not support sequence classification"
	exit(1)
end

function get_sc_inputs(x::COPA, sc_preprocessor)
	premise = rstrip(ispunct, x.premise)
	choice1 = "$(lowercase(x.choice1[1]))$(x.choice1[2:end])"
	choice2 = "$(lowercase(x.choice2[1]))$(x.choice2[2:end])"

	question = x.question
	joiner = question == "cause" ? "because" : "so"
	text_a, text_b = join([premise, joiner, choice1], " "), join([premise, joiner, choice2], " ")
	output = encode_plus(sc_preprocessor.tokenizer, text_a, text_b, add_special_tokens=true, max_length=sc_preprocessor.max_seq_length, truncation=true)
	return output["input_ids"], output["token_type_ids"]	
end

function get_sc_inputs(x::WSC, sc_preprocessor)
    target = x.entity
    pronoun_idx = x.start2

    # Mark the pronoun with asterisks
    words_a = split(x.text)
    words_a[pronoun_idx] = "*$(words_a[pronoun_idx])*"
    text_a = join(words_a, " ")
    text_b = target

	output = encode_plus(sc_preprocessor.tokenizer, text_a, text_b, add_special_tokens=true, max_length=sc_preprocessor.max_seq_length, truncation=true)
	return output["input_ids"], output["token_type_ids"]	
end

function trim_input_ids(;input_ids, pad_token_id, mask_token_id, num_masks)
	@assert size(input_ids)[end] == 1 "Batch size for input to trim_input_ids should be 1"
	input_ids_without_pad = [x for x in reshape(input_ids, :) if x != pad_token_id]

	trimmed_input_ids = Array{Int, 1}()
	mask_count = 0
	for input_id in input_ids_without_pad
		if input_id == mask_token_id
			if mask_count >= num_masks
				continue
			end
			mask_count += 1
		end
		push!(trimmed_input_ids, input_id)
	end
	return reshape(trimmed_input_ids, :, 1)
end

function add_input_features_to_batch!(dataset, input_batch, indices, task_name)
	if task_name=="record"
		input_batch["candidate_token_ids"]=dataset["candidate_token_ids"][indices]
		input_batch["candidate_labels"]=dataset["candidate_labels"][indices]		
		input_batch["original_choices"]=dataset["original_choices"]
		input_batch["question_idx"]=dataset["question_idx"][indices]
	elseif task_name=="copa"
		# Return if not keys -> Incompatible model type, don't need to do anything
		!haskey(dataset, "choice_1_token_ids") && return
		input_batch["choice_1_token_ids"]=dataset["choice_1_token_ids"][indices]
		input_batch["choice_2_token_ids"]=dataset["choice_2_token_ids"][indices]
	elseif task_name=="wsc"
		!haskey(dataset, "target_token_ids") && return
		input_batch["target_id"]=[idx for (idx, x) in enumerate(indices)]
		input_batch["id_to_target"]=dataset["target"][indices]
		input_batch["target_token_ids"]=dataset["target_token_ids"][indices]
	end
end