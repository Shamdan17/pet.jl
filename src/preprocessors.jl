include("pvp.jl")

abstract type preprocessor end

mutable struct MLMPreprocessor <: preprocessor
	mlmtokenizer
	mlmconfig
	max_seq_length
	pvp
	label_map::Dict
end

function MLMPreprocessor(tokenizer, mlmconfig, max_seq_length, pvp, labels)
	label_map = Dict(label=>i for (i, label) in enumerate(labels))
	MLMPreprocessor(tokenizer, mlmconfig, max_seq_length, pvp, label_map)
end


# Converts an input example 
function (p::MLMPreprocessor)(example)

	input_ids, token_type_ids = encode(p.pvp, example, p.max_seq_length)

	attention_mask = [1 for i in 1:length(input_ids)]

	padding_length = p.max_seq_length - length(input_ids)

	# TODO: Skip longer sequences?
	input_ids = [input_ids..., [p.mlmtokenizer.pad_token_id for i in 1:padding_length]...]#[1:p.max_seq_length]

	token_type_ids = [token_type_ids..., [p.mlmtokenizer.pad_token_type_id for i in 1:padding_length]...]#[1:p.max_seq_length]

	attention_mask = [attention_mask..., [0 for i in 1:padding_length]...]#[1:p.max_seq_length]

	@assert length(input_ids) == p.max_seq_length "Got inputs of length: $(length(input_ids)). Expected: $(p.max_seq_length)"
	@assert length(token_type_ids) == p.max_seq_length
	@assert length(attention_mask) == p.max_seq_length

	label = example.labeled ? p.label_map[getLabel(example)] : nothing

	logits = example.logits != nothing ? example.logits : [-1]

	# The if statement here is only needed in case of lm training, which we don't do atm.
	# if example.labeled
	mlm_labels = [i==p.mlmtokenizer.mask_token_id ? 1 : -1 for i in input_ids]
	@assert sum(mlm_labels.==1)==1 "More than 1 or no mask tokens found in string. Input Ids: $input_ids"
	# else
	# 	mlm_labels = [1 for i in 1:length(input_ids)]
	# end

	return Dict(
		"input_ids"=>input_ids,
		"attention_mask"=>attention_mask,
		"token_type_ids"=>token_type_ids,
		"label"=>label,
		"mlm_labels"=>mlm_labels,
		"logits"=>logits,
		"idx"=>example.idx,
		)
end

mutable struct SCPreprocessor <: preprocessor
	sctokenizer
	scconfig
	max_seq_length
	pvp
	label_map::Dict
end

function SCPreprocessor(tokenizer, scconfig, pvp, labels)
	label_map = Dict(label=>i for (i, label) in enumerate(labels))
	SCPreprocessor(tokenizer, scconfig, pvp, labels)
end


# Converts an input example 
function (p::SCPreprocessor)(example)

	input_ids, token_type_ids = encode(p.pvp, example)

	attention_mask = [1 for i in 1:length(input_ids)]

	padding_length = p.max_seq_length - length(input_ids)

	input_ids = [input_ids..., [p.sctokenizer.pad_token_id for i in 1:padding_length]]

	token_type_ids = [token_type_ids..., [p.sctokenizer.pad_token_type_id for i in 1:padding_length]]

	attention_mask = [attention_mask..., [0 for i in 1:padding_length]]

	@assert length(input_ids) == p.max_seq_length
	@assert length(token_type_ids) == p.max_seq_length
	@assert length(attention_mask) == p.max_seq_length

	label = example.labeled ? p.label_map[getLabel(example)] : nothing

	logits = example.logits != nothing ? example.logits : [-1]

	mlm_labels = [-1 for i in 1:length(input_ids)]

	return Dict(
		"input_ids"=>input_ids,
		"attention_mask"=>attention_mask,
		"token_type_ids"=>token_type_ids,
		"label"=>label,
		"mlm_labels"=>mlm_labels,
		"logits"=>logits,
		"idx"=>example.idx,
		)
end

