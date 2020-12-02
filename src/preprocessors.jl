include("pvp.jl")

struct preprocessor end

struct MLMPreprocessor
	mlmtokenizer
	mlmconfig
	max_seq_length
	pvp
	label_map::Dict
end

function MLMPreprocessor(tokenizer, mlmconfig, pvp, labels)
	label_map = Dict(label=>i for (i, label) in enumerate(labels))
	MLMPreprocessor(tokenizer, mlmconfig, pvp, labels)
end


# Converts an input example 
function (p::MLMPreprocessor)(example)

	input_ids, token_type_ids = encode(p.pvp, example)

	attention_mask = [1 for i in 1:length(input_ids)]

	padding_length = p.max_seq_length - length(input_ids)

	input_ids = [input_ids..., [p.mlmtokenizer.pad_token_id for i in 1:padding_length]]

	token_type_ids = [token_type_ids..., [p.mlmtokenizer.pad_token_type_id for i in 1:padding_length]]

	attention_mask = [attention_mask..., [0 for i in 1:padding_length]]

	@assert length(input_ids) == p.max_seq_length
	@assert length(token_type_ids) == p.max_seq_length
	@assert length(attention_mask) == p.max_seq_length

	label = example.labeled ? p.label_map[getLabel(example)] : nothing

	logits = example.logits != nothing ? example.logits else [-1]

	if p.labeled
		mlm_labels = [i==p.mask_token_id ? 1 : -1 for i in input_ids]
	else
		mlm_labels = [1 for i in 1:length(input_ids)]

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