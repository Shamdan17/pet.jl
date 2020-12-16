# Pattern verbalizer pairs
include("data.jl")

abstract type PVP end


mutable struct BoolQPVP<:PVP
	tokenizer
	# wrapper
	pattern_id
	verbalizers
end

function BoolQPVP(tokenizer, pattern_id)
	BoolQPVP(
		tokenizer, 
		pattern_id,
		[
		Dict(
			1=>"False",
			2=>"True"
			),
		Dict(
			1=>"No",
			2=>"Yes"
			)
		]
		)
end

# Input: BoolQ data instance
# Output: The pattern corresponding to the given id 
function (bq::BoolQPVP)(x::BoolQ)
	if bq.pattern_id <= 2
		return [(x.passage, true), ". Question: ", (x.question, true), "? Answer: ", bq.tokenizer.mask_token, "."]
	elseif bq.pattern_id <= 4
		return [(x.passage, true), ". Based on the previous passage, ", (x.question, true), "?", bq.tokenizer.mask_token, "."]
	else
		return ["Based on the following passage, ", (x.question, true), "?", bq.tokenizer.mask_token, ".", (x.passage, true)]
	end
end

# Input: BoolQ data instance
# Output: The pattern corresponding to the given id 
function (bq::BoolQPVP)(label::Int)
	return bq.verbalizers[1 + bq.pattern_id%2][label]
end


function encode(pvp::PVP, datum, max_length)
	parts = pvp(datum)

	parts = [typeof(x)<:Tuple ? x : (x, false) for x in parts]

	parts = [(pvp.tokenizer(x, add_special_tokens=false)["input_ids"], s) for (x, s) in parts]

	truncate(max_length, parts, tokenizer=pvp.tokenizer)

	tokens_a = [token_id for (part, s) in parts for token_id in part]

	input_ids = build_inputs_with_special_tokens(pvp.tokenizer, tokens_a)

	input_type_ids = create_token_type_ids_from_sequences(pvp.tokenizer, tokens_a)

	input_ids, input_type_ids
end

function convert_mlm_logits_to_cls_logits(pvp::PVP, mlm_labels, logits)
	label_list = sort(collect(keys(pvp.verbalizers)))

	m2c_tensor = ones(Int, length(label_list))

	for (label_idx, label) in enumerate(label_list)
		verbalization = pvp(label)

		id = pvp.tokenizer(verbalization, add_special_tokens=false)["input_ids"]

		@assert length(id) == 1 "Verbalization $verbalization does not map to single token. Maps to $id"

		id = id[1]

		# Ensure id isn't special token
		@assert !((id-1) in pvp.tokenizer.tokenizer.all_special_ids)

		m2c_tensor[label_idx] = id
	end

	println("size(m2c_tensor)", size(m2c_tensor))
	println("size(cls_logits)", size(logits))
	# logits = V x N x B
	N, B = size(logits)[2:end]
	logits = reshape(logits, :, N*B)
	cls_logits = reshape(logits[m2c_tensor, :], N, B) # length(label_list) x N x B
	
	# Get the logits corresponding to the masks
	masked_logits = reshape(reshape(cls_logits, length(m2c_tensor), :)[:, reshape(mlm_labels, :).>0], length(m2c_tensor), :)  # length(m2c_tensor)  x B

	# Output should be length(m2c_tensor) x B, could be different if multiple tokens implemented 
	return masked_logits
end


function truncate(max_length, parts_a, parts_b=nothing; tokenizer=tokenizer)
	total_len = seq_length(parts_a) + seq_length(parts_b)
	total_len += tokenizer.tokenizer.num_special_tokens_to_add(parts_b!=nothing)

	num_tokens_to_remove = total_len-max_length

	for _ in 1:num_tokens_to_remove
		if seq_length(parts_a, only_shortenable=true) > seq_length(parts_b, only_shortenable=true)
			remove_last(parts_a)
		else
			remove_last(parts_b)
		end
	end
	parts_a, parts_b
end

# Remove one token from the end of the last shortenable sequence
function remove_last(parts)
	last_idx = maximum([((shortenable && length(seq)>0) ? idx : -1) for (idx, (seq, shortenable)) in enumerate(parts)])
	if last_idx == -1
		@error "Could not shorten sequence to less than max-seq-length. Parts: $parts. Please consider using a higher max-seq-length or removing long sequences."
	else
		parts[last_idx] = (parts[last_idx][1][1:end-1], parts[last_idx][2])
	end

end

function seq_length(parts; only_shortenable=false)
	return parts == nothing ? 0 : sum([(!only_shortenable || shortenable) ? length(x) : 0 for (x, shortenable) in parts])
end

