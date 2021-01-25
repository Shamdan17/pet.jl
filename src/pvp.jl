# Pattern verbalizer pairs
include("data.jl")
using Random

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
function (bq::BoolQPVP)(x::BoolQ; o...)
	if bq.pattern_id <= 2
		return [(x.passage, true), ". Question: ", (x.question, true), "? Answer: ", bq.tokenizer.mask_token, "."], nothing
	elseif bq.pattern_id <= 4
		return [(x.passage, true), ". Based on the previous passage, ", (x.question, true), "?", bq.tokenizer.mask_token, "."], nothing
	else
		return ["Based on the following passage, ", (x.question, true), "?", bq.tokenizer.mask_token, ".", (x.passage, true)], nothing
	end
end

# Input: BoolQ data instance
# Output: The pattern corresponding to the given id 
function (bq::BoolQPVP)(label::Int)
	return bq.verbalizers[1 + bq.pattern_id%2][label]
end

mutable struct RtePVP<:PVP
	tokenizer
	# wrapper
	pattern_id
	verbalizers
end

function RtePVP(tokenizer, pattern_id)
	RtePVP(
		tokenizer, 
		pattern_id,
		[
		Dict(
			1=>"Yes",
			2=>"No",
			),
		Dict(
			1=>"true",
			2=>"false",
			)
		]
		)
end

# Input: RTE data instance
# Output: The pattern corresponding to the given id 
function (rte::RtePVP)(x::RTE; o...)
    # switch text_a and text_b to get the correct order
    text_a = (x.premise, true)
    text_b = (rstrip(ispunct, x.hypothesis), true)

    if rte.pattern_id == 1
        return ["\"", text_b, "\" ?"], [rte.tokenizer.mask_token, ", \"", text_a, "\""]
    elseif rte.pattern_id == 2
        return [text_b, "?"], [rte.tokenizer.mask_token, ",", text_a]
    elseif rte.pattern_id == 3
        return ["\"", text_b, "\" ?"], [rte.tokenizer.mask_token, ". \"", text_a, "\""]
    elseif rte.pattern_id == 4
        return [text_b, "?"], [rte.tokenizer.mask_token, ".", text_a]
    elseif rte.pattern_id == 5
        return [text_a, " question: ", (x.hypothesis, true), " true or false? answer: ", rte.tokenizer.mask_token], nothing
	end
end

# Input: RTE data instance
# Output: The pattern corresponding to the given id 
function (rte::RtePVP)(label::Int)
	if rte.pattern_id == 5
		return rte.verbalizers[2][label]
	else
		return rte.verbalizers[1][label]
	end
end


mutable struct CBPVP<:PVP
	tokenizer
	# wrapper
	pattern_id
	verbalizers
end

function CBPVP(tokenizer, pattern_id)
	CBPVP(
		tokenizer, 
		pattern_id,
		[
		Dict(
			1=>"No",
			2=>"Yes",
			3=>"Maybe"
			),
		Dict(
			1=>"false",
			2=>"true",
			3=>"neither"
			)
		]
		)
end

# Input: CB data instance
# Output: The pattern corresponding to the given id 
function (cb::CBPVP)(x::CB; o...)
    # switch text_a and text_b to get the correct order
    text_a = (x.premise, true)
    text_b = (rstrip(ispunct, x.hypothesis), true)

    if cb.pattern_id == 1
        return ["\"", text_b, "\" ?"], [cb.tokenizer.mask_token, ", \"", text_a, "\""]
    elseif cb.pattern_id == 2
        return [text_b, "?"], [cb.tokenizer.mask_token, ",", text_a]
    elseif cb.pattern_id == 3
        return ["\"", text_b, "\" ?"], [cb.tokenizer.mask_token, ". \"", text_a, "\""]
    elseif cb.pattern_id == 4
        return [text_b, "?"], [cb.tokenizer.mask_token, ".", text_a]
    elseif cb.pattern_id == 5
        return [text_a, " question: ", (x.hypothesis, true), " true, false or neither? answer: ", cb.tokenizer.mask_token], nothing
	end
end

# Input: CB data instance
# Output: The pattern corresponding to the given id 
function (cb::CBPVP)(label::Int)
	if cb.pattern_id == 5
		return cb.verbalizers[2][label]
	else
		return cb.verbalizers[1][label]
	end
end

mutable struct WiCPVP<:PVP
	tokenizer
	# wrapper
	pattern_id
	verbalizers
end

function WiCPVP(tokenizer, pattern_id)
	WiCPVP(
		tokenizer, 
		pattern_id,
		[
		Dict(
			1=>"No",
			2=>"Yes",
			),
		Dict(
			1=>"2",
			2=>"b",
			),
		Dict(
			1=>"different",
			2=>"similar",
			)
		]
		)
end

# Input: WiC data instance
# Output: The pattern corresponding to the given id 
function (wic::WiCPVP)(x::WiC; training=false, o...)
    # switch text_a and text_b to get the correct order
    text_a = (x.sentence1, true)
    text_b = (x.sentence2, true)
    word = x.word

    # Random swap if training
    if training && rand(Bool)
    	tmp = text_a
    	text_a = text_b
    	text_b = tmp
    end

    if wic.pattern_id == 1
        return ["\"", text_a, "\" / \"", text_b, "\" Similar sense of \"$word\"?", wic.tokenizer.mask_token, "."], nothing
    elseif wic.pattern_id == 2
        return [text_a, text_b, "Does $word have the same meaning in both sentences?", wic.tokenizer.mask_token], nothing
    elseif wic.pattern_id == 3
        return [word, " . Sense (1) (a) \"", text_a, "\" (", wic.tokenizer.mask_token, ") \"", text_b, "\""], nothing
    elseif wic.pattern_id == 4
        return ["The meaning of the word $word is ", wic.tokenizer.mask_token, "in the sentences \"", text_a, " and \"", text_b, "\""], nothing
	end
end

# Input: WiC data instance
# Output: The pattern corresponding to the given id 
function (wic::WiCPVP)(label::Int)
	if wic.pattern_id == 3
		return wic.verbalizers[2][label]
	elseif wic.pattern_id == 4
		return wic.verbalizers[3][label]
	else
		return wic.verbalizers[1][label]
	end
end


mutable struct MultiRCPVP<:PVP
	tokenizer
	# wrapper
	pattern_id
	verbalizers
end

function MultiRCPVP(tokenizer, pattern_id)
	MultiRCPVP(
		tokenizer, 
		pattern_id,
		[
		Dict(
			1=>"No",
			2=>"Yes",
			),
		Dict(
			1=>"false",
			2=>"true",
			)
		]
		)
end

# Input: MultiRC data instance
# Output: The pattern corresponding to the given id 
function (mrc::MultiRCPVP)(x::MultiRC; training=false, o...)
	@assert length(x.questions)==1 && length(x.questions[1].options)==1 "MultiRC instances with multiple questions/answers are not supported. Please use the flatten function in data.jl before providing the data to the model"

    passage = (x.passage, true)
    # Don't allow question to be shortened
    question = x.questions[1].question
    answer = x.questions[1].options[1].text

    if mrc.pattern_id == 1
        return [passage, ". Question: ", question, "? Is it ", answer, "?", mrc.tokenizer.mask_token, "."], nothing
    elseif mrc.pattern_id == 2
        return [passage, ". Question: ", question, "? Is the correct answer \"", answer, "\"?", mrc.tokenizer.mask_token, "."], nothing
    elseif mrc.pattern_id == 3
        return [passage, ". Based on the previous passage, ", question, "? Is \"", answer, "\" a correct answer?", mrc.tokenizer.mask_token, "."], nothing
    elseif mrc.pattern_id == 4
        return [passage, question, "- [", mrc.tokenizer.mask_token, "]",  answer], nothing
	end
end

# Input: MultiRC data instance
# Output: The pattern corresponding to the given id 
function (mrc::MultiRCPVP)(label::Int)
	if mrc.pattern_id == 4
		return mrc.verbalizers[2][label]
	else
		return mrc.verbalizers[1][label]
	end
end


mutable struct ReCoRDPVP<:PVP
	tokenizer
	# wrapper
	pattern_id
	verbalizers
end

function ReCoRDPVP(tokenizer, pattern_id)
	ReCoRDPVP(
		tokenizer, 
		pattern_id,
		[]
		)
end

# Input: ReCoRD data instance
# Output: The pattern corresponding to the given id 
function (rcrd::ReCoRDPVP)(x::ReCoRD; training=false, o...)
	@assert length(x.qas)==1 "ReCoRD instances with multiple questions are not supported. Please use the flatten function in data.jl before providing the data to the model"

    passage = replace(x.text, "@highlight\n"=>"- ")
    passage = (passage, true)
    # Don't allow question to be shortened
    question = x.qas[1].query
    @assert occursin("@placeholder", question) "Question $question does not have a placeholder token."

    num_masks = maximum(length(rcrd.tokenizer(c.text, add_special_tokens=false)["input_ids"]) for c in x.entities)

    question = replace(question, "@placeholder"=>rcrd.tokenizer.mask_token^num_masks)

    return [passage, question], nothing
end

# Input: ReCoRD data instance
# Output: Empty string since unused
function (rcrd::ReCoRDPVP)()
	return ""
end


mutable struct COPAPVP<:PVP
	tokenizer
	# wrapper
	pattern_id
	verbalizers
end

function COPAPVP(tokenizer, pattern_id)
	COPAPVP(
		tokenizer, 
		pattern_id,
		[
		]
		)
end

# Input: COPA data instance
# Output: The pattern corresponding to the given id 
function (cpa::COPAPVP)(x::COPA; training=false, o...)
    premise = (rstrip(ispunct, x.premise), true)
    # Don't allow question to be shortened
    choice1 = rstrip(ispunct, "$(lowercase(x.choice1[1]))$(x.choice1[2:end])")
    choice2 = rstrip(ispunct, "$(lowercase(x.choice2[1]))$(x.choice2[2:end])")

	question = x.question
	@assert question in ["cause", "effect"]
	num_masks = maximum(length(cpa.tokenizer(c, add_special_tokens=false)["input_ids"]) for c in [choice1,choice2])

	joiner = question == "cause" ? "because" : ", so"

    if cpa.pattern_id == 1
        return ["\"", choice1, "\" or \"", choice2,"\"?", premise, joiner, cpa.tokenizer.mask_token ^ num_masks, "."], nothing
    elseif cpa.pattern_id == 2
        return [choice1, "or", choice2,"?", premise, joiner, cpa.tokenizer.mask_token ^ num_masks, "."], nothing
    end
end

# Input: COPA data instance
# Output: The pattern corresponding to the given id 
function (cpa::COPAPVP)(label::Int)
	""
end



mutable struct WSCPVP<:PVP
	tokenizer
	# wrapper
	pattern_id
	verbalizers
	rng
end

function WSCPVP(tokenizer, pattern_id)
	WSCPVP(
		tokenizer, 
		pattern_id,
		[
		],
		nothing
		)
end

# Input: COPA data instance
# Output: The pattern corresponding to the given id 
function (wsc::WSCPVP)(x::WSC; training=false, seed=42, o...)
	if wsc.rng == nothing
		wsc.rng = MersenneTwister(seed)
	end
    pronoun = x.pronoun
    target = x.entity
    pronoun_idx = x.start2

    words_a = split(x.text)
    words_a[pronoun_idx] = "*$(words_a[pronoun_idx])*"
    text_a = (join(words_a, " "), true)
    # Don't allow question to be shortened
    num_pads = training ? rand(wsc.rng, 0:3) : 1 

	num_masks = length(wsc.tokenizer(target, add_special_tokens=false)["input_ids"]) + num_pads
	masks = wsc.tokenizer.mask_token^num_masks

    if wsc.pattern_id == 1
        return [text_a, "The pronoun '*$pronoun*' refers to", "$masks."], nothing
    elseif wsc.pattern_id == 2
    	return [text_a, "In the previous sentence, the pronoun '*$pronoun*' refers to", "$masks."], nothing
    elseif wsc.pattern_id == 3
    	return [text_a, "Question: In the passage above, what does the pronoun '*$pronoun*' refer to? Answer: ", "$masks."], nothing
    end
end

# Input: WSC data instance
# Output: The pattern corresponding to the given id 
function (wsc::WSCPVP)(label::Int)
	""
end


function encode(pvp::PVP, datum, max_length; training=false)
	parts_a, parts_b = pvp(datum, training=training)

	parts_a = [typeof(x)<:Tuple ? x : (x, false) for x in parts_a]
	parts_b = parts_b == nothing ? parts_b : [typeof(x)<:Tuple ? x : (x, false) for x in parts_b]

	parts_a = [(pvp.tokenizer(x, add_special_tokens=false)["input_ids"], s) for (x, s) in parts_a]
	parts_b = parts_b == nothing ? parts_b : [(pvp.tokenizer(x, add_special_tokens=false)["input_ids"], s) for (x, s) in parts_b]

	truncate(max_length, parts_a, parts_b, tokenizer=pvp.tokenizer)

	tokens_a = [token_id for (part, s) in parts_a for token_id in part]
	tokens_b = parts_b == nothing ? parts_b : [token_id for (part, s) in parts_b for token_id in part]

	input_ids = build_inputs_with_special_tokens(pvp.tokenizer, tokens_a, tokens_b)

	input_type_ids = create_token_type_ids_from_sequences(pvp.tokenizer, tokens_a, tokens_b)

	input_ids, input_type_ids
end

function convert_mlm_logits_to_cls_logits(pvp::PVP, mlm_labels, logits)
	label_list = sort(collect(keys(pvp.verbalizers[1])))

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

	# logits = V x N x B
	N, B = size(logits)[2:end]
	logits = reshape(logits, :, N*B)
	cls_logits = reshape(logits[m2c_tensor, :], length(label_list), N, B) # length(label_list) x N x B
	
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


pvp_map = Dict(
	"boolq"=>BoolQPVP,
	"cb"=>CBPVP,
	"rte"=>RtePVP,
	"wic"=>WiCPVP,
	"multirc"=>MultiRCPVP,
	"record"=>ReCoRDPVP,
	"copa"=>COPAPVP,
	"wsc"=>WSCPVP,
)