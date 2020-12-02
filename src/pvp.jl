# Pattern verbalizer pairs
include("data.jl")

struct PVP end


struct BoolQPVP:<PVP
	tokenizer
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
		return [x.passage, ". Question: ", x.question, "? Answer: ", bq.tokenizer.mask_token, "."]
	elseif bq.pattern_id <= 4
		return [x.passage, ". Based on the previous passage, ", x.question, "?", bq.tokenizer.mask_token, "."]
	else
		return ["Based on the following passage, ", x.question, "?", bq.tokenizer.mask_token, ".", bq.passage]
end

# Input: BoolQ data instance
# Output: The pattern corresponding to the given id 
function (bq::BoolQPVP)(label::Int)
	return bq.verbalizers[1 + self.pattern_id%2][label]
end


function encode(pvp::pvp, datum)
	parts = pvp(datum)

	parts = [pvp.tokenizer(x, add_special_tokens=false)["input_ids"] for x in parts]

	tokens_a = [token_id for part in parts for token_id in part]

	input_ids = build_inputs_with_special_tokens(pvp.tokenizer, tokens_a)

	input_type_ids = create_token_type_ids_from_sequences(pvp.tokenizer, tokens_a)
end