using PyCall

py"""
from transformers import AlbertTokenizer

def init(modelname):
	return AlbertTokenizer.from_pretrained(modelname)


def tokenize(model, sentence):
	return model(sentence)

def detokenize(model, tokens):
	return model.convert_ids_to_tokens(tokens)
"""

struct AlbertTokenizer
	tokenizer
end

function AlbertTokenizer(modelname::AbstractString)
	return AlbertTokenizer(py"init"(modelname))
end

function (at::AlbertTokenizer)(input::AbstractString)
	out = py"tokenize"(at.tokenizer, input)
	result = Dict()
	result["input_ids"]=[x+1 for x in out["input_ids"]]
	result["token_type_ids"]=[x+1 for x in out["token_type_ids"]]
	result["attention_mask"]=[x for x in out["attention_mask"]]
	result
end

function (at::AlbertTokenizer)(input; clean=true)
	out = py"detokenize"(at.tokenizer, input.-1	)
	if !clean
		return out
	end
	return replace(join(out[2:end-1], ""), "▁"=>" ")
end
