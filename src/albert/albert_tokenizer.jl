using PyCall

py"""
from transformers import AlbertTokenizer

def init(modelname):
	return AlbertTokenizer.from_pretrained(modelname)


def tokenize(model, sentence, **kwargs):
	return model(sentence, truncation=True, **kwargs)

def detokenize(model, tokens, **kwargs):
	return model.convert_ids_to_tokens(tokens, **kwargs)
"""

struct AlbertTokenizer
	tokenizer
	bos_token
	eos_token
	unk_token
	sep_token
	pad_token
	cls_token
	mask_token
	pad_token_id
	pad_token_type_id
	mask_token_id
end

function AlbertTokenizer(modelname::AbstractString)
	return AlbertTokenizer(py"init"(modelname),"[CLS]","[SEP]","<unk>","[SEP]","<pad>","[CLS]","[MASK]",1, 1, 5)
end

function (at::AlbertTokenizer)(input::AbstractString; add_special_tokens=true)
	out = py"tokenize"(at.tokenizer, input, add_special_tokens=add_special_tokens)
	result = Dict()
	result["input_ids"]=[x+1 for x in out["input_ids"]]
	result["token_type_ids"]=[x+1 for x in out["token_type_ids"]]
	result["attention_mask"]=[x for x in out["attention_mask"]]
	result
end

function (at::AlbertTokenizer)(input; clean=true, o...)
	out = py"detokenize"(at.tokenizer, input.-1	, o...)
	if !clean
		return out
	end
	return replace(join(out[2:end-1], ""), "▁"=>" ")
end


function build_inputs_with_special_tokens(tokenizer, input_ids, o...)
	out = tokenizer("")["input_ids"]
	out = [out[1], input_ids..., out[2]]
	if length(o)>0
		out = [out..., build_inputs_with_special_tokens(tokenizer, o[1], o[2:end]...)[2:end]...]
	end
	out
end


function create_token_type_ids_from_sequences(tokenizer, input_ids, o...)
	out = [1 for i in 1:(2+length(input_ids))]

	if length(o)>0
		out = [out..., (create_token_type_ids_from_sequences(tokenizer, o[1], o[2:end]...)[2:end].+1)...]
	end
	out
end
