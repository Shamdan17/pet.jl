include("src/albert_model.jl")
include("src/albert_tokenizer.jl")

tokenizer = AlbertTokenizer("albert-base-v2")

model_url = "https://huggingface.co/albert-base-v2/resolve/main/pytorch_model.bin"
model_path = joinpath("src", "test_files", "pytorch_model.bin")
model_config = joinpath("src", "test_files", "config.json")

# Download model pretrained parameters if not downloaded before.
!isfile(model_path) && download(model_url, model_path)

mlm = pretrainedAlbertForMLM("src/test_files/pytorch_model.bin", "src/test_files/config.json", atype=Array{Float32})

println("Enter a sentence, you can mask certain words using [MASK]")
input = "The capital of France is [MASK]."
# input = readline()

input_ids = tokenizer(input)["input_ids"]

# Forward pass
model_out = mlm(reshape(input_ids, :, 1))

# Get most likely tokens
model_out = [x[1] for x in argmax(model_out["logits"], dims=1)]

# Detokenize
output = tokenizer(reshape(model_out, :), clean=true)

println("Output: ", output)