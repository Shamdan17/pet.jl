# layers.jl contains layers that are not specific to ALBERT

using Knet
using Statistics: mean, std
using BenchmarkTools


# Layer Normalization
mutable struct LayerNorm; a; b; ϵ; end 

"""
    LayerNorm(dmodel)

Creates an layer normalization layer. Inputs should be hidden vectors with hidden size of dmodel

Input shape: Tensor of arbitrary number of hidden vectors [dmodel, o...]
Output shape: Identical shape of [dmodel, o...]
"""
function LayerNorm(dmodel; eps=1e-12)
    a = param(dmodel; init=ones, atype=Array{Float32})
    b = param(dmodel; init=zeros, atype=Array{Float32})
    LayerNorm(a, b, eps)
end


function (l::LayerNorm)(x, o...)
    μ = mean(x,dims=1)
    # Albert Implementation uses corrected == false when testing
    # Source: https://github.com/huggingface/transformers/blob/b592728eff9996e2cff1c5107438c4989aaa8149/src/transformers/models/albert/modeling_albert.py#L239 
    σ = std(x,mean=μ,dims=1, corrected=false)
    ϵ = eltype(x)(l.ϵ)
    l.a .* (x .- μ) ./ (σ .+ ϵ) .+ l.b # TODO: doing x .- μ twice?
end


# Primitive Embed Layer
mutable struct Embed; w; end

"""
    Embed(vocab_size, dmodel)

Creates an embedding layer with a vocabulary size of vocab_size and embedding size of dmodel

Input shape: Tensor of arbitrary dimensions [o...]
Output shape: [dmodel, o...]

Usage:
```jldoctest
julia> emb = Embed(500, 64); # Create embedding with with a vocabulary size of 500 and embedding size of 64

julia> input = [idx for idx in 1:30]; # Input indices, shape: [30]

julia> embeddings = emb(input); # Embeddings of size [64, 30]
```
"""
function Embed(vocab_size, dmodel)
    Embed(param(dmodel, vocab_size))
end

function (e::Embed)(x, o...); e.w[:,x]; end



"""
    SubLayer(layer, dmodel, pdrop)

Creates an sublayer with a residual connection and a layernorm. Calculates LayerNorm(x+dropout(layer(x)))

Input shape: Hidden vector of arbitrary dimensions [dmodel, o...]
Output shape: [dmodel, o...]
```
"""
mutable struct SubLayer; layer; norm; pdrop; end

function SubLayer(layer, dmodel::Int, pdrop::Number)
    SubLayer(layer, LayerNorm(dmodel), pdrop)
end

function (l::SubLayer)(x, xs...)
    l.norm(x .+ dropout(l.layer(x, xs...), l.pdrop))
end


"""
    Linear(input, outputs...; bias=true)

Creates an generalized linear/affine layer which accepts arbitrary dimensional inputs and can output an arbitrary number of dimensions.

Input shape: Hidden vector of arbitrary dimensions [input, o...]
Output shape: [outputs..., o...]
```
"""
mutable struct Linear; w; b; end


function Linear(input::Int, outputs...; bias=true)
    Linear(param(outputs..., input),
        bias ? param0(outputs...) : nothing)
end

function (l::Linear)(x, o...)
    W1, W2, X1, X2 = size(l.w)[1:end-1], size(l.w)[end], size(x)[1], size(x)[2:end]
    # @show W1,W2,X1,X2, size(x)
    @assert W2 === X1
    y = reshape(l.w,:,W2) * reshape(x,X1,:)
    y = reshape(y, W1..., X2...)
    if l.b!=nothing; y = y .+ l.b; end
    y
end


"""
    FeedForwardNetwork(dmodel::Int, ffn_dim::Int, activation)

Creates an generalized FeedForwardNetwork with input dimension dmodel and hidden dimension ffn_dim
A FeedForwardNetwork will project the inputs to the hidden dimension, apply a given activation, then project back to the input dimension.

Activation can be a custom function or one of the following strings:

"relu"      =>  relu
"tanh"      =>  tanh
"elu"       =>  elu
"sigm"      =>  sigmoid
"gelu"      =>  gelu
"new_gelu"  =>  new_gelu (This is a tanh approximation of gelu. This is the default for BERT and ALBERT)
"identity"  =>  identity


Input shape: Hidden vector of arbitrary dimensions [dmodel, o...]
Output shape: [dmodel, o...]
```
"""
mutable struct FeedForwardNetwork
    fc1::Linear
    fc2::Linear
    activation
end

function FeedForwardNetwork(dmodel::Int, ffn_dim::Int, activation)
    @assert !(typeof(activation)<:AbstractString) || haskey(activations, activation)
    if typeof(activation)<:AbstractString
        activation = activations[activation]
    end
    fc1 = Linear(dmodel, ffn_dim)
    fc2 = Linear(ffn_dim, dmodel)
    FeedForwardNetwork(fc1, fc2, activation)
end

(f::FeedForwardNetwork)(x, o...) = f.fc2(f.activation.(f.fc1(x, o...)), o...)
