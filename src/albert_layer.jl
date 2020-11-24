using Knet.Ops21 # For activations
include("albert_attention.jl")
include("albert_embeddings.jl")
include("layers.jl")
include("new_gelu.jl")

# Valid activation functions
activations = Dict(
    "relu"=>relu,
    "tanh"=>tanh,
    "elu"=>elu,
    "sigm"=>sigm,
    "gelu"=>gelu,
    "new_gelu"=>new_gelu, # Approx gelu using tanh. This is what is used in BERT as well as ALBERT
    "identity"=>identity
)

# No longer needed. Model accepts either a custom activation function or the name of one of the available functions
# # Register an activation with a name
# macro registerActivation(name::String, act)
#     :(activations[$name]=$act)
# end


# FFN Block which includes layernorm
mutable struct FeedForwardBlock
    ffn_sublayer
end

function FeedForwardBlock(dmodel::Int, ffn_dim::Int, activation, pdrop, atype=atype())
    ffn = FeedForwardNetwork(dmodel, ffn_dim, activation, atype=atype)
    ffn_sublayer = SubLayer(ffn, dmodel, pdrop, atype=atype)
    FeedForwardBlock(ffn_sublayer)
end

(f::FeedForwardBlock)(x, o...) = f.ffn_sublayer(x,o...)

# ALBERT layer which applies self attention then a feed forward network.
mutable struct ALBERTLayer
    attn_block::ALBERTAttentionBlock
    ffn_block::FeedForwardBlock
end

function ALBERTLayer(dmodel::Int, num_heads::Int, ffn_dim::Int, attention_pdrop, layer_pdrop, activation="new_gelu", atype=atype())
    attn_block = ALBERTAttentionBlock(dmodel,num_heads,attention_pdrop, layer_pdrop, atype=atype)
    ffn_block = FeedForwardBlock(dmodel, ffn_dim, activation, layer_pdrop, atype=atype)
    ALBERTLayer(attn_block, ffn_block)
end

(a::ALBERTLayer)(x, o...) = a.ffn_block(a.attn_block(x, o...), o...)