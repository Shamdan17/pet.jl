include("albert_attention.jl")
include("albert_embeddings.jl")
include("layers.jl")


# FFN Block which includes layernorm
mutable struct FeedForwardBlock
    ffn_sublayer
end

function FeedForwardBlock(dmodel::Int, ffn_dim::Int, activation, pdrop; atype=atype())
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

function ALBERTLayer(dmodel::Int, num_heads::Int, ffn_dim::Int, attention_pdrop, layer_pdrop; activation="new_gelu", atype=atype())
    attn_block = ALBERTAttentionBlock(dmodel,num_heads,attention_pdrop, layer_pdrop, atype=atype)
    ffn_block = FeedForwardBlock(dmodel, ffn_dim, activation, layer_pdrop, atype=atype)
    ALBERTLayer(attn_block, ffn_block)
end

function (a::ALBERTLayer)(
    x, 
    attention_mask=nothing,
    head_mask=nothing,
    output_attentions=false,
    output_hidden_states=false)
    
    # TODO: headmask implementation
    attn_out, attn_scores = a.attn_block(x, keymask=attention_mask, return_scores=output_attentions)

    ffn_out = a.ffn_block(attn_out)

    ffn_out, attn_scores
end

    