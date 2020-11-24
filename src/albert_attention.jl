using Knet
include("layers.jl")

# @size a (b, c) is equivalent to @assert size(a) == (b, c)
macro size(z, s); esc(:(@assert  size($z) == $s  string(summary($z),!=,$s))); end

mutable struct MultiHeadAttention
    q_proj::Linear
    k_proj::Linear
    v_proj::Linear
    o_proj::Linear
    dropout
    scale
    # Whether or not to mask attention to future tokens
    # Not used in ALBERT since it's bidirectional, is kept for future reuse in transformers with decoders
    selfmask::Bool
end

function MultiHeadAttention(dmodel::Int, num_heads::Int, dropout; selfmask=false, scale=1/sqrt(dmodel÷num_heads))
    # dmodel MUST be a multiple of nheads
    @assert dmodel % num_heads == 0
    dk = dmodel ÷ num_heads
    q_proj = Linear(dmodel, dk, num_heads)
    k_proj = Linear(dmodel, dk, num_heads)
    v_proj = Linear(dmodel, dk, num_heads)
    o_proj = Linear(dmodel, dmodel)
    MultiHeadAttention(q_proj, k_proj, v_proj, o_proj, dropout, scale, selfmask)
end

function (m::MultiHeadAttention)(queries, queried; keymask=nothing, o...)
    # queries: HxT1xB
    # queried: HxT2xB
    dk, nheads = size(m.q_proj.w)[1:2]
    H, T1, B = size(queries); T2 = size(queried, 2);  @assert size(queried, 1) === H; @assert size(queried, 3) === B
    # Get q, k, v
    q = m.q_proj(queries);                            @size q (dk,nheads,T1,B)
    k = m.k_proj(queried); v = m.v_proj(queried);     @size k (dk,nheads,T2,B); @size v (dk,nheads,T2,B)
    # attn_scores shape: T1 x T2 x nheads x B
    q, v = permutedims.((q, v), ((3, 1, 2, 4),));     @size q (T1,dk,nheads,B); @size v (T2,dk,nheads,B)
    k = permutedims(k, (1, 3, 2, 4));                 @size k (dk,T2,nheads,B)
    s = bmm(q, k);                                    @size s (T1, T2, nheads, B)
    s = s .* eltype(s)(m.scale);                      @size s (T1, T2, nheads, B)
    s = attnmask(s, keymask, m.selfmask);             @size s (T1, T2, nheads, B)
    s = softmax(s, dims=2)
    # You might be interested in changing dropout here
    # to be something similar to channel dropout in CNNs
    s = dropout(s, m.dropout);                        @size s (T1, T2, nheads, B)
    c = bmm(s, v);                                    @size c (T1, dk, nheads, B)
    c = reshape(c, (T1, :, B));                       @size c (T1, dk*nheads, B)
    o = m.o_proj(permutedims(c, (2, 1, 3)));         @size o (dk*nheads, T1, B)
    o
end

function attnmask(input, keymask, do_selfmask) # s = (Tq, Tk, H, B), keymask = (Tk, B), selfmask=Boolean
    mask = nothing
    if keymask != nothing
        @assert size(keymask) == (size(input, 1), size(input, 4))
        mask = reshape(mask, size(input, 1), 1, 1, size(input, 4))
    end
    if do_selfmask
        @assert size(input, 1) == size(input, 2)
        T = size(input, 1)
        # TODO: Check whether this should be rotated
        tmask = [j<=i for i in 1:T, j in 1:T]
        tmask = reshape(tmask, T, T, 1, 1)
        if mask != nothing
            mask = mask .& tmask
        else
            mask = tmask
        end
    end
    if mask == nothing
        return input
    else
        return input .+ oftype(input, -1e9 * .!mask)
    end
end

# Self attention
(m::MultiHeadAttention)(x, o...) = m(x, x, o...)


mutable struct ALBERTAttentionBlock
    attn_layer
end

function ALBERTAttentionBlock(dmodel::Int, num_heads::Int, attention_pdrop, output_dropout)
    attention_layer = MultiHeadAttention(dmodel, num_heads, attention_pdrop)
    sublayer = SubLayer(attention_layer, dmodel, output_dropout)
    ALBERTAttentionBlock(sublayer)
end

ALBERTAttentionBlock(dmodel::Int, num_heads::Int, pdrop) = ALBERTAttentionBlock(dmodel, num_heads, pdrop, pdrop)

(a::ALBERTAttentionBlock)(input, o...) = a.attn_layer(input, o...)


