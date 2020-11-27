include("layers.jl")

# ALBERT's Embedding layer. ALBERT's Embedding layer is identical to BERT. The projection from embed size to hidden size is 
# not done within the embedding layer.  
mutable struct ALBERTEmbedding
    word_embeds::Embed
    token_type_embeds::Embed
    pos_embeds::Embed
    lnorm::LayerNorm
    pdrop
end


function ALBERTEmbedding(vocab_size, type_size, embed_size, max_positions, dropout=0; atype=atype())
    word_embeds = Embed(vocab_size, embed_size, atype=atype)
    token_type_embeds = Embed(type_size, embed_size, atype=atype)
    pos_embeds = Embed(max_positions, embed_size, atype=atype)
    lnorm = LayerNorm(embed_size, atype=atype)
    return ALBERTEmbedding(word_embeds, token_type_embeds, pos_embeds, lnorm, dropout)
end

function (b::ALBERTEmbedding)(input_ids, type_embeds=nothing, position_ids=nothing, o...)
    # Size input_ids: TxB
    # Size type_embeds: TxB
    # Size position_ids: TxB
    # If type_embeds are nothing, default to first type
    if type_embeds == nothing
        type_embeds = ones(Int, 1, 1) # Shape: 1x1
    end
    # If position_embeds are nothing, default to indices
    if position_ids == nothing
        position_ids = collect(1:size(input_ids, 1)) # Shape: Tx1
    end
    # Embed everything
    embeddings = b.word_embeds(input_ids).+b.token_type_embeds(type_embeds).+b.pos_embeds(position_ids)
    # Apply layernorm then dropout
    dropout(b.lnorm(embeddings), b.pdrop)    
end
