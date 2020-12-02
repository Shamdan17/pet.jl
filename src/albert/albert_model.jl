using PyCall
include("albert_attention.jl")
include("albert_embeddings.jl")
include("albert_layer.jl")
include("layers.jl")
include("albert_config.jl")
@pyimport torch


mutable struct AlbertLayerGroup
    albert_layers
end

function AlbertLayerGroup(config::ALBERTConfig; atype=atype())
    layers = [ALBERTLayer(
            config.hidden_size, 
            config.num_attention_heads,
            config.intermediate_size,
            config.attention_probs_dropout_prob,
            config.hidden_dropout_prob,
            activation=config.hidden_act,
            atype=atype
            ) for _ in 1:config.inner_group_num]
    AlbertLayerGroup(layers)
end

function (alg::AlbertLayerGroup)(
        x; # HxTxB                           
        attention_mask=nothing,
        head_mask=nothing, 
        output_attentions=false,
        output_hidden_states=false
    )
    layer_hidden_states = []
    layer_attentions = []
    for (idx, albert_layer) in enumerate(alg.albert_layers)
        layer_output = albert_layer(
                                x, 
                                attention_mask, 
                                head_mask == nothing ? nothing : head_mask[idx], 
                                output_attentions
                            )
        # Keep track of attentions if needed
        if output_attentions
            push!(layer_attentions, layer_output[end])
        end
        # Save hidden states if needed 
        if output_hidden_states; push!(layer_hidden_states, layer_output[1]); end
        
        x=layer_output[1]
    end
    
    result = []
    push!(result, x)
    if output_hidden_states; push!(result, layer_hidden_states); end
    if output_attentions; push!(result, layer_attentions); end
    
    tuple(result...)
end

# The transformer consists of the intermediate layers of albert (embedding projection + encoder layers)
# This is to be consistent with huggingface
mutable struct AlbertTransformer
    config::ALBERTConfig
    embedding_hidden_mapping::Linear
    albert_layer_groups::AbstractArray
end

function AlbertTransformer(config::ALBERTConfig; atype=atype())
    embedding_hidden_mapping = Linear(config.embedding_size, config.hidden_size, atype=atype)
    albert_layer_groups = [AlbertLayerGroup(config, atype=atype) for _ in 1:config.num_hidden_groups]
    AlbertTransformer(config, embedding_hidden_mapping, albert_layer_groups)
end

function (at::AlbertTransformer)(
        x;
        attention_mask=nothing,
        head_mask=nothing,
        output_attentions=false,
        output_hidden_states=false,
        return_dict=true
    )
    # Embed from embed dim to hidden dim
    x = at.embedding_hidden_mapping(x)

    all_hiddens = []
    if output_hidden_states; push!(all_hiddens, x); end
    all_attentions = []

    # Number of layers in a hidden group
    layers_per_group = at.config.num_hidden_layers÷at.config.num_hidden_groups
    
    for i in 1:at.config.num_hidden_layers
        # Which AlbertLayerGroup to use
        group_idx = (i+at.config.num_hidden_layers-1)÷at.config.num_hidden_layers
        layer_group_output = at.albert_layer_groups[group_idx](
            x, # HxTxB                           
            attention_mask=attention_mask,
            head_mask=head_mask, 
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states             
        )

        # Save attentions if needed
        if output_attentions
            layer_attention = layer_group_output[end]
            push!(all_attentions, layer_attention)
        end
        
        # Save hidden states if needed 
        if output_hidden_states; push!(all_hiddens, layer_group_output[2]); end
        
        x = layer_group_output[1]
    end
    
    if !return_dict
        result = []
        push!(result, x)
        if output_hidden_states; push!(result, all_hiddens); end
        if output_attentions; push!(result, all_attentions); end
        return tuple(result...)
    else
        result = Dict()
        result["output"]=x
        if output_hidden_states; result["hiddens"]=all_hiddens; end
        if output_attentions; result["attentions"]=all_attentions; end        
        return result
    end
end

# This consists of the embeddings + transformer. To use, attach different heads as needed 
mutable struct AlbertModel
    config
    embeddings::ALBERTEmbedding
    encoder::AlbertTransformer
    pooler::Dense
end

function AlbertModel(config::ALBERTConfig; atype=atype())
    embeddings = ALBERTEmbedding(
        config.vocab_size,
        config.type_vocab_size,
        config.embedding_size,
        config.max_position_embeddings,
        config.hidden_dropout_prob,
        atype=atype
    )
    encoder = AlbertTransformer(
        config,
        atype=atype
    )
    pooler = Dense(config.hidden_size, config.hidden_size, activation="tanh")
    return AlbertModel(
        config,
        embeddings,
        encoder,
        pooler
    )
end

function (am::AlbertModel)(
        input_ids;  # TxB
        attention_mask=nothing,
        token_type_ids=nothing,
        position_ids=nothing,
        head_mask=nothing,
        output_attentions=false,
        output_hidden_states=false,
        return_dict=true
    )
    
    embeds = am.embeddings(input_ids, token_type_ids, position_ids)
    
    if attention_mask == nothing
        attention_mask = ones(Int, size(input_ids)...)
    end
    
    encoder_outputs = am.encoder(
        embeds, 
        attention_mask=attention_mask,
        head_mask=head_mask,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        return_dict=return_dict
    )
    
    sequence_outputs = return_dict ? encoder_outputs["output"] : encoder_outputs[1]
    
    pooler_output = (am.pooler == nothing) ? nothing : am.pooler(sequence_outputs[:,1,:])
    
    if !return_dict
        return tuple(sequence_outputs, pooler_output, encoder_outputs[2:end]...)
    else
        result = Dict()
        result["last_hidden_state"]=sequence_outputs
        result["pooler_output"] = pooler_output
        if output_hidden_states; result["hiddens"]=encoder_outputs["hiddens"]; end
        if output_attentions; result["attentions"]=encoder_outputs["attentions"]; end        
        return result
    end
end

mutable struct AlbertMLMHead
    config::ALBERTConfig
    projection_layer
    decoder
    lnorm
end

function AlbertMLMHead(config::ALBERTConfig; atype=atype())
    projection_layer = Dense(config.hidden_size, config.embedding_size, activation=config.hidden_act, atype=atype)
    decoder=Linear(config.embedding_size, config.vocab_size, atype=atype)
    lnorm = LayerNorm(config.vocab_size, atype=atype) # They don't use dropout here
    AlbertMLMHead(config, projection_layer, decoder, lnorm)
end

function (mlmh::AlbertMLMHead)(x)
    x = mlmh.projection_layer(x)
    mlmh.decoder(mlmh.lnorm(x))
end

mutable struct AlbertForMaskedLM
    config::ALBERTConfig
    albert::AlbertModel
    predictions
end

function AlbertForMaskedLM(config::ALBERTConfig; atype=atype())
    AlbertForMaskedLM(
        config, 
        AlbertModel(config, atype=atype), 
        AlbertMLMHead(config, atype=atype)
    )
end

function (amlm::AlbertForMaskedLM)(
        x;
        attention_mask=nothing,
        token_type_ids=nothing,
        position_ids=nothing,
        head_mask=nothing,
        output_attentions=false,
        output_hidden_states=false,
        return_dict=true
    )
    x = amlm.albert(
        x, 
        attention_mask=attention_mask,
        token_type_ids=token_type_ids,
        position_ids=position_ids,
        head_mask=head_mask,
        # input_embeds=inputs_embeds,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        return_dict=return_dict
    )
    

    sequence_outputs = return_dict ? x["last_hidden_state"] : x[1]
    
    logits = amlm.predictions(sequence_outputs)
    
    if !return_dict
        return tuple(logits, outputs[3:end]...)
    else
        result = Dict()
        result["logits"]=logits
        if output_hidden_states; result["hiddens"]=x["hiddens"]; end
        if output_attentions; result["attentions"]=x["attentions"]; end
        return result
    end
end

function (amlm::AlbertForMaskedLM)(
        x,
        labels,
        average=true;
        attention_mask=nothing,
        token_type_ids=nothing,
        position_ids=nothing,
        head_mask=nothing,
        output_attentions=false,
        output_hidden_states=false,
        return_dict=true
    )
    loss = 0
    output = amlm(x, 
        attention_mask=attention_mask,
        token_type_ids=token_type_ids,
        position_ids=position_ids,
        head_mask=head_mask,
        # inputs_embeds=inputs_embeds,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        return_dict=return_dict)
    
    output= return_dict ? output["logits"] : output[1]
    
    return nll(output, labels, average=average)
end

mutable struct AlbertForSequenceClassification
    config::ALBERTConfig
    num_labels
    albert::AlbertModel
    classifier
    pdrop
end

function AlbertForSequenceClassification(config::ALBERTConfig, num_labels, classifier_dropout_prob; atype=atype())
    albert = AlbertModel(config, atype=atype)
    classifier=Linear(config.hidden_size, num_labels, atype=atype)
    AlbertForSequenceClassification(config, num_labels, albert, classifier, classifier_dropout_prob)
end

function (asc::AlbertForSequenceClassification)(
        x;
        attention_mask=nothing,
        token_type_ids=nothing,
        position_ids=nothing,
        head_mask=nothing,
        output_attentions=false,
        output_hidden_states=false,
        return_dict=true
    )
    
    x = asc.albert(
        x, 
        attention_mask=attention_mask,
        token_type_ids=token_type_ids,
        position_ids=position_ids,
        head_mask=head_mask,
        inputs_embeds=inputs_embeds,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        return_dict=return_dict
    )
    
    pooler_output = return_dict ? x["pooler_output"] : x[2]
    
    pooler_output = dropout(pooler_outout, asc.pdrop)
    
    logits = asc.classifier(pooler_output)
    
    if !return_dict
        return tuple(logits, outputs[3:end]...)
    else
        result = Dict()
        result["logits"]=logits
        result["hidden_states"]=result["hidden_states"]
        result["attentions"]=result["attentions"]
        return result
    end
end

function (asc::AlbertForSequenceClassification)(
        x,
        labels,
        average=true;
        attention_mask=nothing,
        token_type_ids=nothing,
        position_ids=nothing,
        head_mask=nothing,
        output_attentions=false,
        output_hidden_states=false,
        return_dict=true
    )
    loss = 0
    output = asc(x, 
        attention_mask=attention_mask,
        token_type_ids=token_type_ids,
        position_ids=position_ids,
        head_mask=head_mask,
        inputs_embeds=inputs_embeds,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        return_dict=return_dict)
    
    output= return_dict ? output["logits"] : output[1]
    
    if asc.num_labels == 1 # Regression
        loss = sum((reshape(output, :).-reshape(labels, :)).^2)
        return average ? loss/length(labels) : (loss, labels)
    else 
        return nll(output, reshape(labels, :), average=average)
    end
end

# Only loads base model at the moment (doesn't load weights for heads)
function pretrainedAlbertModel(modelpath::AbstractString, modelconfig; atype=atype())
    if typeof(modelconfig)<:AbstractString
        modelconfig = ALBERTConfig(modelconfig)
    end
    
    model = AlbertModel(modelconfig, atype=atype)
    
    # Import model
    weights = torch.load(modelpath)
    initPretrainedAlbertModel(model, weights, atype=atype)
end
    
function initPretrainedAlbertModel(model::AlbertModel, weights::Dict; atype=atype())
    # Initialize embeddings
    model.embeddings.word_embeds.w = Param(atype(weights["albert.embeddings.word_embeddings.weight"][:cpu]()[:numpy]()'));
    model.embeddings.token_type_embeds.w = Param(atype(weights["albert.embeddings.token_type_embeddings.weight"][:cpu]()[:numpy]()'));
    model.embeddings.pos_embeds.w = Param(atype(weights["albert.embeddings.position_embeddings.weight"][:cpu]()[:numpy]()'));
    model.embeddings.lnorm.a = Param(atype(weights["albert.embeddings.LayerNorm.weight"][:cpu]()[:numpy]()));
    model.embeddings.lnorm.b = Param(atype(weights["albert.embeddings.LayerNorm.bias"][:cpu]()[:numpy]()));
    
    # Initialize embedding->hidden projection layer
    model.encoder.embedding_hidden_mapping.w = Param(atype(weights["albert.encoder.embedding_hidden_mapping_in.weight"][:cpu]()[:numpy]()))
    model.encoder.embedding_hidden_mapping.b = Param(atype(weights["albert.encoder.embedding_hidden_mapping_in.bias"][:cpu]()[:numpy]()))
    
    # Initialize layers
    for group_idx in 1:length(model.encoder.albert_layer_groups)
        group = model.encoder.albert_layer_groups[group_idx]
        for layer_idx in 1:length(group.albert_layers)
            layer = group.albert_layers[layer_idx]
            
            prefix = "albert.encoder.albert_layer_groups.$(group_idx-1).albert_layers.$(layer_idx-1)."
            # Attention part
            attn = layer.attn_block
            # Attention lnorm
            attn.lnorm.a = Param(atype(weights["albert.encoder.albert_layer_groups.0.albert_layers.0.attention.LayerNorm.weight"][:cpu]()[:numpy]()))
            attn.lnorm.b = Param(atype(weights["albert.encoder.albert_layer_groups.0.albert_layers.0.attention.LayerNorm.bias"][:cpu]()[:numpy]()))
            # Attention weights
            attn.attn_layer.q_proj.w = Param(atype(reshape(weights["albert.encoder.albert_layer_groups.0.albert_layers.0.attention.query.weight"][:cpu]()[:numpy](),size(attn.attn_layer.q_proj.w))))
            attn.attn_layer.q_proj.b = Param(atype(reshape(weights["albert.encoder.albert_layer_groups.0.albert_layers.0.attention.query.bias"][:cpu]()[:numpy](),size(attn.attn_layer.q_proj.b))))
            attn.attn_layer.v_proj.w = Param(atype(reshape(weights["albert.encoder.albert_layer_groups.0.albert_layers.0.attention.value.weight"][:cpu]()[:numpy](),size(attn.attn_layer.v_proj.w))))
            attn.attn_layer.v_proj.b = Param(atype(reshape(weights["albert.encoder.albert_layer_groups.0.albert_layers.0.attention.value.bias"][:cpu]()[:numpy](),size(attn.attn_layer.v_proj.b))))
            attn.attn_layer.k_proj.w = Param(atype(reshape(weights["albert.encoder.albert_layer_groups.0.albert_layers.0.attention.key.weight"][:cpu]()[:numpy](),size(attn.attn_layer.k_proj.w))))
            attn.attn_layer.k_proj.b = Param(atype(reshape(weights["albert.encoder.albert_layer_groups.0.albert_layers.0.attention.key.bias"][:cpu]()[:numpy](),size(attn.attn_layer.k_proj.b))))
            attn.attn_layer.o_proj.w = Param(atype(reshape(weights["albert.encoder.albert_layer_groups.0.albert_layers.0.attention.dense.weight"][:cpu]()[:numpy](),size(attn.attn_layer.o_proj.w))))
            attn.attn_layer.o_proj.b = Param(atype(reshape(weights["albert.encoder.albert_layer_groups.0.albert_layers.0.attention.dense.bias"][:cpu]()[:numpy](),size(attn.attn_layer.o_proj.b))))

            # Feed Forward Network
            ffn = layer.ffn_block
            # FFN lnorm
            ffn.ffn_sublayer.norm.a = Param(atype(weights["albert.encoder.albert_layer_groups.0.albert_layers.0.full_layer_layer_norm.weight"][:cpu]()[:numpy]()))
            ffn.ffn_sublayer.norm.b = Param(atype(weights["albert.encoder.albert_layer_groups.0.albert_layers.0.full_layer_layer_norm.bias"][:cpu]()[:numpy]()))
            # FFN weights
            ffn.ffn_sublayer.layer.fc1.w = Param(atype(weights["albert.encoder.albert_layer_groups.0.albert_layers.0.ffn.weight"][:cpu]()[:numpy]()))
            ffn.ffn_sublayer.layer.fc1.b = Param(atype(weights["albert.encoder.albert_layer_groups.0.albert_layers.0.ffn.bias"][:cpu]()[:numpy]()))
            ffn.ffn_sublayer.layer.fc2.w = Param(atype(weights["albert.encoder.albert_layer_groups.0.albert_layers.0.ffn_output.weight"][:cpu]()[:numpy]()))
            ffn.ffn_sublayer.layer.fc2.b = Param(atype(weights["albert.encoder.albert_layer_groups.0.albert_layers.0.ffn_output.bias"][:cpu]()[:numpy]()))
        end
    end
    
    # Initialize pooler
    model.pooler.w = Param(atype(weights["albert.pooler.weight"][:cpu]()[:numpy]()))
    model.pooler.b = Param(atype(weights["albert.pooler.bias"][:cpu]()[:numpy]()))
    
    model
end

function pretrainedAlbertForMLM(modelpath::AbstractString, modelconfig; atype=atype())
    if typeof(modelconfig)<:AbstractString
        modelconfig = ALBERTConfig(modelconfig)
    end
    
    model = AlbertModel(modelconfig, atype=atype)
    
    # Import model
    weights = torch.load(modelpath)
    albertModel = initPretrainedAlbertModel(AlbertModel(modelconfig, atype=atype), weights, atype=atype)
    
    MLMHead = AlbertMLMHead(modelconfig, atype=atype)
    MLMHead = initPretrainedAlbertForMLM(MLMHead, weights, atype=atype)
    
    AlbertForMaskedLM(modelconfig,albertModel,MLMHead)
end

function initPretrainedAlbertForMLM(model::AlbertMLMHead, weights::Dict; atype=atype())
    model.projection_layer.w = Param(atype(weights["predictions.dense.weight"][:cpu]()[:numpy]()))
    model.projection_layer.b = Param(atype(weights["predictions.dense.bias"][:cpu]()[:numpy]()))
    model.decoder.w = Param(atype(weights["predictions.decoder.weight"][:cpu]()[:numpy]()))
    # model.decoder.b = Param(atype(weights["predictions.decoder.bias"][:cpu]()[:numpy]())) # "predictions.decoder.bias" is all zeros, refer to https://docs.google.com/document/d/1MMoR9aq0JYVj1hYxcWxGWzBqwq_KFtUs4zMW1-fLsPg/edit#
    model.decoder.b = Param(atype(weights["predictions.bias"][:cpu]()[:numpy]())) 
    model.lnorm.a = Param(atype(weights["predictions.LayerNorm.weight"][:cpu]()[:numpy]()))
    model.lnorm.b = Param(atype(weights["predictions.LayerNorm.bias"][:cpu]()[:numpy]()))
    model
end

