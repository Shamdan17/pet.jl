using JSON

struct ALBERTConfig
    vocab_size
    embedding_size
    hidden_size
    num_hidden_layers
    num_hidden_groups
    num_attention_heads
    intermediate_size
    inner_group_num
    hidden_act
    hidden_dropout_prob
    attention_probs_dropout_prob
    max_position_embeddings
    type_vocab_size
    initializer_range
    layer_norm_eps
    classifier_dropout_prob
    position_embedding_type
    pad_token_id
    bos_token_id
    eos_token_id
end

function ALBERTConfig(;vocab_size=30000,
                        embedding_size=128,
                        hidden_size=4096,
                        num_hidden_layers=12,
                        num_hidden_groups=1,
                        num_attention_heads=64,
                        intermediate_size=16384,
                        inner_group_num=1,
                        hidden_act="gelu_new",
                        hidden_dropout_prob=0,
                        attention_probs_dropout_prob=0,
                        max_position_embeddings=512,
                        type_vocab_size=2,
                        initializer_range=0.02,
                        layer_norm_eps=1e-12,
                        classifier_dropout_prob=0.1,
                        position_embedding_type="absolute",
                        pad_token_id=0,
                        bos_token_id=2,
                        eos_token_id=3)
    ALBERTConfig(vocab_size,
                embedding_size,
                hidden_size,
                num_hidden_layers,
                num_hidden_groups,
                num_attention_heads,
                intermediate_size,
                inner_group_num,
                hidden_act,
                hidden_dropout_prob,
                attention_probs_dropout_prob,
                max_position_embeddings,
                type_vocab_size,
                initializer_range,
                layer_norm_eps,
                classifier_dropout_prob,
                position_embedding_type,
                pad_token_id,
                bos_token_id,
                eos_token_id)
end

function ALBERTConfig(json::String)
    try
        if isfile(json)
            return ALBERTConfig(read(json, String))
        end
    catch e; end

    dct = JSON.parse(json)
    
    ALBERTConfig(
        attention_probs_dropout_prob = get(dct,"attention_probs_dropout_prob",0),
        bos_token_id = get(dct,"bos_token_id",2),
        classifier_dropout_prob = get(dct,"classifier_dropout_prob",0.1),
        embedding_size = get(dct,"embedding_size",128),
        eos_token_id = get(dct,"eos_token_id",3),
        hidden_act = get(dct,"hidden_act","gelu_new"),
        hidden_dropout_prob = get(dct,"hidden_dropout_prob",0),
        hidden_size = get(dct,"hidden_size",4096),
        initializer_range = get(dct,"initializer_range",0.02),
        inner_group_num = get(dct,"inner_group_num",1),
        intermediate_size = get(dct,"intermediate_size",16384),
        layer_norm_eps = get(dct,"layer_norm_eps",1e-12),
        max_position_embeddings = get(dct,"max_position_embeddings",512),
        num_attention_heads = get(dct,"num_attention_heads",64),
        num_hidden_groups = get(dct,"num_hidden_groups",1),
        num_hidden_layers = get(dct,"num_hidden_layers",12),
        pad_token_id = get(dct,"pad_token_id",0),
        type_vocab_size = get(dct,"type_vocab_size",2),
        vocab_size = get(dct,"vocab_size",30000),
    )
end
