using JSON
using Test
using Random

# A dictionary containing the names of all defined datatypes
registeredtypes = Dict{String, Any}()

export registeredtypes

# Register a datatype with a name
macro registerDataType(name::String, typ)
    :(registeredtypes[$name]=$typ)
end
# Directory of one line json files to test datatype constructors
testdatadir = joinpath("../data", "test")

# All data objects are subtypes of this abstract type
abstract type datum end

function getChoices(d::datum)
    error("Not implemented for this datatype. Please define the function for this datatype.")
end

function getLabel(d::datum)
    error("Not implemented for this datatype. Please define the function for this datatype.")
end

# Returns an array of datum objects with exclusively only 1 label per datum.
function flatten(d::datum; o...)
    [d]
end

function flatten(d::AbstractArray{<:datum}; o...)
    d
end


# BoolQ data object
# Original paper: https://arxiv.org/abs/1905.10044
mutable struct BoolQ <: datum
    question::String    # Question
    passage::String     # Passage
    label::Bool         # Whether the Question is true or false
    idx::Int            # Index in the original dataset
    labeled::Bool       # Whether the data object has a label. If false, the label is meaningless
    logits              # logits for all the labels
    BoolQ(question::String, passage::String, label::Bool, idx::Int) = new(question, passage, label, idx, true, nothing)
    BoolQ(question::String, passage::String, label::Nothing, idx::Int) = new(question, passage, false, idx, false, nothing)
    BoolQ(question::String, passage::String, idx::Int) = new(question, passage, false, idx, false, nothing)
end

@registerDataType "BoolQ" BoolQ

function BoolQ(json::AbstractString)
    contents = JSON.parse(json)
    BoolQ(contents["question"],
          contents["passage"],
          get(contents, "label", nothing),
          contents["idx"])
end

function getChoices(b::BoolQ)
    return [1, 2]
end

function getLabel(b::BoolQ)
    @assert b.labeled "This instance is not labeled."
    return b.label ? 2 : 1
end

function setLabel(b::BoolQ, label::Int)
    b.label = 2==label
    b.labeled = true
end

function setLabel(b::BoolQ, label::Bool)
    b.label = label
    b.labeled = true
end

function removeLabel(b::BoolQ)
    b.labeled = false
end

# CommitmentBank data object
# Original paper: https://semanticsarchive.net/Archive/Tg3ZGI2M/Marneffe.pdf
mutable struct CB <: datum
    premise::String     # Premise
    hypothesis::String  # Hypothesis
    label::String       # Whether the hypothesis is an entailment, contradiction, or neutral to the passage
    idx::String         # Index in the original dataset
    labeled::Bool       # Whether the data object has a label. If false, the label is meaningless
    logits              # logits for all the labels
    CB(premise::String, hypothesis::String,label::String, idx::String) = new(premise, hypothesis, label, idx, true, nothing)
    CB(premise::String, hypothesis::String,label::Nothing, idx::String) = new(premise, hypothesis, "None", idx, false, nothing)
    CB(premise::String, hypothesis::String, idx::String) = new(premise, hypothesis, "None", idx, false, nothing)
end

@registerDataType "CB" CB

function CB(json::AbstractString)
    contents = JSON.parse(json)
    CB(contents["premise"],
          contents["hypothesis"],
          get(contents, "label", nothing),
          "$(contents["idx"])")
end

function getChoices(c::CB)
    return [1, 2, 3]
end

function getLabel(c::CB)
    @assert c.labeled "This instance is not labeled."
    return argmax(["contradiction", "entailment", "neutral"].==c.label)
end


function setLabel(cb::CB, label::Int)
    cb.label = ["contradiction", "entailment", "neutral"][label]
    cb.labeled = true
end

function setLabel(cb::CB, label::String)
    cb.label = label
    cb.labeled = true
end

function removeLabel(cb::CB)
    cb.labeled = false
end

# COPA data object
# Original paper: https://people.ict.usc.edu/~gordon/publications/AAAI-SPRING11A.PDF
mutable struct COPA <: datum
    premise::String     # Premise
    choice1::String     # First choice
    choice2::String     # Second choice
    question::String    # Whether the right choice is the effect or the cause of the premise
    label::Int          # The correct choice
    idx::Int            # Index in the original dataset
    labeled::Bool       # Whether the data object has a label. If false, the label is meaningless
    logits              # logits for all the labels
    COPA(premise::String, choice1::String, choice2::String, question::String, label::Int64, idx::Int64) = new(premise, choice1, choice2, question, label, idx, true, nothing)
    COPA(premise::String, choice1::String, choice2::String, question::String, label::Nothing, idx::Int64) = new(premise, choice1, choice2, question, -1, idx, false, nothing)
    COPA(premise::String, choice1::String, choice2::String, question::String, idx::Int64) = new(premise, choice1, choice2, question, -1, idx, false, nothing)
end

@registerDataType "COPA" COPA

function COPA(json::AbstractString)
    contents = JSON.parse(json)
    label = get(contents,"label",nothing)
    if label != nothing
        label+=1
    end
    COPA(contents["premise"],
          contents["choice1"],
          contents["choice2"],
          contents["question"],
          label,
          contents["idx"])
end

function getChoices(c::COPA)
    return [1, 2]
end

function getLabel(c::COPA)
    @assert c.labeled "This instance is not labeled."
    return c.label
end

function setLabel(cp::COPA, label::Int)
    cp.label = label
    cp.labeled = true
end

function removeLabel(cp::COPA)
    cp.labeled = false
end

# Adds a mirrored copy of the instance
function flatten(d::COPA; o...)
    mr = deepcopy(d)
    tmp = mr.choice1
    mr.choice1 = mr.choice2
    mr.choice2 = tmp
    mr.label = 3-mr.label
    [d, mr]
end

function flatten(d::AbstractArray{<:COPA}; o...)
    flattened = [Iterators.flatten(flatten.(d))...]
end

# MultiRC data object
# Original paper: https://www.aclweb.org/anthology/N18-1023/
mutable struct MultiRCOption
    text::String
    label::Bool
    idx::Int
    labeled::Bool
end

function MultiRCOption(content::Dict)
    text = content["text"]
    label = get(content, "label", false)
    idx = content["idx"]
    labeled = "label" in keys(content)
    MultiRCOption(text, label, idx, labeled)
end

mutable struct MultiRCQuestion
    question::String
    options::Array{MultiRCOption}
    idx::Int
    labeled::Bool
end

function MultiRCQuestion(content::Dict)
    question = content["question"]
    idx = content["idx"]
    options = [MultiRCOption(o) for o in content["answers"]]
    labeled=options[1].labeled
    MultiRCQuestion(question, options, idx, labeled)
end

mutable struct MultiRC <: datum
    passage::String     # The passage
    questions::Array{MultiRCQuestion} # An array of questions about the passage along with answer options
    idx                 # Index in the original dataset
    labeled::Bool       # Whether the questions have their correct answers labeled
    logits              # logits for all the labels
    MultiRC(passage::String, questions::Array{MultiRCQuestion}, idx, labeled::Bool)=new(passage, questions, idx, labeled, nothing)
end

function MultiRC(json::AbstractString)
    content = JSON.parse(json)
    passage = content["passage"]["text"]
    questions = [MultiRCQuestion(q) for q in content["passage"]["questions"]]
    idx = content["idx"]
    labeled = questions[1].labeled
    MultiRC(passage, questions, idx, labeled)
end

function flatten(d::MultiRC; o...)
    flattened = []
    for q in d.questions
        for a in q.options
            curq = MultiRCQuestion(q.question, [a], q.idx, a.labeled)
            curmrc = MultiRC(d.passage, [curq], [d.idx, q.idx, a.idx], d.labeled)
            push!(flattened, curmrc)
        end
    end
    flattened
end

function flatten(d::AbstractArray{MultiRC}; o...)
    flattened = [Iterators.flatten(flatten.(d))...]
end

@registerDataType "MultiRC" MultiRC


function getChoices(m::MultiRC)
    if length(m.questions)==1&&length(m.questions[1].options)==1
        return m.questions[1].options[1].text
    else
        return [[a.text for a in q.options] for q in m.questions]
    end
end

function getLabel(m::MultiRC)
    @assert m.labeled "This instance is not labeled."
    if length(m.questions)==1&&length(m.questions[1].options)==1
        return m.questions[1].options[1].label ? 2 : 1
    else
        return [[a.label ? 2 : 1 for a in q.options] for q in m.questions]
    end
end


function setLabel(m::MultiRC, label::Int)
    @assert length(m.questions)==1 && length(m.questions[1].options)==1 "Only flattened MultiRC instances support setLabel"
    m.questions[1].options[1].label = label == 2
    m.labeled = true
    m.questions[1].labeled=true
    m.questions[1].options[1].labeled=true
end

function removeLabel(m::MultiRC)
    @assert length(m.questions)==1 && length(m.questions[1].options)==1 "Only flattened MultiRC instances support removeLabel"
    m.labeled = false
    m.questions[1].labeled=false
    m.questions[1].options[1].labeled=false
end


mutable struct ReCoRDAnswer
    start::Int
    ending::Int
    text::String
end

function ReCoRDAnswer(content::Dict, ctx::String)
    st = content["start"]
    nd = content["end"]
    indices = collect(eachindex(ctx))
    text = ctx[indices[st+1]:indices[nd+1]]
    ReCoRDAnswer(st, nd, text)
end

mutable struct ReCoRDQuestion
    query::String       
    answers::Array{ReCoRDAnswer}
    idx::Int
    labeled::Bool
end

function ReCoRDQuestion(content::Dict, ctx::String)
    query = content["query"]
    answers = [ReCoRDAnswer(q, ctx) for q in get(content, "answers", [])]
    idx = content["idx"]
    labeled = length(answers)>0
    ReCoRDQuestion(query, answers, idx, labeled)
end


mutable struct ReCoRDEntity
    start::Int
    ending::Int
    text::String
end

function ReCoRDEntity(content::Dict, ctx::String)
    st = content["start"]
    nd = content["end"]
    indices = collect(eachindex(ctx))
    text = ctx[indices[st+1]:indices[nd+1]]
    ReCoRDEntity(st, nd, text)
end

# ReCoRD data object
mutable struct ReCoRD <: datum
    source::String      # Text source
    text::String        # Context text
    entities::Array{ReCoRDEntity} # Array of entities
    qas::Array{ReCoRDQuestion} # Array of questions
    idx::Int            # Index in the original dataset
    labeled::Bool       # Whether the questions have their correct answers labeled
    logits              # logits for all the labels
    ReCoRD(source::String,text::String,entities::Array,qas::Array,idx::Int,labeled::Bool)=new(source,text,entities,qas,idx,labeled,nothing)
end

@registerDataType "ReCoRD" ReCoRD

function ReCoRD(json::AbstractString)
    content = JSON.parse(json)
    source = content["source"]
    text = content["passage"]["text"]
    # text = replace(content["passage"]["text"], "@highlight\n"=>"- ") -> Do this at the end, otherwise the indices are messed up
    entities = [ReCoRDEntity(q, text) for q in content["passage"]["entities"]]
    questions = [ReCoRDQuestion(q, text) for q in content["qas"]]
    labeled = questions[1].labeled
    idx = content["idx"]
    ReCoRD(source, text, entities, questions, idx, labeled)
end

function getChoices(r::ReCoRD)
    ents = [ent.text for ent in r.entities]
    # Get unique only
    ents = [ent for ent in Set(ents)]
    return [ents for q in r.qas]
end

# function getLabel(r::ReCoRD)
#     @assert r.labeled "This instance is not labeled."
#     ents = getChoices(r)
#     answers = [[(ent in [a.text for a in q.answers]) ? 1 : 0 for ent in ents[1]] for q in r.qas]
#     return answers
# end

function getLabel(r::ReCoRD)
    @assert r.labeled "This instance is not labeled."
    return 2
end


function flatten(d::ReCoRD; rng=MersenneTwister(42), max_train_candidates_per_question=10, set="train")
    flattened = []
    for q in d.qas
        answers = [a.text for a in q.answers]
        candidates = [e for e in getChoices(d)[1] if !(e in answers)]
        # Labeled
        if set=="train"
            for (idx, a) in enumerate(Set(answers))
                cur_candidates = candidates[randperm(rng, length(candidates))]
                if length(candidates)>max_train_candidates_per_question-1
                    cur_candidates = cur_candidates[1:(max_train_candidates_per_question-1)]
                end
                cur_entities = [ReCoRDEntity(-1, -1, x) for x in [a, cur_candidates...]]
                cur_answer = ReCoRDAnswer(-1, -1, a)
                cur_question = deepcopy(q)
                cur_question.answers = [cur_answer]

                cur_datum = deepcopy(d)
                cur_datum.entities = cur_entities
                cur_datum.qas = [cur_question]

                push!(flattened, cur_datum)
            end
        else #Unlabeled
            cur_question = deepcopy(q)

            cur_datum = deepcopy(d)
            cur_datum.qas = [cur_question]

            uniqueents = [ReCoRDEntity(-1, -1, e) for e in getChoices(d)[1]]
            cur_datum.entities = uniqueents

            push!(flattened, cur_datum)
        end
    end
    flattened
end

function flatten(d::AbstractArray{ReCoRD}; set="train")
    rng = MersenneTwister(42);
    flattened = [Iterators.flatten(flatten.(d, rng=rng, set=set))...]
end

# RTE data object
mutable struct RTE <: datum
    premise::String     # Premise
    hypothesis::String  # Hypothesis
    label::String       # Whether the hypothesis is an entailment, or not an entailment to the passage
    idx::String         # Index in the original dataset
    labeled::Bool       # Whether the questions have their correct answers labeled
    logits              # logits for all the labels
    RTE(premise::String, hypothesis::String, label::String, idx::String) = new(premise, hypothesis, label, idx, true, nothing)
    RTE(premise::String, hypothesis::String, label::Nothing, idx::String) = new(premise, hypothesis, "None", idx, false, nothing)
    RTE(premise::String, hypothesis::String, idx::String) = new(premise, hypothesis, "None", idx, false, nothing)
end

@registerDataType "RTE" RTE

function RTE(json::AbstractString)
    contents = JSON.parse(json)
    RTE(contents["premise"],
          contents["hypothesis"],
          get(contents,"label",nothing),
          "$(contents["idx"])")
end

function getChoices(r::RTE)
    return [1, 2]
end

function getLabel(r::RTE)
    @assert r.labeled "This instance is not labeled."
    return argmax(["entailment", "not_entailment"].==r.label)
end

function setLabel(r::RTE, label::Int)
    r.label = ["entailment", "not_entailment"][label]
    r.labeled = true
end

function setLabel(r::RTE, label::String)
    r.label = label
    r.labeled = true
end

function removeLabel(r::RTE)
    r.labeled = false
end


# WiC data object
mutable struct WiC <: datum
    word::String        # Shared word
    sentence1::String   # First sentence
    sentence2::String   # Second sentence
    label::Bool         # Whether the shared word is used in the same sense in both sentences
    start1::Int         # Start index of shared word in first sentence
    end1::Int           # End index of shared word in first sentence
    start2::Int         # Start index of shared word in second sentence
    end2::Int           # End index of shared word in second sentence 
    idx::Int            # Index in the original dataset
    labeled::Bool       # Whether the questions have their correct answers labeled
    logits              # logits for all the labels
    WiC(word::String, sentence1::String, sentence2::String, label::Bool, start1::Int64, end1::Int64, start2::Int64, end2::Int64, idx::Int64)=new(word, sentence1, sentence2, label, start1, end1, start2, end2, idx, true, nothing)  
    WiC(word::String, sentence1::String, sentence2::String, label::Nothing, start1::Int64, end1::Int64, start2::Int64, end2::Int64, idx::Int64)=new(word, sentence1, sentence2, false, start1, end1, start2, end2, idx, false, nothing)  
    WiC(word::String, sentence1::String, sentence2::String, start1::Int64, end1::Int64, start2::Int64, end2::Int64, idx::Int64)=new(word, sentence1, sentence2, false, start1, end1, start2, end2, idx, false, nothing)  
end

@registerDataType "WiC" WiC

function WiC(json::AbstractString)
    contents = JSON.parse(json)
    WiC(contents["word"],
        contents["sentence1"],
        contents["sentence2"],
        get(contents,"label",nothing),
        contents["start1"],
        contents["end1"],
        contents["start2"],
        contents["end2"],
        contents["idx"])
end


function getChoices(w::WiC)
    return [1, 2]
end

function getLabel(w::WiC)
    @assert w.labeled "This instance is not labeled."
    return w.label ? 2 : 1
end

function setLabel(w::WiC, label::Int)
    w.label = label==2
    w.labeled = true
end

function setLabel(w::WiC, label::Bool)
    w.label = label
    w.labeled = true
end

function removeLabel(w::WiC)
    w.labeled = false
end

# WSC data object
mutable struct WSC <: datum
    text::String        # Sentence
    label::Bool         # Whether the pronoun refers to the entity
    start1::Int         # Start index of the entity's span
    entity::String      # The entity starting at the given index
    start2::Int         # Start index of the pronoun in the sentence
    pronoun::String        # The pronoun starting at the given index 
    idx::Int            # Index in the original dataset
    labeled::Bool       # Whether the questions have their correct answers labeled
    logits              # logits for all the labels
    WSC(text::String, label::Bool, start1::Int64, entity::String, start2::Int64, pronoun::String, idx::Int64)=new(text, label, start1, entity, start2, pronoun, idx, true, nothing)
    WSC(text::String, label::Nothing, start1::Int64, entity::String, start2::Int64, pronoun::String, idx::Int64)=new(text, false, start1, entity, start2, pronoun, idx, false, nothing)
    WSC(text::String, start1::Int64, entity::String, start2::Int64, pronoun::String, idx::Int64)=new(text, false, start1, entity, start2, pronoun, idx, false, nothing)
end

@registerDataType "WSC" WSC

function WSC(json::AbstractString)
    contents = JSON.parse(json)
    WSC(contents["text"],
        get(contents,"label",nothing),
        contents["target"]["span1_index"]+1,
        contents["target"]["span1_text"],
        contents["target"]["span2_index"]+1,
        contents["target"]["span2_text"],
        contents["idx"])
end


function getChoices(w::WSC)
    return [1, 2]
end

function getLabel(w::WSC)
    @assert w.labeled "This instance is not labeled."
    return w.label ? 2 : 1
end


function setLabel(w::WSC, label::Int)
    w.label = label==2
    w.labeled = true
end

function setLabel(w::WSC, label::Bool)
    w.label = label
    w.labeled = true
end

function removeLabel(w::WSC)
    w.labeled = false
end

@testset "Testing datatype constructors" begin
    for (name, datatype) in registeredtypes
        filename = name*"-labeled.jsonl"
        txt = read(open(joinpath(testdatadir, filename), "r"), String)
        @test datatype(txt) != nothing
        @test getLabel(datatype(txt)) != nothing
        @test getChoices(datatype(txt)) != nothing
        

        filename = name*"-unlabeled.jsonl"
        txt = read(open(joinpath(testdatadir, filename), "r"), String)
        @test_throws AssertionError getLabel(datatype(txt))
        @test getChoices(datatype(txt)) != nothing
    end
end;
