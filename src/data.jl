using JSON
using Test

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

# CommitmentBank data object
# Original paper: https://semanticsarchive.net/Archive/Tg3ZGI2M/Marneffe.pdf
mutable struct CB <: datum
    premise::String     # Premise
    hypothesis::String  # Hypothesis
    label::String       # Whether the hypothesis is an entailment, contradiction, or neutral to the passage
    idx::Int            # Index in the original dataset
    labeled::Bool       # Whether the data object has a label. If false, the label is meaningless
    logits              # logits for all the labels
    CB(premise::String, hypothesis::String,label::String, idx::Int) = new(premise, hypothesis, label, idx, true, nothing)
    CB(premise::String, hypothesis::String,label::Nothing, idx::Int) = new(premise, hypothesis, "None", idx, false, nothing)
    CB(premise::String, hypothesis::String, idx::Int) = new(premise, hypothesis, "None", idx, false, nothing)
end

@registerDataType "CB" CB

function CB(json::AbstractString)
    contents = JSON.parse(json)
    CB(contents["premise"],
          contents["hypothesis"],
          get(contents, "label", nothing),
          contents["idx"])
end

function getChoices(c::CB)
    return [1, 2, 3]
end

function getLabel(c::CB)
    @assert c.labeled "This instance is not labeled."
    return argmax(["entailment", "contradiction", "neutral"].==c.label)
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
    COPA(contents["premise"],
          contents["choice1"],
          contents["choice2"],
          contents["question"],
          get(contents,"label",nothing),
          contents["idx"])
end

function getChoices(c::COPA)
    return [0, 1]
end

function getLabel(c::COPA)
    @assert c.labeled "This instance is not labeled."
    return c.label
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
    idx::Int            # Index in the original dataset
    labeled::Bool       # Whether the questions have their correct answers labeled
end

function MultiRC(json::AbstractString)
    content = JSON.parse(json)
    passage = content["passage"]["text"]
    questions = [MultiRCQuestion(q) for q in content["passage"]["questions"]]
    idx = content["idx"]
    labeled = questions[1].labeled
    MultiRC(passage, questions, idx, labeled)
end

@registerDataType "MultiRC" MultiRC


function getChoices(m::MultiRC)
    return [[a.text for a in q.options] for q in m.questions]
end

function getLabel(m::MultiRC)
    @assert m.labeled "This instance is not labeled."
    return [[a.label for a in q.options] for q in m.questions]
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
end

@registerDataType "ReCoRD" ReCoRD

function ReCoRD(json::AbstractString)
    content = JSON.parse(json)
    source = content["source"]
    text = content["passage"]["text"]
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

function getLabel(r::ReCoRD)
    @assert r.labeled "This instance is not labeled."
    ents = getChoices(r)
    answers = [[(ent in [a.text for a in q.answers]) ? 1 : 0 for ent in ents[1]] for q in r.qas]
    return answers
end



# RTE data object
mutable struct RTE <: datum
    premise::String     # Premise
    hypothesis::String  # Hypothesis
    label::String       # Whether the hypothesis is an entailment, or not an entailment to the passage
    idx::Int            # Index in the original dataset
    labeled::Bool       # Whether the questions have their correct answers labeled
    logits              # logits for all the labels
    RTE(premise::String, hypothesis::String, label::String, idx::Int64) = new(premise, hypothesis, label, idx, true, nothing)
    RTE(premise::String, hypothesis::String, label::Nothing, idx::Int64) = new(premise, hypothesis, "None", idx, false, nothing)
    RTE(premise::String, hypothesis::String, idx::Int64) = new(premise, hypothesis, "None", idx, false, nothing)
end

@registerDataType "RTE" RTE

function RTE(json::AbstractString)
    contents = JSON.parse(json)
    RTE(contents["premise"],
          contents["hypothesis"],
          get(contents,"label",nothing),
          contents["idx"])
end

function getChoices(r::RTE)
    return [1, 2]
end

function getLabel(r::RTE)
    @assert r.labeled "This instance is not labeled."
    return argmax(["entailment", "not_entailment"].==r.label)
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
        contents["target"]["span1_index"],
        contents["target"]["span1_text"],
        contents["target"]["span2_index"],
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
