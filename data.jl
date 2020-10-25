using JSON
using Test

# A dictionary containing the names of all defined datatypes
registeredtypes = Dict{String, Any}()
# Register a datatype with a name
macro registerDataType(name::String, typ)
    :(registeredtypes[$name]=$typ)
end
# Directory of one line json files to test datatype constructors
testdatadir = joinpath("data", "test")

# All data objects are subtypes of this abstract type
abstract type datum end



# BoolQ data object
# Original paper: https://arxiv.org/abs/1905.10044
struct BoolQ <: datum
    question::String    # Question
    passage::String     # Passage
    label::Bool         # Whether the Question is true or false
    idx::Int            # Index in the original dataset
    labeled::Bool       # Whether the data object has a label. If false, the label is meaningless
    BoolQ(question::String, passage::String, label::Bool, idx::Int) = new(question, passage, label, idx, true)
    BoolQ(question::String, passage::String, label::Nothing, idx::Int) = new(question, passage, false, idx, false)
    BoolQ(question::String, passage::String, idx::Int) = new(question, passage, false, idx, false)
end

@registerDataType "BoolQ" BoolQ

function BoolQ(json::AbstractString)
    contents = JSON.parse(json)
    BoolQ(contents["question"],
          contents["passage"],
          get(contents, "label", nothing),
          contents["idx"])
end



# CommitmentBank data object
# Original paper: https://semanticsarchive.net/Archive/Tg3ZGI2M/Marneffe.pdf
struct CB <: datum
    premise::String     # Premise
    hypothesis::String  # Hypothesis
    label::String       # Whether the hypothesis is an entailment, contradiction, or neutral to the passage
    idx::Int            # Index in the original dataset
    labeled::Bool       # Whether the data object has a label. If false, the label is meaningless
    CB(premise::String, hypothesis::String,label::String, idx::Int) = new(premise, hypothesis, label, idx, true)
    CB(premise::String, hypothesis::String,label::Nothing, idx::Int) = new(premise, hypothesis, "None", idx, false)
    CB(premise::String, hypothesis::String, idx::Int) = new(premise, hypothesis, "None", idx, false)
end

@registerDataType "CB" CB

function CB(json::AbstractString)
    contents = JSON.parse(json)
    CB(contents["premise"],
          contents["hypothesis"],
          get(contents, "label", nothing),
          contents["idx"])
end




# COPA data object
# Original paper: https://people.ict.usc.edu/~gordon/publications/AAAI-SPRING11A.PDF
struct COPA <: datum
    premise::String     # Premise
    choice1::String     # First choice
    choice2::String     # Second choice
    question::String    # Whether the right choice is the effect or the cause of the premise
    label::Int          # The correct choice
    idx::Int            # Index in the original dataset
    labeled::Bool       # Whether the data object has a label. If false, the label is meaningless
    COPA(premise::String, choice1::String, choice2::String, question::String, label::Int64, idx::Int64) = new(premise, choice1, choice2, question, label, idx, true)
    COPA(premise::String, choice1::String, choice2::String, question::String, label::Nothing, idx::Int64) = new(premise, choice1, choice2, question, -1, idx, false) 
    COPA(premise::String, choice1::String, choice2::String, question::String, idx::Int64) = new(premise, choice1, choice2, question, -1, idx, false)
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




# MultiRC data object
# Original paper: https://www.aclweb.org/anthology/N18-1023/
struct MultiRCOption
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

struct MultiRCQuestion
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

struct MultiRC <: datum
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




# TODO: ReCoRD




# RTE data object
struct RTE <: datum
    premise::String     # Premise
    hypothesis::String  # Hypothesis
    label::String       # Whether the hypothesis is an entailment, or not an entailment to the passage
    idx::Int            # Index in the original dataset
    labeled::Bool       # Whether the questions have their correct answers labeled
    RTE(premise::String, hypothesis::String, label::String, idx::Int64) = new(premise, hypothesis, label, idx, true)
    RTE(premise::String, hypothesis::String, label::Nothing, idx::Int64) = new(premise, hypothesis, "None", idx, false)
    RTE(premise::String, hypothesis::String, idx::Int64) = new(premise, hypothesis, "None", idx, false)
end

@registerDataType "RTE" RTE

function RTE(json::AbstractString)
    contents = JSON.parse(json)
    RTE(contents["premise"],
          contents["hypothesis"],
          get(contents,"label",nothing),
          contents["idx"])
end




# WiC data object
struct WiC <: datum
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
    WiC(word::String, sentence1::String, sentence2::String, label::Bool, start1::Int64, end1::Int64, start2::Int64, end2::Int64, idx::Int64)=new(word, sentence1, sentence2, label, start1, end1, start2, end2, idx, true)  
    WiC(word::String, sentence1::String, sentence2::String, label::Nothing, start1::Int64, end1::Int64, start2::Int64, end2::Int64, idx::Int64)=new(word, sentence1, sentence2, false, start1, end1, start2, end2, idx, false)  
    WiC(word::String, sentence1::String, sentence2::String, start1::Int64, end1::Int64, start2::Int64, end2::Int64, idx::Int64)=new(word, sentence1, sentence2, false, start1, end1, start2, end2, idx, false)  
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




# WSC data object
struct WSC <: datum
    text::String        # Sentence
    label::Bool         # Whether the pronoun refers to the entity
    start1::Int         # Start index of the entity's span
    entity::String      # The entity starting at the given index
    start2::Int         # Start index of the pronoun in the sentence
    pronoun::String        # The pronoun starting at the given index 
    idx::Int            # Index in the original dataset
    labeled::Bool       # Whether the questions have their correct answers labeled
    WSC(text::String, label::Bool, start1::Int64, entity::String, start2::Int64, pronoun::String, idx::Int64)=new(text, label, start1, entity, start2, pronoun, idx, true)
    WSC(text::String, label::Nothing, start1::Int64, entity::String, start2::Int64, pronoun::String, idx::Int64)=new(text, false, start1, entity, start2, pronoun, idx, false)
    WSC(text::String, start1::Int64, entity::String, start2::Int64, pronoun::String, idx::Int64)=new(text, false, start1, entity, start2, pronoun, idx, false)
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

@testset "Testing datatype constructors" begin
    for (name, datatype) in registeredtypes
        filename = name*"-labeled.jsonl"
        txt = read(open(joinpath(testdatadir, filename), "r"), String)
        @test datatype(txt) != nothing
        
        filename = name*"-unlabeled.jsonl"
        txt = read(open(joinpath(testdatadir, filename), "r"), String)
        @test datatype(txt) != nothing
    end
end
