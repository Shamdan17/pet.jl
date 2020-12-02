include("data.jl")
include("metrics.jl")
using Random
using Statistics
using ArgParse


struct model
    predictor # This model takes a datum struct and outputs a prediction
    evaluation_metric::AbstractArray # The identifiers of evaluation metrics used
end

function (m::model)(x)
    if typeof(x)<:AbstractArray
        m.predictor.(x)
    else
        m.predictor(x)
    end
end

function (m::model)(x, y)
    preds = m(x)
    # Expand
    preds_exp = []
    [(typeof(x)<:AbstractArray) ? [push!(preds_exp, i) for i in x] : push!(preds_exp, x) for x in preds]
    y_exp = []
    [(typeof(x)<:AbstractArray) ? [push!(y_exp, i) for i in x] : push!(y_exp, x) for x in y]

    metricvals = []
    for mname in m.evaluation_metric
        # Get metric
        metric = registeredmetrics[mname]()
        # Evaluate metric on 
        push!(metricvals, (mname, mean([metric(a, b) for (a, b) in zip(preds_exp, y_exp)])))
    end
    
    join(["$mname:$val" for (mname, val) in metricvals], "\t")
end


# Returns a random choice as the label
# ReCoRD and MultiRC have multiple choices and labels of each choice are 0/1
function getRandomBaseline(datatypename::String, metric::AbstractArray, trainingpath::String, validationpath::String)

    multilabel = datatypename == "ReCoRD" || datatypename == "MultiRC"
    function predictor(x)
        choices = getChoices(x)
        if multilabel
            prediction = [Random.rand((0, 1), (x)) for x in length.(choices)]
        else
            prediction = rand(choices)
        end
    end
    model(predictor, metric)
end

# Returns the most common option as the label
# ReCoRD's methodology is not properly defined in the SuperGLUE paper so I could not replicate their method
# Most common label for MultiRC options is 0 -> F1 score is 0 because no True Positives
function getMostCommonAnswerBaseline(datatypename::String, metric::AbstractArray, trainingpath::String, validationpath::String)

    @assert datatypename != "ReCoRD" "ReCoRD not supported yet."
    multilabel = datatypename == "ReCoRD" || datatypename == "MultiRC"
    data = registeredtypes[datatypename].(readlines(trainingpath))
    dct = Dict{Any, Int}()
    mx = -1
    mxlbl = -1
    for obj in data
        labels = collect(getLabel(obj))
        for lbl in labels
            dct[lbl] = get(dct, lbl, 0) + 1
            if dct[lbl] > mx
                mx = dct[lbl]
                mxlbl = lbl
            end
        end
    end
    
    function predictor(x)
        choices = getChoices(x)
        if multilabel
            prediction = [[mxlbl for i in 1:x] for x in length.(choices)]
        else
            prediction = mxlbl
        end
    end
    model(predictor, metric)
end


# Gets the ygold labels and evaluate
function evaluate(model, data)
    ygold = [getLabel(x) for x in data]
    model(data, ygold)
end

# Datasets and corresponding metrics
datatypemetrics = Dict([
        ("BoolQ", ["Acc"]),
        ("CB", ["Acc"]),
        ("COPA", ["Acc"]),
        ("MultiRC", ["Acc", "F1"]),
        ("ReCoRD", ["EM", "F1"]),
        ("RTE", ["Acc"]),
        ("WiC", ["Acc"]),
        ("WSC", ["Acc"])
        ])



function parse_commandline()
    s = ArgParseSettings()
    @add_arg_table s begin
        "--dataset", "-d"
            help = "The dataset to evaluate. Possible options: BoolQ, CB, COPA, MultiRC, ReCoRD, RTE, WiC, WSC, all"
            arg_type = String
            default = "all"
        "--method", "-m"
            help = "The Baseline type. Possible options: Random, MostCommon"
            arg_type = String
            default = "all"
    end

    return parse_args(s)
end

function main()
    parsed_args = parse_commandline()

    metrics_to_eval = Dict()

    if parsed_args["dataset"]!="all"
        metrics_to_eval[parsed_args["dataset"]] = datatypemetrics[parsed_args["dataset"]] 
    else
        for (dataset, metric) in datatypemetrics
            metrics_to_eval[dataset] = metric
        end
    end

    if parsed_args["method"] == "Random" || parsed_args["method"]=="all"
        println("=================================================")
        println("Most Common Answer Baseline:")
        for (dataset, metric) in metrics_to_eval
            root_path = joinpath("../data", "SuperGLUE", dataset)

            train_path = joinpath(root_path, "train.jsonl")
            val_path = joinpath(root_path, "val.jsonl")

            rndBaseline = getRandomBaseline(dataset, metric, train_path, val_path)
            validationpath = val_path
            datatype = registeredtypes[dataset]
            vld_instances = datatype.(readlines(validationpath))

            println("$dataset\t$(evaluate(rndBaseline, vld_instances))")
        end
    end


    if parsed_args["method"] == "MostCommon" || parsed_args["method"]=="all"
        println("=================================================")
        println("Random Baseline:")
        for (dataset, metric) in metrics_to_eval
            if dataset == "ReCoRD"; continue; end
            
            root_path = joinpath("../data", "SuperGLUE", dataset)

            train_path = joinpath(root_path, "train.jsonl")
            val_path = joinpath(root_path, "val.jsonl")

            rndBaseline = getMostCommonAnswerBaseline(dataset, metric, train_path, val_path)

            datatype = registeredtypes[dataset]
            vld_instances = datatype.(readlines(val_path))

            println("$dataset\t $(evaluate(rndBaseline, vld_instances))")
        end
    end
end


main()