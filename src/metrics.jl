# A dictionary containing the names of all defined evaluation metrics
registeredmetrics = Dict{String, Any}()

export registeredmetrics

# Register a metric with a name
macro registerMetric(name::String, mtrc)
    :(registeredmetrics[$name]=$mtrc)
end

# Accuracy metric
struct Accuracy; end

function (a::Accuracy)(x, y)
    return x == y
end
function (a::Accuracy)(x, y::AbstractArray)
    return x in y
end
function (a::Accuracy)(x::AbstractArray, y::AbstractArray)
    return mean(x .== y)
end
@registerMetric("Acc", Accuracy)

# At the moment Exact Match is the same as Accuracy. Will be updated if that changes.
@registerMetric("EM", Accuracy)


# F1 metric
struct F1; end

function (f::F1)(x::AbstractArray, y::AbstractArray)
    possible_labels = collect(Set(y))
    N = length(possible_labels)
    f1s=[]
    for lbl in possible_labels
        TP = sum((x.==lbl).&(y.==lbl))
        FP = sum((x.==lbl).&(y.!=lbl))
        FN = sum((x.!=lbl).&(y.==lbl))
        f1 = 2TP/(2TP+FP+FN)
        push!(f1s, f1)
    end
    return mean(f1s)
end
@registerMetric("F1", F1)
