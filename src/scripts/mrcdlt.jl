include("data.jl")

task_name = "MultiRC"
task_type = registeredtypes[task_name]

trnset = flatten(task_type.(readlines("../data/FewGLUE/$task_name/train.jsonl")));
unlblset = flatten(task_type.(readlines("../data/FewGLUE/$task_name/unlabeled.jsonl")));
evalst = flatten(task_type.(readlines("../data/SuperGLUE/$task_name/val.jsonl")));
metrics = ["acc"]

@show size.((trnset, unlblset, evalst))
