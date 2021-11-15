include("modeling.jl")

label_list=[1, 2];
dtype=KnetArray{Float32}
albcnf = ALBERTConfig("xxlargev2/config.json")

task_name = "MultiRC"
task_type = registeredtypes[task_name]

trnset = flatten(task_type.(readlines("../data/FewGLUE/$task_name/train.jsonl")));
unlblset = flatten(task_type.(readlines("../data/FewGLUE/$task_name/unlabeled.jsonl")));
evalst = flatten(task_type.(readlines("../data/SuperGLUE/$task_name/val.jsonl")));
metrics = ["acc"]

pet_model_cnfg = WrapperConfig(albcnf, "xxlargev2/pytorch_model.bin", "mlm", lowercase(task_name), 512, label_list, 1)

sc_model_cnfg = WrapperConfig(albcnf, "xxlargev2/pytorch_model.bin", "sc", lowercase(task_name), 512, label_list, 1)

#pet_train_cnf = TrainConfig(weight_decay=0.01,learning_rate=1e-5, num_train_epochs=3,batch_size=8,gradient_accumulation_steps=2, temperature=2, max_steps=250)
#pet_eval_cnf = EvalConfig(metrics=metrics, batch_size=32)

pet_train_cnf = TrainConfig(weight_decay=0.01,learning_rate=1e-5, num_train_epochs=3,batch_size=2,gradient_accumulation_steps=8, temperature=2, max_steps=250)
pet_eval_cnf = EvalConfig(metrics=metrics, batch_size=16)



sc_train_cnf = TrainConfig(weight_decay=0.01,learning_rate=1e-5, use_logits=true, num_train_epochs=1, max_steps=5000,gradient_accumulation_steps=8, temperature=2, batch_size=2)
sc_eval_cnf = EvalConfig(metrics=metrics, batch_size=16)

train_pet(
    ensemble_model_config=pet_model_cnfg, 
    ensemble_train_config=pet_train_cnf, 
    ensemble_eval_config=pet_eval_cnf, 
    final_model_config=sc_model_cnfg, 
    final_train_config=sc_train_cnf, 
    final_eval_config=sc_eval_cnf,
    pattern_ids=[1, 2, 3], 
    output_dir="outputs/multrirc/multirc_ipet",
    ensemble_repetitions=3, final_repetitions=1,
    reduction="wmean", train_data=trnset, unlabeled_data=unlblset,
    eval_data=evalst, do_train=true, do_eval=true)
