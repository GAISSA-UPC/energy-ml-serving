from optimum.exporters.tasks import TasksManager

model_type = 'gpt-neox'
#t5, ['feature-extraction', 'feature-extraction-with-past', 'text2text-generation', 'text2text-generation-with-past']
#codegen,['feature-extraction', 'feature-extraction-with-past', 'text-generation', 'text-generation-with-past']
# gpt-neo, ['feature-extraction', 'feature-extraction-with-past', 'text-generation', 'text-generation-with-past', 'text-classification']
# gpt2, ['feature-extraction', 'feature-extraction-with-past', 'text-generation', 'text-generation-with-past', 'text-classification', 'token-classification']

model_checkpoint = {'codet5-base':"Salesforce/codet5-base", 'codet5p-220':'Salesforce/codet5p-220m', 
                    'codegen-350-mono':"Salesforce/codegen-350M-mono", 'gpt-neo-125m':"EleutherAI/gpt-neo-125M",
                    'codeparrot-small':'codeparrot/codeparrot-small', 'pythia-410m':"EleutherAI/pythia-410m"} # model:checkpoint

#optimum-cli export onnx --task text2text-generation --model "Salesforce/codet5-base" models/onnx_2/codet5-base
#optimum-cli export onnx --task text2text-generation-with-past --model 'Salesforce/codet5p-220m' models/onnx_2/codet5p-220
#optimum-cli export onnx --task text-generation-with-past --model "Salesforce/codegen-350M-mono" models/onnx_2/codegen-350M-mono, killeds
#optimum-cli export onnx --task text-generation-with-past --model "EleutherAI/gpt-neo-125M" models/onnx_2/gpt-neo-125M
#optimum-cli export onnx --task text-generation-with-past --model 'codeparrot/codeparrot-small' models/onnx_2/codeparrot-small
#optimum-cli export onnx --task text-generation-with-past --model "EleutherAI/gpt-neo-125M" models/onnx_2/gpt-neo-125M
#optimum-cli export onnx --task text-generation-with-past --model "EleutherAI/pythia-410m" models/onnx_2/pythia-410m # killed





# For instance, for the ONNX export.
backend = "onnx"
tasks = list(TasksManager.get_supported_tasks_for_model_type(model_type, backend).keys())

print(tasks)