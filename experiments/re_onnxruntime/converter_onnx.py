""" 
Script to save models with ONNX formats

Models: 

ORTModelForSeq2SeqLM -> t5 models
ORTModelForCausalLM -> codegen, gpt-neo, pythia (gptneox), codeparrot

OVModelForCausalLM

optimum-cli export onnx --model local_path --task question-answering distilbert_base_uncased_squad_onnx/

Commands used 
#optimum-cli export onnx --task text2text-generation --model "Salesforce/codet5-base" models/onnx_2/codet5-base
#optimum-cli export onnx --task text2text-generation-with-past --model 'Salesforce/codet5p-220m' models/onnx_2/codet5p-220
#optimum-cli export onnx --task text-generation-with-past --model "Salesforce/codegen-350M-mono" models/onnx_2/codegen-350M-mono, killeds
#optimum-cli export onnx --task text-generation-with-past --model "EleutherAI/gpt-neo-125M" models/onnx_2/gpt-neo-125M
#optimum-cli export onnx --task text-generation-with-past --model 'codeparrot/codeparrot-small' models/onnx_2/codeparrot-small
#optimum-cli export onnx --task text-generation-with-past --model "EleutherAI/gpt-neo-125M" models/onnx_2/gpt-neo-125M
#optimum-cli export onnx --task text-generation-with-past --model "EleutherAI/pythia-410m" models/onnx_2/pythia-410m # killed

python experiments/re_onnxruntime/converter_onnx.py;python experiments/re_onnxruntime/converter_openvino.py;
"""
import os
from optimum.onnxruntime import ORTModelForMaskedLM, ORTModelForQuestionAnswering, ORTModelForCausalLM, ORTModelForSeq2SeqLM
from transformers import AutoTokenizer

#from optimum.intel import OVModelForCausalLM

re = 'onnx'

model_checkpoints = ["bert-base-uncased", 't5-small', "Salesforce/codegen-350M-mono","EleutherAI/pythia-70m", "Salesforce/codet5p-220m"]

# ORTModelForSeq2SeqLM, ORTModelForSeq2SeqLM, ORTModelForCausalLM, ORTModelForSeq2SeqLM, ORTModelForSeq2SeqLM, ORTModelForSeq2SeqLM

#models = [ 'codet5-base', 'codet5p-220',  'gpt-neo-125m', 'codeparrot-small', 'pythia-410m'] # bloom, pythia
models = [ 'codet5-base', 'codet5p-220',  'codeparrot-small', 'pythia-410m'] # bloom, pythia

model_checkpoint = {'codet5-base':"Salesforce/codet5-base", 'codet5p-220':'Salesforce/codet5p-220m', 
                    'codegen-350M-mono':"Salesforce/codegen-350M-mono", 'gpt-neo-125m':"EleutherAI/gpt-neo-125M",
                    'codeparrot-small':'codeparrot/codeparrot-small', 'pythia-410m':"EleutherAI/pythia-410m"} # model:checkpoint

# codegen not converted in GCP vm
for model in models:
    save_directory = f"models/{re}/{model}/"
    if os.path.exists(save_directory):
        print(f"Model {model} is already exported...\n")
        continue
    try:
        # select checkpoint
        name = model
        checkpoint = model_checkpoint[name]
        model = name

        # change to your directory

        print(f"Saving {model} ...\n")
        print(f"Checkpoint {checkpoint} ...\n")


        # change the class
        # use_cache = True for text-generation, check transformers library if modeling_{model} has this functionality
        #ort_model = ORTModelForMaskedLM.from_pretrained(model_checkpoint, export=True, )

        if model in [ 'codet5-base','codet5p-220']:
            ort_model = ORTModelForSeq2SeqLM.from_pretrained(checkpoint, export=True, use_cache = True) # t5, 
        else:
            ort_model = ORTModelForCausalLM.from_pretrained(checkpoint, export=True, use_cache = True) # codegen, gpt-neo, pythia (gptneox), codeparrot


        # [codegen]
        tokenizer = AutoTokenizer.from_pretrained(checkpoint)


        # tokenizer = AutoTokenizer.from_pretrained(
        #     "EleutherAI/pythia-410m",
        #     revision="step3000",
        #     cache_dir="../pythia-410m/step3000",
        #     )

        text = "def hello_world():"

        # tokenize
        inputs = tokenizer(text, return_tensors="pt")


        # generate
        #tokens = ov_model.generate(**inputs)
        #print(tokens)
        # Save the onnx model and tokenizer
        ort_model.save_pretrained(save_directory)

        print(f"\nModel '{model}' successfully saved in '{save_directory}'")
        print(f"----------------------------------------------------------")

    except Exception as e:
        print(f"Error saving model: {e}")
        print(f"----------------------------------------------------------")

    #tokenizer.save_pretrained(save_directory)
    