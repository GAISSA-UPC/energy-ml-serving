""" 
Script to save models with ONNX and OpenVINO formats

Temp files saved in, remove it:
C:\\Users\\[user]\\AppData\\Local\\Temp\\tmpoiu9iwxj\\*
"""
#from optimum.onnxruntime import ORTModelForMaskedLM, ORTModelForQuestionAnswering, ORTModelForCausalLM, ORTModelForSeq2SeqLM
import os
from transformers import AutoTokenizer
from optimum.intel import OVModelForCausalLM, OVModelForSeq2SeqLM

MAX_LENGTH = 128

re = 'ov'
model_checkpoints = ["bert-base-uncased", 't5-small', "Salesforce/codegen-350M-mono","EleutherAI/pythia-70m", "Salesforce/codet5p-220m"]
models = [ 'codet5-base', 'codet5p-220', 'codegen-350M-mono', 'gpt-neo-125m', 'codeparrot-small', 'pythia-410m'] # bloom, pythia
models = [ 'codet5-base', 'codet5p-220', 'codeparrot-small', 'pythia-410m'] # bloom, pythia
models = ['pythia-410m'] 
# 
#model_checkpoints = ["Salesforce/codet5-base", 'Salesforce/codet5p-220m', "Salesforce/codegen-350M-mono",
#                     "EleutherAI/gpt-neo-125M",'codeparrot/codeparrot-small']

model_checkpoint = {'codet5-base':"Salesforce/codet5-base", 'codet5p-220':'Salesforce/codet5p-220m', 
                    'codegen-350M-mono':"Salesforce/codegen-350M-mono", 'gpt-neo-125m':"EleutherAI/gpt-neo-125M",
                    'codeparrot-small':'codeparrot/codeparrot-small', 'pythia-410m':"EleutherAI/pythia-410m",
                    'tinyllama':'TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T',
                    'pythia1-4b':'EleutherAI/pythia-1.4b',
                    'phi2':'microsoft/phi-2',
                    'olmo':'allenai/OLMo-1B-hf'} # model:checkpoint

# codegen not converted in GCP vm

# select checkpoint

for model in models:
    save_directory = f"models/{re}/{model}/"
    if os.path.exists(save_directory):
        print(f"Model {model} is already exported...\n")
        continue
    try:
        #name = models[0]
        checkpoint = model_checkpoint[model]
        #model = name

        # change to your directory
        

        print(f"Saving {model} ...\n")
        print(f"Checkpoint {checkpoint} ...\n")

        if model in [ 'codet5-base','codet5p-220']:
            ov_model = OVModelForSeq2SeqLM.from_pretrained(checkpoint, export=True, use_cache = True) # t5,
        elif model in ['pythia-410m','pythia1-4b','tinyllama', 'phi2','olmo']:
            ov_model = OVModelForCausalLM.from_pretrained(checkpoint, export=True, use_cache = False)
        else:
            ov_model = OVModelForCausalLM.from_pretrained(checkpoint, export=True, use_cache = True) # codegen, gpt-neo, pythia (gptneox), codeparrot

        #ov_model = OVModelForCausalLM.from_pretrained("EleutherAI/pythia-410m",
        #    revision="step3000",
        #    cache_dir="../pythia-410m/step3000", export=True, use_cache = True)

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
        tokens = ov_model.generate(**inputs, max_length=MAX_LENGTH,no_repeat_ngram_size=2,)
        print(tokens)
        # Save the onnx model and tokenizer

        ov_model.save_pretrained(save_directory)
        print(f"\nModel '{model}' successfully saved in '{save_directory}'")

    except Exception as e:
        print(f"Error saving model: {e}")
        print(f"----------------------------------------------------------")
    
    print(f"----------------------------------------------------------")

    #tokenizer.save_pretrained(save_directory)