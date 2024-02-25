
# huggingface model into pytorch

import os

# from transformers import BertTokenizer, BertForMaskedLM
# from transformers import T5Tokenizer, T5ForConditionalGeneration
# # [codegen]
# from transformers import , AutoTokenizer
# # [pythia-70m]
# from transformers import GPTNeoXForCausalLM, AutoTokenizer

from transformers import T5ForConditionalGeneration, AutoTokenizer, AutoModelForSeq2SeqLM,AutoModelForCausalLM

import torch


model_checkpoint = {'codet5-base':"Salesforce/codet5-base", 'codet5p-220':'Salesforce/codet5p-220m', 
                    'codegen-350-mono':"Salesforce/codegen-350M-mono", 'gpt-neo-125m':"EleutherAI/gpt-neo-125M",
                    'codeparrot-small':'codeparrot/codeparrot-small', 'pythia-410m':"EleutherAI/pythia-410m"} # model:checkpoint

models = [ 'codet5-base', 'codet5p-220', 'codegen-350-mono', 'gpt-neo-125m', 'codeparrot-small', 'pythia-410m'] # bloom, pythia
models = [ 'codet5-base', 'codet5p-220', 'codeparrot-small', 'pythia-410m'] # bloom, pythia

#model_name = models[5]
#checkpoint = model_checkpoint[model_name]

#tokenizer = T5Tokenizer.from_pretrained("t5-small")
#model = T5ForConditionalGeneration.from_pretrained("t5-small")

#checkpoint = "Salesforce/codegen-350M-mono"
#model = AutoModelForCausalLM.from_pretrained(checkpoint, torch_dtype = 'auto')

#tokenizer = AutoTokenizer.from_pretrained(checkpoint)


for model in models:
    save_directory = f"models/torch/{model}/"
    if os.path.exists(save_directory):
        print(f"Model {model} is already exported...\n")
        continue
    try:
        # select checkpoint
        checkpoint = model_checkpoint[model]

        # change to your directory

        print(f"Saving {model} ...\n")
        print(f"Checkpoint {checkpoint} ...\n")


        # change the class
        # use_cache = True for text-generation, check transformers library if modeling_{model} has this functionality
        #ort_model = ORTModelForMaskedLM.from_pretrained(model_checkpoint, export=True, )

        #model = AutoModelForCausalLM.from_pretrained(checkpoint, torch_dtype = 'auto')

        if model in [ 'codet5-base','codet5p-220']:
            torch_model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint)
            #ORTModelForSeq2SeqLM.from_pretrained(checkpoint, export=True, use_cache = True) # t5, 
        else:
            torch_model = AutoModelForCausalLM.from_pretrained(checkpoint)
            #ORTModelForCausalLM.from_pretrained(checkpoint, export=True, use_cache = True) # codegen, gpt-neo, pythia (gptneox), codeparrot

        # [codegen]
        tokenizer = AutoTokenizer.from_pretrained(checkpoint)

        torch_model.save_pretrained(save_directory, )

        print(f"\nModel '{model}' successfully saved in '{save_directory}'")
        print(f"----------------------------------------------------------")

    except Exception as e:
        print(f"Error saving model: {e}")
        print(f"----------------------------------------------------------")