
# huggingface model into pytorch

from transformers import BertTokenizer, BertForMaskedLM
from transformers import T5Tokenizer, T5ForConditionalGeneration
# [codegen]
from transformers import AutoModelForCausalLM, AutoTokenizer
# [pythia-70m]
from transformers import GPTNeoXForCausalLM, AutoTokenizer

from transformers import T5ForConditionalGeneration, AutoTokenizer

import torch


model_checkpoint = {'codet5-base':"Salesforce/codet5-base", 'codet5p-220':'Salesforce/codet5p-220m', 
                    'codegen-350-mono':"Salesforce/codegen-350M-mono", 'gpt-neo-125m':"EleutherAI/gpt-neo-125M",
                    'codeparrot-small':'codeparrot/codeparrot-small', 'pythia-410m':"EleutherAI/pythia-410m"} # model:checkpoint

models = [ 'codet5-base', 'codet5p-220', 'codegen-350-mono', 'gpt-neo-125m', 'codeparrot-small', 'pythia-410m'] # bloom, pythia

model_name = models[5]
checkpoint = model_checkpoint[model_name]

#tokenizer = T5Tokenizer.from_pretrained("t5-small")
#model = T5ForConditionalGeneration.from_pretrained("t5-small")

#checkpoint = "Salesforce/codegen-350M-mono"
#model = AutoModelForCausalLM.from_pretrained(checkpoint, device_map = 'auto', torch_dtype = 'auto')
model = AutoModelForCausalLM.from_pretrained(checkpoint, torch_dtype = 'auto')
#model = T5ForConditionalGeneration.from_pretrained(checkpoint, torch_dtype = 'auto') # t5 models
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

#change
save_directory = f"models/torch/{model_name}"

model.save_pretrained(save_directory, saved_model=True)

#torch.save(model.state_dict(), save_directory)