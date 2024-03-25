"""This script export torchscript models
"""
import os

from transformers import BertModel, BertTokenizer, BertConfig
import torch
from transformers import AutoTokenizer, AutoConfig
from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM


from transformers import T5ForConditionalGeneration
from transformers import T5Tokenizer, RobertaTokenizer

MAX_LENGTH = 128

models = [ 'codet5-base', 'codet5p-220', 'codegen-350M-mono', 'gpt-neo-125m', 'codeparrot-small', 'pythia-410m'] # bloom, pythia
models = [ 'codet5-base', 'codet5p-220', 'codeparrot-small', 'pythia-410m'] # bloom, pythia
models = [ 'codet5-base', 'codet5p-220', ] # bloom, pythia

#models = [ 'tinyllama', ] 

model_checkpoint = {'codet5-base':"Salesforce/codet5-base", 'codet5p-220':'Salesforce/codet5p-220m', 
                    'codegen-350M-mono':"Salesforce/codegen-350M-mono", 'gpt-neo-125m':"EleutherAI/gpt-neo-125M",
                    'codeparrot-small':'codeparrot/codeparrot-small', 'pythia-410m':"EleutherAI/pythia-410m",
                    'tinyllama':'TinyLlama/TinyLlama-1.1B-intermediate-step-1195k-token-2.5T'} # model:checkpoint


for model in models:
    save_directory = f"models/torchscript/{model}.pt"
    if os.path.exists(save_directory):
        print(f"Model {model} is already exported...\n")
        continue
    try:
        # select checkpoint
        model_name = model
        checkpoint = model_checkpoint[model_name]

        print(f"Saving {model_name} ...\n")
        print(f"Checkpoint {checkpoint} ...\n")

        #config = AutoConfig.from_pretrained(checkpoint,torchscript=True)


        if model_name in [ 'codet5-base', 'codet5p-220']:
            # OVModelForSeq2SeqLM
            #model = T5ForConditionalGeneration.from_pretrained(checkpoint, torchscript =True)
            model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint, torchscript =True)
            
            tokenizer = AutoTokenizer.from_pretrained(checkpoint, torchscript=True)
        else:
            model = AutoModelForCausalLM.from_pretrained(checkpoint, torchscript =True, attn_implementation="eager")
            tokenizer = AutoTokenizer.from_pretrained(checkpoint, torchscript=True)


        #tokenizer.add_special_tokens({'pad_token': '[PAD]'})

        #if condition:\n            return condition\n        else:\n            return None\n\n    def _get_condition(self
        dummy_input = '''def rolling_min(numbers: List[int]) -> int:
            """
            Return the minimum number of numbers in the rolling window.

            :param numbers: The numbers to return.
            :return: The minimum number of numbers in the rolling window.
            """
            return min(numbers)'''
        dummy_input = "def rolling_max(numbers: List[int]) ->"
        #dummy_input = "if condition:\n            return condition\n        else:\n            return None\n\n    def _get_condition(self"
        #inputs = tokenizer.encode_plus(dummy_input,max_length = int(20),pad_to_max_length = True, add_special_tokens = True, return_tensors = 'pt')
        #inputs = tokenizer.encode_plus(dummy_input,max_length = int(20),padding = True, add_special_tokens = True, return_tensors = 'pt',truncation=True)
        #inputs = tokenizer.encode_plus(dummy_input, return_tensors = 'pt', max_length = int(20),padding = True)
        if model_name in ['codeparrot-small', 'pythia-410m','tinyllama']:
            #tokenizer.add_special_tokens({'pad_token': '[PAD]'})
            tokenizer.pad_token = tokenizer.eos_token
            inputs = tokenizer.encode_plus(dummy_input,max_length = MAX_LENGTH, padding = 'max_length', return_tensors = 'pt',truncation='only_second')
        else:
            inputs = tokenizer.encode_plus(dummy_input,max_length = MAX_LENGTH, padding = 'max_length', return_tensors = 'pt')
        #inputs = tokenizer(dummy_input, return_tensors="pt", truncation=True, max_length=int(30))
        print("inputs:",inputs)

        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        # none_int_tensor = torch.full((128,), -1, dtype=torch.int)
        # none_int_tensor = input_ids[:, -1:]
        # none_int_tensor = torch.ones((1, 128), dtype=torch.int64)
        # sos_token_id = tokenizer.bos_token_id  # or `tokenizer.cls_token_id` if `bos_token_id` is not set
        # print("sos_token_id: ", sos_token_id)
        # decoder_input_ids = torch.tensor([[sos_token_id]])  # Shape: [1, 1]
        # print("decoder_input_ids: ",decoder_input_ids)
        input_tuple = (input_ids,attention_mask, input_ids) # decoder_input_ids["input_ids"]

        # If your model has multiple inputs they must passed in order that is defined by your mode
        # script mode by usign torch.jit.trace
        if model_name in [ 'codet5-base','codet5p-220']:
            traced_model = torch.jit.trace(model, input_tuple) # t5 models
            #traced_model = torch.jit.script(model) # not supported for some functions definitions
        else:
            traced_model = torch.jit.trace(model, [input_ids])



        save_directory = f"models/torchscript/{model_name}.pt"

        #traced model is a TorchScript module
        torch.jit.save(traced_model,save_directory)
        print(f"\nModel '{model}' successfully saved in '{save_directory}'")

    except Exception as e:
        print(f"Error saving model: {e}")
        print(f"----------------------------------------------------------")
    
    print(f"----------------------------------------------------------")
    