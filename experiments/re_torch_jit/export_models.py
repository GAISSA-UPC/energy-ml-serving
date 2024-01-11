"""This script export torchscript models
"""

from transformers import BertModel, BertTokenizer, BertConfig
import torch
from transformers import AutoTokenizer, AutoConfig
from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM


from transformers import T5ForConditionalGeneration
from transformers import T5Tokenizer, RobertaTokenizer

models = [ 'codet5-base', 'codet5p-220', 'codegen-350M-mono', 'gpt-neo-125m', 'codeparrot-small', 'pythia-410m'] # bloom, pythia
model_checkpoint = {'codet5-base':"Salesforce/codet5-base", 'codet5p-220':'Salesforce/codet5p-220m', 
                    'codegen-350M-mono':"Salesforce/codegen-350M-mono", 'gpt-neo-125m':"EleutherAI/gpt-neo-125M",
                    'codeparrot-small':'codeparrot/codeparrot-small', 'pythia-410m':"EleutherAI/pythia-410m"} # model:checkpoint


for model in models:
    try:
        # select checkpoint
        model_name = model
        checkpoint = model_checkpoint[model_name]

        # change to your directory
        save_directory = f"models/torchscript/{model_name}/"

        print(f"Saving {model_name} ...\n")
        print(f"Checkpoint {checkpoint} ...\n")

        #config = AutoConfig.from_pretrained(checkpoint,torchscript=True)


        if model_name in [ 'codet5-base', 'codet5p-220']:
            # OVModelForSeq2SeqLM
            #model = T5ForConditionalGeneration.from_pretrained(checkpoint, torchscript =True)
            model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint, torchscript =True)
            
            tokenizer = AutoTokenizer.from_pretrained(checkpoint, torchscript=True)
        else:
            model = AutoModelForCausalLM.from_pretrained(checkpoint, torchscript =True)
            tokenizer = AutoTokenizer.from_pretrained(checkpoint, torchscript=True)


        #tokenizer.add_special_tokens({'pad_token': '[PAD]'})

        dummy_input = "def hello_world():"
        #inputs = tokenizer.encode_plus(dummy_input,max_length = int(20),pad_to_max_length = True, add_special_tokens = True, return_tensors = 'pt')
        #inputs = tokenizer.encode_plus(dummy_input,max_length = int(20),padding = True, add_special_tokens = True, return_tensors = 'pt',truncation=True)
        inputs = tokenizer.encode_plus(dummy_input, return_tensors = 'pt',)
        print("inputs:",inputs)

        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        input_tuple = (input_ids,attention_mask, input_ids) # decoder_input_ids["input_ids"]

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
        print(f"----------------------------------------------------------")

    except Exception as e:
        print(f"Error saving model: {e}")
        print(f"----------------------------------------------------------")