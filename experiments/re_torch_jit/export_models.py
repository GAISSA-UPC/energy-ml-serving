"""This script export torchscript models

nohup python3 experiments/re_torch_jit/export_models.py > experiments/re_torch_jit/export_jit.out 2>&1 &

"""
import os
import torch
from transformers import AutoTokenizer
from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM
from transformers import AutoConfig


from transformers import GemmaTokenizer, AutoModelForCausalLM

# Enable Device-Side Assertions
#os.environ["TORCH_USE_CUDA_DSA"] = "1"

MAX_LENGTH = 128

# do it one by one, Error saving model: CUDA out of memory. Tried to allocate 144.00 MiB. GPU 
models = [ 'codet5-base', 'codet5p-220', 'codegen-350M-mono', 'gpt-neo-125m', 'codeparrot-small', 'pythia-410m'] # bloom, pythia
models = [ 'codet5-base', 'codet5p-220', 'codeparrot-small', 'pythia-410m'] # bloom, pythia
#models = [ 'codet5-base', 'codet5p-220', ] # bloom, pythia

models = ['pythia1-4b', 'tinyllama', 'phi2', 'codeparrot-small', 'pythia-410m'] 
models = [ 'codeparrot-small' ] 
models = ['codegemma-2b']

models = ['codegemma-2b','bloomz-560m', 'stablecode-3b','tiny_starcoder',
          'starcoder-GPTQ','rtlabs-stablecode-3B']

models = ['bloomz-560m', 'stablecode-3b',]
models = ['codegemma-2b', 'tiny_starcoder']#'tiny_starcoder',
models = ['gemma-2b']
models = ['starcoderbase-1b']
models = ['bloomz-1b1']
models = ['stablecode-3b-completion']


model_checkpoint = {'codet5-base':"Salesforce/codet5-base", 'codet5p-220':'Salesforce/codet5p-220m', 
                    'codegen-350M-mono':"Salesforce/codegen-350M-mono", 'gpt-neo-125m':"EleutherAI/gpt-neo-125M",
                    'codeparrot-small':'codeparrot/codeparrot-small', 'pythia-410m':"EleutherAI/pythia-410m",
                    'tinyllama':'TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T',
                    'llama3in':'meta-llama/Meta-Llama-3-8B-Instruct', # 
                    'mistral7b':'mistralai/Mistral-7B-Instruct-v0.2',
                    'gemma2b':'google/gemma-2b',
                    'gpt2large':'openai-community/gpt2-large',
                    'pythia1-4b':'EleutherAI/pythia-1.4b',
                    'mpt-1-4b':'mosaicml/mpt-1b-redpajama-200b',
                    'tiger-7b':'TigerResearch/tigerbot-7b-base',
                    'llama2-7b':'meta-llama/Llama-2-7b-hf',
                    'mobillama':'MBZUAI/MobiLlama-1B',
                    'phi2':'microsoft/phi-2',
                    'olmo':'allenai/OLMo-1B-hf',
                    
                    'gemma-2-2b':'google/gemma-2-2b',
                    'starcoder2-3b':'bigcode/starcoder2-3b', #no
                    'starcoder-GPTQ':'TheBloke/starcoder-GPTQ',
                    'rtlabs-stablecode-3B':'rtlabs/StableCode-3B',
                    'PowerLM-3b':'ibm/PowerLM-3b',
                    'refact':'smallcloudai/Refact-1_6B-fim',
                    
                    'bloomz-560m':'bigscience/bloomz-560m', #yes
                    'stablecode-3b':'stabilityai/stablecode-instruct-alpha-3b',
                    'tiny_starcoder':'bigcode/tiny_starcoder_py',
                    'codegemma-2b':'google/codegemma-2b', #yes
                    
                    'gemma-2b':'google/gemma-2b',
                    'starcoderbase-1b':'bigcode/starcoderbase-1b',
                    'bloomz-1b1':'bigscience/bloomz-1b1',
                    'stablecode-3b-completion':'stabilityai/stablecode-completion-alpha-3b-4k',
                    
                    } # model:checkpoint

device = 'cuda'  #[CHANGE]
if device =='cuda':
    torch.cuda.empty_cache()

for model in models:
    save_directory = f"models/torchscript/{model}.pt" #[CHANGE]
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
        elif model_name in ['codegemma-2b', 'gemma-2b','stablecode-3b-completion']:
            
            tokenizer = AutoTokenizer.from_pretrained(checkpoint,torchscript=True,trust_remote_code=True, padding_side='left')
            model = AutoModelForCausalLM.from_pretrained(checkpoint,torchscript=True,trust_remote_code=True)
        elif model_name in ['stablecode-3b-completion']:
            
            model = AutoModelForCausalLM.from_pretrained(checkpoint, torchscript =True, attn_implementation="eager",trust_remote_code=True)
            tokenizer = AutoTokenizer.from_pretrained(checkpoint, torchscript=True,trust_remote_code=True)
            
            #if hasattr(model, "tie_weights"):
            #    model.tie_weights()  # Ensure the final layer weights are tied properly

        elif model_name in ['tiny_starcoder','starcoderbase-1b', ]:
            model = AutoModelForCausalLM.from_pretrained(checkpoint, torchscript =True, attn_implementation="eager",trust_remote_code=True,force_download=True)
            tokenizer = AutoTokenizer.from_pretrained(checkpoint, torchscript=True,trust_remote_code=True,)   
            
        else:
            model = AutoModelForCausalLM.from_pretrained(checkpoint, torchscript =True, attn_implementation="eager",trust_remote_code=True)
            tokenizer = AutoTokenizer.from_pretrained(checkpoint, torchscript=True,trust_remote_code=True)


        if model_name in ['pythia1-4b','mistral7b','gpt2large',
                          'mpt-1-4b', 'llama2-7b', 'mobillama','phi2','olmo',
                        'bloomz-560m',
                        'rtlabs-stablecode-3B','bloomz-1b1','stablecode-3b-completion'
                        ]: # 
            print("Adding token")
            tokenizer.add_special_tokens({'pad_token': '[PAD]'})

        #if condition:\n            return condition\n        else:\n            return None\n\n    def _get_condition(self
        dummy_input = '''def rolling_min(numbers: List[int]) -> int:
            """
            Return the minimum number of numbers in the rolling window.

            :param numbers: The numbers to return.
            :return: The minimum number of numbers in the rolling window.
            """
            return min(numbers)'''
        #dummy_input = "def rolling_max(numbers: List[int]) ->"
        
        #dummy_input = "if condition:\n            return condition\n        else:\n            return None\n\n    def _get_condition(self"
        #inputs = tokenizer.encode_plus(dummy_input,max_length = int(20),pad_to_max_length = True, add_special_tokens = True, return_tensors = 'pt')
        #inputs = tokenizer.encode_plus(dummy_input,max_length = int(20),padding = True, add_special_tokens = True, return_tensors = 'pt',truncation=True)
        #inputs = tokenizer.encode_plus(dummy_input, return_tensors = 'pt', max_length = int(20),padding = True)
        if model_name in ['codeparrot-small', 'pythia-410m','tinyllama', 'tiger-7b', 'llama2-7b',
                          'mobillama','phi2','olmo',
                           'bloomz-560m', 'starcoder2-3b',
                            'tiny_starcoder', 'starcoderbase-1b','bloomz-1b1',
                           ]:
            #tokenizer.add_special_tokens({'pad_token': '[PAD]'})
            tokenizer.pad_token = tokenizer.eos_token
            print("Pad token ID:", model.config.pad_token_id)
            inputs = tokenizer.encode_plus(dummy_input,max_length = MAX_LENGTH, padding = 'max_length', return_tensors = 'pt',truncation='only_second')
        else:
            inputs = tokenizer(dummy_input, return_tensors="pt",max_length = MAX_LENGTH, padding = 'max_length').to(device)


            #inputs = tokenizer.encode_plus(dummy_input,max_length = MAX_LENGTH, padding = 'max_length', return_tensors = 'pt')
        #inputs = tokenizer(dummy_input, return_tensors="pt", truncation=True, max_length=int(30))
        print("inputs:",inputs)
        
        # Ensure the model's embedding layer is resized to accommodate the new token:
        # add this if you have problems like: "Error saving model: index out of range in self"
        
        model.resize_token_embeddings(len(tokenizer))
        
        # if model_name == 'codegemma-2b':
        #     prompt_len = inputs["input_ids"].shape[-1]
        #     outputs = model.generate(**inputs, max_new_tokens=100)
        #     print(tokenizer.decode(outputs[0][prompt_len:]))

        
        input_ids = inputs["input_ids"]
        print("1 ------")
        
        attention_mask = inputs["attention_mask"]
        # none_int_tensor = torch.full((128,), -1, dtype=torch.int)
        # none_int_tensor = input_ids[:, -1:]
        # none_int_tensor = torch.ones((1, 128), dtype=torch.int64)
        # sos_token_id = tokenizer.bos_token_id  # or `tokenizer.cls_token_id` if `bos_token_id` is not set
        # print("sos_token_id: ", sos_token_id)
        # decoder_input_ids = torch.tensor([[sos_token_id]])  # Shape: [1, 1]
        # print("decoder_input_ids: ",decoder_input_ids)

        print("2 ------")
        
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        model.to(device)

        print("3 ------")
        input_tuple = (input_ids,attention_mask, input_ids) # decoder_input_ids["input_ids"]
        
        print("Input IDs shape:", inputs['input_ids'].shape)
        print("Attention mask shape:", inputs['attention_mask'].shape)
        
        # If your model has multiple inputs they must passed in order that is defined by your mode
        # script mode by usign torch.jit.trace
        if model_name in [ 'codet5-base','codet5p-220']:
            traced_model = torch.jit.trace(model, input_tuple) # t5 models
            traced_model = torch.jit.script(model) # not supported for some functions definitions
        else:
            print("4 ------")
            model(input_ids) # added nov
            traced_model = torch.jit.trace(model, [input_ids])
            #traced_model = torch.jit.script(model) # not supported for some functions definitions

        print("5 ------")


        #traced model is a TorchScript module
        torch.jit.save(traced_model,save_directory)
        print(f"\nModel '{model}' successfully saved in '{save_directory}'")

    except Exception as e:
        print(f"Error saving model: {e}")
        print(f"----------------------------------------------------------")
    
    print(f"----------------------------------------------------------")
    