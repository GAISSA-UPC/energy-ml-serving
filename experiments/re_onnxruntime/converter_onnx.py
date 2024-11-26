""" 
Script to save models with ONNX formats

Commands that can be used instead
#optimum-cli export onnx --task text2text-generation --model "Salesforce/codet5-base" models/onnx_2/codet5-base
#optimum-cli export onnx --task text2text-generation-with-past --model 'Salesforce/codet5p-220m' models/onnx_2/codet5p-220
#optimum-cli export onnx --task text-generation-with-past --model "Salesforce/codegen-350M-mono" models/onnx_2/codegen-350M-mono, killeds
#optimum-cli export onnx --task text-generation-with-past --model "EleutherAI/gpt-neo-125M" models/onnx_2/gpt-neo-125M
#optimum-cli export onnx --task text-generation-with-past --model 'codeparrot/codeparrot-small' models/onnx_2/codeparrot-small
#optimum-cli export onnx --task text-generation-with-past --model "EleutherAI/gpt-neo-125M" models/onnx_2/gpt-neo-125M
#optimum-cli export onnx --task text-generation-with-past --model "EleutherAI/pythia-410m" models/onnx_2/pythia-410m # killed

python experiments/re_onnxruntime/converter_onnx.py;python experiments/re_onnxruntime/converter_openvino.py;

Temp files saved in, remove it:
C:\\Users\\[user]\\AppData\\Local\\Temp\\tmpoiu9iwxj\\*
"""
import os
import torch
from optimum.onnxruntime import ORTModelForMaskedLM, ORTModelForQuestionAnswering, ORTModelForCausalLM, ORTModelForSeq2SeqLM
from transformers import AutoTokenizer, AutoConfig

import logging
#logging.basicConfig(level=logging.DEBUG)

#from optimum.intel import OVModelForCausalLM
MAX_LENGTH = 128

re = 'onnx'

# ORTModelForSeq2SeqLM, ORTModelForSeq2SeqLM, ORTModelForCausalLM, ORTModelForSeq2SeqLM, ORTModelForSeq2SeqLM, ORTModelForSeq2SeqLM

#models = [ 'codet5-base', 'codet5p-220',  'gpt-neo-125m', 'codeparrot-small', 'pythia-410m'] # bloom, pythia
models = [ 'codet5-base', 'codet5p-220',  'codeparrot-small', 'pythia-410m'] # bloom, pythia
#models = [ 'tinyllama', ] 
models = ['codegemma-2b']

#models = ['codegemma-2b', 'bloomz-560m', 'starcoder2-3b']
#models = ['codegemma-2b','bloomz-560m', 'stablecode-3b','starcoder2-3b','PowerLM-3b','refact']

models = ['codegemma-2b','bloomz-560m', 'stablecode-3b','tiny_starcoder',
          'starcoder-GPTQ','rtlabs-stablecode-3B']
#TheBloke/starcoder-GPTQ Error saving model starcoder-GPTQ: No package metadata was found for auto-gptq
models = ['rtlabs-stablecode-3B','stablecode-3b']

models = ['codegemma-2b', 'tiny_starcoder',]
models = ['gemma-2b']
#models = ['starcoderbase-1b']
models = ['bloomz-1b1']
models = ['stablecode-3b-completion']


model_checkpoint = {'codet5-base':"Salesforce/codet5-base", 'codet5p-220':'Salesforce/codet5p-220m', 
                    'codegen-350M-mono':"Salesforce/codegen-350M-mono", 'gpt-neo-125m':"EleutherAI/gpt-neo-125M",
                    'codeparrot-small':'codeparrot/codeparrot-small', 'pythia-410m':"EleutherAI/pythia-410m",
                    'tinyllama':'TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T',
                    'phi3': 'microsoft/Phi-3-mini-128k-instruct',
                    'llama3in':'meta-llama/Meta-Llama-3-8B-Instruct', # killed gaissa server
                    'mistral7b':'mistralai/Mistral-7B-Instruct-v0.2',
                    'pythia1-4b':'EleutherAI/pythia-1.4b',
                    'phi2':'microsoft/phi-2',
                    'olmo':'allenai/OLMo-1B-hf',
                    'gemma-2-2b':'google/gemma-2-2b',
                    
                    'starcoder2-3b':'bigcode/starcoder2-3b', #no
                    'starcoder-GPTQ':'TheBloke/starcoder-GPTQ',
                    'rtlabs-stablecode-3B':'rtlabs/StableCode-3B',
                    'PowerLM-3b':'ibm/PowerLM-3b',
                    'refact':'smallcloudai/Refact-1_6B-fim',
                    
                    'codegemma-2b':'google/codegemma-2b', #yes
                    'bloomz-560m':'bigscience/bloomz-560m', #yes
                    'stablecode-3b':'stabilityai/stablecode-instruct-alpha-3b',
                    'tiny_starcoder':'bigcode/tiny_starcoder_py',
                    'gemma-2b':'google/recurrentgemma-2b',
                    'starcoderbase-1b':'bigcode/starcoderbase-1b',
                    'bloomz-1b1':'bigscience/bloomz-1b1',
                    'stablecode-3b-completion':'stabilityai/stablecode-completion-alpha-3b-4k',
                    } # model:checkpoint

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
        #device = 'cpu'
        #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        #print(f'device: {device}')
        
        # change the class
        # use_cache = True for text-generation, check transformers library if modeling_{model} has this functionality
        #ort_model = ORTModelForMaskedLM.from_pretrained(model_checkpoint, export=True, )

        if model in [ 'codet5-base','codet5p-220']:
            ort_model = ORTModelForSeq2SeqLM.from_pretrained(checkpoint, export=True, use_cache = True) # t5, 
        elif model in ['starcoder2-3b--']:
            config = AutoConfig.from_pretrained(checkpoint)
            ort_model = ORTModelForCausalLM.from_pretrained(checkpoint, config = config, export=True, use_cache = True,
                                                            force_download=True)
        elif model in ['stablecode-3b',]:
            config = AutoConfig.from_pretrained(checkpoint)
            ort_model = ORTModelForCausalLM.from_pretrained(checkpoint, config = config, export=True, use_cache = False,
                                                            #force_download=True,
                                                            trust_remote_code=True,)
        elif model in ['stablecode-3b-completion',]:
            config = AutoConfig.from_pretrained(checkpoint)
            #Please either pass use_cache=True, use_io_binding=True (default), or use_cache=False, use_io_binding=False.
            ort_model = ORTModelForCausalLM.from_pretrained(checkpoint, config = config, export=True, use_cache=True, use_io_binding=True,
                                                            #force_download=True,
                                                            trust_remote_code=True,)
        elif model in ['gemma--']:
            ort_model = ORTModelForCausalLM.from_pretrained(checkpoint, export=True, use_cache = False)
        else:
            print("----- Loading ------")
            config = AutoConfig.from_pretrained(checkpoint)
            ort_model = ORTModelForCausalLM.from_pretrained(checkpoint, config = config, export=True, use_cache = True) # codegen, gpt-neo, pythia (gptneox), codeparrot


        # [codegen]
        tokenizer = AutoTokenizer.from_pretrained(checkpoint)


        # tokenizer = AutoTokenizer.from_pretrained(
        #     "EleutherAI/pythia-410m",
        #     revision="step3000",
        #     cache_dir="../pythia-410m/step3000",
        #     )

        text = "def hello_world():"

        print("----- tokenizing ------")
        
        # tokenize
        inputs = tokenizer(text, return_tensors="pt")

        #models = ['rtlabs-stablecode-3B','stablecode-3b']

        #print(f"ort_model.main_input_name: {ort_model.main_input_name}")
        #inputs = {key: value for key, value in inputs.items() if key in ort_model.main_input_name}
        # Check if the model has 'main_input_name' (specific to ONNX Runtime)
        if hasattr(model, "main_input_name"):
            inputs = {key: value for key, value in inputs.items() if key in model.main_input_name}
        else:
            # For PyTorch models, no filtering is needed
            inputs = {key: value for key, value in inputs.items()}
            
        
        print(f"inputs: {inputs}")
        if model in ['stablecode-3b-completion',]:
            if 'token_type_ids' in inputs: # GPT models do not use this token_type_ids
                del inputs['token_type_ids']
            
        #ort_model.to(device)
        
        print("----- generating tokens from example ------")
        # generate
        tokens = ort_model.generate(**inputs, max_length=MAX_LENGTH,no_repeat_ngram_size=2,)
        print(tokens)
        # Save the onnx model and tokenizer
        ort_model.save_pretrained(save_directory)

        print(f"\nModel '{model}' successfully saved in '{save_directory}'")
        print(f"----------------------------------------------------------")

    except Exception as e:
        print(f"Error saving model {model}: {e}")
        print(f"----------------------------------------------------------")

    #tokenizer.save_pretrained(save_directory)
    