"""
How to use a seq2seq or CausalLM to generate code.
Compares against using pipeline() abstracted function.
"""

import os

# from transformers import BertTokenizer, BertForMaskedLM
# from transformers import T5Tokenizer, T5ForConditionalGeneration
# # [codegen]
# from transformers import , AutoTokenizer
# # [pythia-70m]
# from transformers import GPTNeoXForCausalLM, AutoTokenizer

# Model classes
from transformers import AutoModel, AutoModelForSeq2SeqLM, AutoModelForCausalLM
from transformers import LlamaForCausalLM,LlamaTokenizerFast
# Tokenizer classes
from transformers import AutoTokenizer, AutoConfig

import torch
from transformers import pipeline

SAVE_MODEL = True
CHECK_MODEL = True
MAX_LENGTH = 128

model_checkpoint = {'codet5-base':"Salesforce/codet5-base", 'codet5p-220':'Salesforce/codet5p-220m', 
                    'codegen-350-mono':"Salesforce/codegen-350M-mono", 'gpt-neo-125m':"EleutherAI/gpt-neo-125M",
                    'codeparrot-small':'codeparrot/codeparrot-small', 'pythia-410m':"EleutherAI/pythia-410m",
                    'tinyllama':'TinyLlama/TinyLlama-1.1B-intermediate-step-1195k-token-2.5T'} # model:checkpoint
# TinyLlama-1.1B-intermediate-step-1195k-token-2.5T
# TinyLlama/TinyLlama-1.1B-Chat-v0.1

#models = [ 'codet5-base', 'codet5p-220', 'codegen-350-mono', 'gpt-neo-125m', 'codeparrot-small', 'pythia-410m'] # bloom, pythia
#models = [ 'codet5-base', 'codet5p-220', 'codeparrot-small', 'pythia-410m'] # bloom, pythia
models = ['tinyllama']
#model_name = models[5]

#checkpoint = model_checkpoint[model_name]

#tokenizer = T5Tokenizer.from_pretrained("t5-small")
#model = T5ForConditionalGeneration.from_pretrained("t5-small")

#checkpoint = "Salesforce/codegen-350M-mono"
#model = AutoModelForCausalLM.from_pretrained(checkpoint, torch_dtype = 'auto')

#tokenizer = AutoTokenizer.from_pretrained(checkpoint)

def my_pipeline(model, input_text):
    # Specify the model
    #model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

    # Load the tokenizer and model
    #LlamaForCausalLM,LlamaTokenizerFast
    checkpoint = model_checkpoint[model]

    
    if model in [ 'codet5-base','codet5p-220']:
        torch_model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint)
            #ORTModelForSeq2SeqLM.from_pretrained(checkpoint, export=True, use_cache = True) # t5, 
    elif model in ['tinyllama']:
        #torch_model = AutoModel.from_pretrained(checkpoint)
        config = AutoConfig.from_pretrained(checkpoint)
        torch_model = AutoModelForCausalLM.from_pretrained(checkpoint,config = config)
        

    else:
        config = AutoConfig.from_pretrained(checkpoint)
        torch_model = AutoModelForCausalLM.from_pretrained(checkpoint, config = config)

    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    #torch_model = AutoModelForCausalLM.from_pretrained(checkpoint, config = config)

    # Prepare the device (GPU or CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch_model.to(device)

    # Encode the input text and send to the appropriate device
    input_ids = tokenizer.encode(input_text, return_tensors="pt").to(device)

    # Generate a sequence of tokens following the input
    #output_sequences = model.generate(input_ids,do_sample=False, temperature=0.3, top_p=0.80)

    
    # Generate output
    output_sequences = torch_model.generate(
        input_ids,
        max_length=MAX_LENGTH,  # Adjust based on the expected length of output
        #num_return_sequences=1,  # Number of sequences to generate
        #pad_token_id=tokenizer.eos_token_id,  # Usually necessary for open-ended generation
        no_repeat_ngram_size=2,  # Prevents repeating ngrams
        #early_stopping=True,  # Stops generation when all sequences hit the eos_token
        #temperature=0.7,  # Controls randomness. Adjust if needed
        #top_k=50,  # Limits the number of top-k tokens to consider at each step
        #top_p=0.95,  # Nucleus sampling: considers the smallest set of tokens whose cumulative probability exceeds the threshold p
        #do_sample=True,  # Enables stochastic sampling to generate the output
        #bos_token_id =  1,
        #eos_token_id = 2,
        #pad_token_id = 0,
    )


    # Decode and print the output
    #generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    #output_sequences = model.generate(input_ids, max_new_tokens=250,)# top_k=20

    # Decode the output tokens to a string
    output_text = tokenizer.decode(output_sequences[0], skip_special_tokens=True)

    return output_text

for model in models:
    save_directory = f"models/torch/{model}/"
    if SAVE_MODEL:
        print(f"Saving {model} ...\n")
        if os.path.exists(save_directory):
            print(f"Model {model} is already exported...\n")
            continue
    try:
        # select checkpoint
        checkpoint = model_checkpoint[model]

        # change to your directory
        
        print(f"Checkpoint {checkpoint} ...\n")

        # change the class
        # use_cache = True for text-generation, check transformers library if modeling_{model} has this functionality
        #ort_model = ORTModelForMaskedLM.from_pretrained(model_checkpoint, export=True, )

        #model = AutoModelForCausalLM.from_pretrained(checkpoint, torch_dtype = 'auto')

        if model in [ 'codet5-base','codet5p-220']:
            torch_model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint)
            #ORTModelForSeq2SeqLM.from_pretrained(checkpoint, export=True, use_cache = True) # t5, 
        elif model in ['tinyllama']:
            #torch_model = AutoModel.from_pretrained(checkpoint)
            torch_model = AutoModelForCausalLM.from_pretrained(checkpoint)

        else:
            torch_model = AutoModelForCausalLM.from_pretrained(checkpoint, torch_dtype=torch.bfloat16)
            #ORTModelForCausalLM.from_pretrained(checkpoint, export=True, use_cache = True) # codegen, gpt-neo, pythia (gptneox), codeparrot

        # [codegen]
        tokenizer = AutoTokenizer.from_pretrained(checkpoint)

        if CHECK_MODEL:
            print(f"\nModel '{model}' Checkpoint '{checkpoint}' ")
           
            # Print the classes of the loaded model and tokenizer
            print(f"Model class: {torch_model.__class__.__name__}")
            print(f"Tokenizer class: {tokenizer.__class__.__name__}")

            text = "Write a Python sum function"

            print(f"----------------------------------------------------------")
            print("Default all, simple generate:")
            input_ids = tokenizer(text, return_tensors="pt").input_ids
            # simply generate a single sequence
            generated_ids = torch_model.generate(input_ids, max_length=MAX_LENGTH,no_repeat_ngram_size=2,)#max_new_tokens=200
            print(tokenizer.decode(generated_ids[0], skip_special_tokens=True))

            print(f"----------------------------------------------------------")
            print("Using own function of pipeline:")
            outputs = my_pipeline(model,text)
            print(outputs)
            
            print(f"----------------------------------------------------------")
            print(f"Using abstract pipeline()")
            print(f"model.generation_config")
            # Create the pipeline using the loaded model and tokenizer
            
            pipe = pipeline("text-generation", model=torch_model, tokenizer=tokenizer, ) # Adjust `device` as per your setup
            # Access the model's configuration
            model_config = pipe.model.generation_config
            print(model_config)
            # Print the generation-related configuration parameters
            
            print("-----------------------")
            # Generate text
            outputs = pipe(text,max_length = MAX_LENGTH,)
            print(outputs)
        if SAVE_MODEL:
            torch_model.save_pretrained(save_directory, )
            print(f"\nModel '{model}' successfully saved in '{save_directory}'")
            print(f"----------------------------------------------------------")

    except Exception as e:
        print(f"Error : {e}")
        print(f"----------------------------------------------------------")

