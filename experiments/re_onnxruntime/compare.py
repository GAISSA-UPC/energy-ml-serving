"""
Script to compare times, using ONNX runtime engine and torch, with CPUEP and CUDAEP

"""

from optimum.onnxruntime import ORTModelForCausalLM
from transformers import AutoTokenizer, AutoModelForCausalLM
import time

device = 'cuda' #change
engine = 'onnx' #change

# Load pre-trained model and tokenizer
model_name = "models/onnx/tinyllama/"
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.add_special_tokens({'pad_token': '[PAD]'})
#model = AutoModelForCausalLM.from_pretrained(model_name)

# Initialize the Optimum model for ONNX
#onnx_model = ORTModelForCausalLM.from_pretrained(model_name, from_transformers=True)

# Export to ONNX
if engine == 'onnx':
    onnx_model_path = "models/onnx/tinyllama/"
else:
    onnx_model_path = "models/torch/tinyllama/"

prompt_list = [

    "Check if in given list of numbers, are",
    "['()', '(())', '(()",
    ">>> truncate_number(3.5)",
    "def below_zero(operations: List[int])",
    "Mean Absolute Deviation is the average absolute difference between each",
    "Insert a number 'delimeter' between every",
    "[2, 3, 1, 3",
    " Filter an input list of strings only for ones that contain",
    "Empty sum should be equal to 0 and empty product should be equal",
    "def rolling_max(numbers: List[int]) ->",
] *10

#prompt_list = ["What is the capital of France?"]

# Prepare a batch of prompts
prompts = ["What is the future of AI?"] * 16   # Example prompt replicated to create a batch
#prompts = ['What is the future of AI?', 'What is the future of oI?']
print(prompts)
print(type(prompts))

if device == 'cuda':
    if engine == 'onnx':
        model = ORTModelForCausalLM.from_pretrained(onnx_model_path,provider='CUDAExecutionProvider')#provider=exec_provider
    else:
        model = AutoModelForCausalLM.from_pretrained(onnx_model_path,)#provider=exec_provider

else:
    if engine == 'onnx':
        model = ORTModelForCausalLM.from_pretrained(onnx_model_path,provider='CPUExecutionProvider')#provider=exec_provider
    else:
        model = AutoModelForCausalLM.from_pretrained(onnx_model_path,)

start_time_load = time.time()
model.to(device)
model_to = time.time() - start_time_load

print(f"Model to: --- {model_to} seconds ---" )

start_time = time.time()
for p in prompt_list:
    #prompts = ["What is the future of AI?","What is the capital of France?"]   # Example prompt replicated to create a batch
    prompts = [p] *64
    #prompts = [p]  #

    
    inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=False, max_length=128) #padding=True,

    # Load the ONNX model using Optimum

    if device == 'cuda':
        print("---- Using CUDA ----")
        inputs = inputs.to(device)
        #model.to(device) # move only when loaded
        

    # Run inference
    outputs = model.generate(**inputs, max_length=100)
    decoded_outputs = [tokenizer.decode(output, skip_special_tokens=True) for output in outputs]


    for line in decoded_outputs:
        print(line)
    print(len(decoded_outputs))

execution_time = time.time() - start_time

print(f"--- {execution_time} seconds ---" )
