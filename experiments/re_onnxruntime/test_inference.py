"""
Script to test the exported models

Modify before testing:
- runtime engine
- model

if problems with CUDAEP:
pip uninstall onnxruntime_gpu
just install onnxruntime-gpu


"""

from transformers import AutoModelForCausalLM, AutoTokenizer

from optimum.onnxruntime import ORTModelForCausalLM 
from optimum.intel import OVModelForCausalLM

from transformers import AutoTokenizer, pipeline

MAX_LENGTH=128
runtime_engine = 'onnx'

exec_provider='CUDAExecutionProvider' # CPUExecutionProvider, CUDAExecutionProvider ## [CHANGE]

model_name = 'pythia-410m'

print(f"Testing for {runtime_engine}")

model_dir = None
if runtime_engine == 'onnx':
    model_dir = f'models/onnx/{model_name}'
    #model_dir = 'models/onnx/onnx_codegen_3'
elif runtime_engine == 'ov' and exec_provider=='CPUExecutionProvider':
    model_dir = f'models/ov/{model_name}'


def predict( user_input: str):
        

    # model = AutoModelForCausalLM.from_pretrained(onnx_dir, device_map = 'auto', torch_dtype = 'auto')
    
    tokenizer = AutoTokenizer.from_pretrained(f"models/onnx/{model_name}")

    model = None
    if runtime_engine == 'onnx':
        #model = ORTModelForCausalLM.from_pretrained(model_dir, use_cache=False)
        
        model = ORTModelForCausalLM.from_pretrained(model_dir, use_cache=True, provider=exec_provider)
    elif runtime_engine == 'ov' and exec_provider == 'CPUExecutionProvider':
        #model = OVModelForCausalLM.from_pretrained(model_dir, device_map = 'auto', torch_dtype = 'auto', use_cache=True, provider=exec_provider)
        model = OVModelForCausalLM.from_pretrained(model_dir, use_cache=True, provider=exec_provider)
        
    #text = "def get_random_element(dictionary):"

    #completion = model.generate(**tokenizer(text, return_tensors="pt"))
    #completion = model.generate(**tokenizer(text, return_tensors="pt"),max_new_tokens =25)
    response = {
        "prediction" : infer(user_input,model,tokenizer),
    }
    
    return response


def infer( text: str, model, tokenizer) -> str:
    # tokenize
    inputs = tokenizer(text, return_tensors="pt")
    # generate
    tokens = model.generate(**inputs, max_length=MAX_LENGTH,no_repeat_ngram_size=2,)
    print(tokens.shape)
    print("tokens",tokens)
    print(tokens[0])
    print(tokens[0].shape)
    
    # decode
    prediction = tokenizer.decode(tokens[0])
    return prediction


print(predict("def hello_world"))