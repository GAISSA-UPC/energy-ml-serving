from transformers import AutoModelForCausalLM, AutoTokenizer

from optimum.onnxruntime import ORTModelForCausalLM 
#from optimum.intel import OVModelForCausalLM

from transformers import AutoTokenizer, pipeline

runtime_engine = 'onnx'

print(f"Testing for {runtime_engine}")
model_dir = None
if runtime_engine == 'onnx':
    model_dir = 'models/onnx/gpt-neo-125m'
    #model_dir = 'models/onnx/onnx_codegen_3'
elif runtime_engine == 'ov':
    model_dir = 'models/ov/ov_codegen'


def predict( user_input: str):
        

    # model = AutoModelForCausalLM.from_pretrained(onnx_dir, device_map = 'auto', torch_dtype = 'auto')
    
    tokenizer = AutoTokenizer.from_pretrained(model_dir)

    model = None
    if runtime_engine == 'onnx':
        #model = ORTModelForCausalLM.from_pretrained(model_dir, use_cache=False)
        
        model = ORTModelForCausalLM.from_pretrained(model_dir, device_map = 'auto', torch_dtype = 'auto', use_cache=True)
    elif runtime_engine == 'ov':
        model = OVModelForCausalLM.from_pretrained(model_dir, device_map = 'auto', torch_dtype = 'auto', use_cache=True)
        
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
    tokens = model.generate(**inputs)
    print(tokens.shape)
    print("tokens",tokens)
    print(tokens[0])
    print(tokens[0].shape)
    
    # decode
    prediction = tokenizer.decode(tokens[0])
    return prediction


print(predict("def hello_worl"))