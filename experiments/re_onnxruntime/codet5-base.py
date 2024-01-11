from transformers import T5ForConditionalGeneration, AutoTokenizer
from optimum.onnxruntime import ORTModelForSeq2SeqLM
#from optimum.intel import OVModelForSeq2SeqLM

from transformers import AutoTokenizer, pipeline


runtime_engine = 'onnx'

print(f"Testing for {runtime_engine}")
model_dir = None
if runtime_engine == 'onnx':
    model_dir = 'models/onnx/codet5-base'
elif runtime_engine == 'ov':
    model_dir = 'models/ov/ov_codet5p'

def predict( user_input: str):
        
    
    #device = "cpu" # for GPU usage or "cpu" for CPU usage

    #tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    # torch_dtype = 'auto' not implemented
    #model = T5ForConditionalGeneration.from_pretrained(checkpoint, device_map = 'auto')

    tokenizer = AutoTokenizer.from_pretrained(model_dir) # ./bert-base-uncased
    
    model = None
    if runtime_engine == 'onnx':
        model = ORTModelForSeq2SeqLM.from_pretrained(model_dir,    return_dict = True, use_cache = True)
    elif runtime_engine == 'ov':
        model = OVModelForSeq2SeqLM.from_pretrained(model_dir,    return_dict = True, use_cache = True)
            
    response = {
        "prediction" : infer(user_input,model,tokenizer),
    }
    return response

def infer( text: str, model, tokenizer) -> str:
    inputs = tokenizer.encode(text, return_tensors="pt").to('cpu')
    #outputs = model.generate(inputs, max_length=10,max_new_tokens = 30)
    outputs = model.generate(inputs, max_new_tokens = 30)
    print(outputs)
    prediction = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return prediction

print(predict("def hello_world():<extra_id_0>"))