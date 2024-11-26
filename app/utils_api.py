"""
Utils for API
"""
import os
import torch

# [t5, codet5p_220m]   torch
from transformers import  AutoModelForSeq2SeqLM # T5ForConditionalGeneration,
# [pythia-70m] torch    
#from transformers import GPTNeoXForCausalLM
# [codeparrot-small] torch
from transformers import AutoModelForCausalLM
# [codegen]
#from transformers import AutoModelForCausalLM,AutoModelForSeq2SeqLM

# gptneo
#from transformers import pipeline

from transformers import AutoTokenizer #torchscript, OV, ONNX, torch
from optimum.onnxruntime import  ORTModelForSeq2SeqLM, ORTModelForCausalLM #ONNX
from optimum.intel import OVModelForSeq2SeqLM, OVModelForCausalLM  # OV


if torch.cuda.is_available():
    device = "cuda:0" 
    GPU_ID = os.getenv("GPU_DEVICE_ORDINAL", 0)
else:
    device = "cpu"

device='cpu' #cpu cuda ## [CHANGE]

exec_provider='CPUExecutionProvider'
if device == 'cuda':
    exec_provider='CUDAExecutionProvider'
else:
    exec_provider='CPUExecutionProvider'

# VERIFY TORCHSCRIPT MODELS ARE EXPORTED WITH CUDA IF USING IT # [CHANGE]

#new_models = ['bloomz-560m', 'stablecode-3b',]
# After you have exported all models you can quickly test if they are working with generic slm created class:
new_models = ['starcoderbase-1b','bloomz-1b1','stablecode-3b-completion']
test_model = new_models[0]


info_models = {
    "torch" : {
        "codet5-base":{"m_class":AutoModelForSeq2SeqLM,"model_dir":"models/torch/codet5-base","tokenizer_dir":'models/onnx/codet5-base'},
        "codet5p-220":{"m_class":AutoModelForSeq2SeqLM,"model_dir":"models/torch/codet5p-220","tokenizer_dir":'models/onnx/codet5p-220'},
        "codeparrot-small":{"m_class":AutoModelForCausalLM,"model_dir":"models/torch/codeparrot-small","tokenizer_dir":'models/onnx/codeparrot-small'},
        "pythia-410m":{"m_class":AutoModelForCausalLM,"model_dir":"models/torch/pythia-410m","tokenizer_dir":'models/onnx/pythia-410m'},
        "tinyllama":{"m_class":AutoModelForCausalLM,"model_dir":"models/torch/tinyllama","tokenizer_dir":'models/onnx/tinyllama'},
        "phi2":{"m_class":AutoModelForCausalLM,"model_dir":"models/torch/phi2","tokenizer_dir":'models/onnx/phi2'},
        "pythia1-4b":{"m_class":AutoModelForCausalLM,"model_dir":f"models/torch/pythia1-4b","tokenizer_dir":f'models/onnx/pythia1-4b'},
        "bloomz-560m":{"m_class":AutoModelForCausalLM,"model_dir":"models/torch/bloomz-560m","tokenizer_dir":'models/onnx/bloomz-560m'},
        "stablecode-3b":{"m_class":AutoModelForCausalLM,"model_dir":"models/torch/stablecode-3b","tokenizer_dir":'models/onnx/stablecode-3b'},
        "codegemma-2b":{"m_class":AutoModelForCausalLM,"model_dir":"models/torch/codegemma-2b","tokenizer_dir":'models/onnx/codegemma-2b'},
        "tiny_starcoder":{"m_class":AutoModelForCausalLM,"model_dir":"models/torch/tiny_starcoder","tokenizer_dir":'models/onnx/tiny_starcoder'},
        "starcoderbase-1b":{"m_class":AutoModelForCausalLM,"model_dir":"models/torch/starcoderbase-1b","tokenizer_dir":'models/onnx/starcoderbase-1b'},
        "bloomz-1b1":{"m_class":AutoModelForCausalLM,"model_dir":"models/torch/bloomz-1b1","tokenizer_dir":'models/onnx/bloomz-1b1'},
        "stablecode-3b-completion":{"m_class":AutoModelForCausalLM,"model_dir":"models/torch/stablecode-3b-completion","tokenizer_dir":'models/onnx/stablecode-3b-completion'},
        
        "slm":{"m_class":AutoModelForCausalLM,"model_dir":f"models/torch/{test_model}","tokenizer_dir":f'models/onnx/{test_model}'},
    },
    "onnx" : {
        "codet5-base":{"m_class":ORTModelForSeq2SeqLM,"model_dir":"models/onnx/codet5-base","tokenizer_dir":'models/onnx/codet5-base'},
        "codet5p-220":{"m_class":ORTModelForSeq2SeqLM,"model_dir":'models/onnx/codet5p-220',"tokenizer_dir":'models/onnx/codet5p-220'},
        "codeparrot-small":{"m_class":ORTModelForCausalLM,"model_dir":'models/onnx/codeparrot-small',"tokenizer_dir":'models/onnx/codeparrot-small'},
        "pythia-410m":{"m_class":ORTModelForCausalLM,"model_dir":'models/onnx/pythia-410m',"tokenizer_dir":'models/onnx/pythia-410m'},
        "tinyllama":{"m_class":ORTModelForCausalLM,"model_dir":'models/onnx/tinyllama',"tokenizer_dir":'models/onnx/tinyllama'},
        "phi2":{"m_class":ORTModelForCausalLM,"model_dir":'models/onnx/phi2',"tokenizer_dir":'models/onnx/phi2'},
        "pythia1-4b":{"m_class":ORTModelForCausalLM,"model_dir":f'models/onnx/pythia1-4b',"tokenizer_dir":f'models/onnx/pythia1-4b'},
        "bloomz-560m":{"m_class":ORTModelForCausalLM,"model_dir":"models/onnx/bloomz-560m","tokenizer_dir":'models/onnx/bloomz-560m'},
        "stablecode-3b":{"m_class":ORTModelForCausalLM,"model_dir":"models/onnx/stablecode-3b","tokenizer_dir":'models/onnx/stablecode-3b'},
        "codegemma-2b":{"m_class":ORTModelForCausalLM,"model_dir":"models/onnx/codegemma-2b","tokenizer_dir":'models/onnx/codegemma-2b'},
        "tiny_starcoder":{"m_class":ORTModelForCausalLM,"model_dir":"models/onnx/tiny_starcoder","tokenizer_dir":'models/onnx/tiny_starcoder'},
        "starcoderbase-1b":{"m_class":ORTModelForCausalLM,"model_dir":"models/onnx/starcoderbase-1b","tokenizer_dir":'models/onnx/starcoderbase-1b'},
        "bloomz-1b1":{"m_class":ORTModelForCausalLM,"model_dir":"models/onnx/bloomz-1b1","tokenizer_dir":'models/onnx/bloomz-1b1'},
        "stablecode-3b-completion":{"m_class":ORTModelForCausalLM,"model_dir":"models/onnx/stablecode-3b-completion","tokenizer_dir":'models/onnx/stablecode-3b-completion'},
        
        
        "slm":{"m_class":ORTModelForCausalLM,"model_dir":f'models/onnx/{test_model}',"tokenizer_dir":f'models/onnx/{test_model}'},

    },
    "ov" : {
        "codet5-base":{"m_class":OVModelForSeq2SeqLM,"model_dir":"models/ov/codet5-base"},
        "codet5p-220":{"m_class":OVModelForSeq2SeqLM,"model_dir":'models/ov/codet5p-220'},
        "codeparrot-small":{"m_class":OVModelForCausalLM,"model_dir":'models/ov/codeparrot-small'},
        "pythia-410m":{"m_class":OVModelForCausalLM,"model_dir":'models/ov/pythia-410m'},
        "tinyllama":{"m_class":OVModelForCausalLM,"model_dir":'models/ov/tinyllama'},
        "phi2":{"m_class":OVModelForCausalLM,"model_dir":'models/ov/phi2'},
        "pythia1-4b":{"m_class":OVModelForCausalLM,"model_dir":f'models/ov/pythia1-4b'},
        
        "bloomz-560m":{"m_class":OVModelForCausalLM,"model_dir":f'models/ov/bloomz-560m'},
        "stablecode-3b":{"m_class":OVModelForCausalLM,"model_dir":f'models/ov/stablecode-3b'},
        "codegemma-2b":{"m_class":OVModelForCausalLM,"model_dir":f'models/ov/codegemma-2b'},
        "tiny_starcoder":{"m_class":OVModelForCausalLM,"model_dir":f'models/ov/tiny_starcoder'},
        "starcoderbase-1b":{"m_class":OVModelForCausalLM,"model_dir":f'models/ov/starcoderbase-1b'},
        "bloomz-1b1":{"m_class":OVModelForCausalLM,"model_dir":f'models/ov/bloomz-1b1'},
        "stablecode-3b-completion":{"m_class":OVModelForCausalLM,"model_dir":f'models/ov/stablecode-3b-completion'},
        
        
        "slm":{"m_class":OVModelForCausalLM,"model_dir":f'models/ov/{test_model}'},

    },
    "torchscript" : {
        "codet5-base":{"m_class":"","model_dir":'models/torchscript/codet5-base.pt'},
        "codet5p-220":{"m_class":"","model_dir":'models/torchscript/codet5p-220.pt'},
        "codeparrot-small":{"m_class":"","model_dir":'models/torchscript/codeparrot-small.pt'},
        "pythia-410m":{"m_class":"","model_dir":'models/torchscript/pythia-410m.pt'},
        "tinyllama":{"m_class":"","model_dir":'models/torchscript/tinyllama.pt'},
        "phi2":{"m_class":"","model_dir":'models/torchscript/phi2.pt'},
        "pythia1-4b":{"m_class":"","model_dir":f'models/torchscript/pythia1-4b.pt'},
        "bloomz-560m":{"m_class":"","model_dir":f'models/torchscript/bloomz-560m.pt'},
        "stablecode-3b":{"m_class":"","model_dir":f'models/torchscript/stablecode-3b.pt'},
        "codegemma-2b":{"m_class":"","model_dir":f'models/torchscript/codegemma-2b.pt'},
        "tiny_starcoder":{"m_class":"","model_dir":f'models/torchscript/tiny_starcoder.pt'},
        "starcoderbase-1b":{"m_class":"","model_dir":f'models/torchscript/starcoderbase-1b.pt'},
        "bloomz-1b1":{"m_class":"","model_dir":f'models/torchscript/bloomz-1b1.pt'},
        "stablecode-3b-completion":{"m_class":"","model_dir":f'models/torchscript/stablecode-3b-completion.pt'},
            
        "slm":{"m_class":"","model_dir":f'models/torchscript/{test_model}.pt'},
    },
}