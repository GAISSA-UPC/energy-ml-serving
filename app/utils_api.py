"""
Utils for API
"""


import torch

# [t5, codet5p_220m]   torch
from transformers import  AutoModelForSeq2SeqLM # T5ForConditionalGeneration,
# [pythia-70m] torch    
from transformers import GPTNeoXForCausalLM
# [codeparrot-small] torch
from transformers import AutoModelForCausalLM
# [codegen]
#from transformers import AutoModelForCausalLM,AutoModelForSeq2SeqLM

# gptneo
#from transformers import pipeline

from transformers import AutoTokenizer #torchscript, OV, ONNX, torch
from optimum.onnxruntime import  ORTModelForSeq2SeqLM, ORTModelForCausalLM #ONNX
from optimum.intel import OVModelForSeq2SeqLM, OVModelForCausalLM  # OV

info_models = {
    "torch" : {
        "codet5-base":{"m_class":AutoModelForSeq2SeqLM,"model_dir":"models/torch/codet5-base","tokenizer_dir":'models/onnx/codet5-base'},
        "codet5p-220":{"m_class":AutoModelForSeq2SeqLM,"model_dir":"models/torch/codet5p-220","tokenizer_dir":'models/onnx/codet5p-220'},
        "codeparrot-small":{"m_class":AutoModelForCausalLM,"model_dir":"models/torch/codeparrot-small","tokenizer_dir":'models/onnx/codeparrot-small'},
        "pythia-410m":{"m_class":AutoModelForCausalLM,"model_dir":"models/torch/pythia-410m","tokenizer_dir":'models/onnx/pythia-410m'},
        "tinyllama":{"m_class":AutoModelForCausalLM,"model_dir":"models/torch/tinyllama","tokenizer_dir":'models/onnx/tinyllama'},
    },
    "onnx" : {
        "codet5-base":{"m_class":ORTModelForSeq2SeqLM,"model_dir":"models/onnx/codet5-base","tokenizer_dir":'models/onnx/codet5-base'},
        "codet5p-220":{"m_class":ORTModelForSeq2SeqLM,"model_dir":'models/onnx/codet5p-220',"tokenizer_dir":'models/onnx/codet5p-220'},
        "codeparrot-small":{"m_class":ORTModelForCausalLM,"model_dir":'models/onnx/codeparrot-small',"tokenizer_dir":'models/onnx/codeparrot-small'},
        "pythia-410m":{"m_class":ORTModelForCausalLM,"model_dir":'models/onnx/pythia-410m',"tokenizer_dir":'models/onnx/pythia-410m'},
        "tinyllama":{"m_class":ORTModelForCausalLM,"model_dir":'models/onnx/tinyllama',"tokenizer_dir":'models/onnx/tinyllama'},
    },
    "ov" : {
        "codet5-base":{"m_class":OVModelForSeq2SeqLM,"model_dir":"models/ov/codet5-base"},
        "codet5p-220":{"m_class":OVModelForSeq2SeqLM,"model_dir":'models/ov/codet5p-220'},
        "codeparrot-small":{"m_class":OVModelForCausalLM,"model_dir":'models/ov/codeparrot-small'},
        "pythia-410m":{"m_class":OVModelForCausalLM,"model_dir":'models/ov/pythia-410m'},
        "tinyllama":{"m_class":OVModelForCausalLM,"model_dir":'models/ov/tinyllama'},
    },
    "torchscript" : {
        "codet5-base":{"m_class":"","model_dir":'models/torchscript/codet5-base.pt'},
        "codet5p-220":{"m_class":"","model_dir":'models/torchscript/codet5p-220.pt'},
        "codeparrot-small":{"m_class":"","model_dir":'models/torchscript/codeparrot-small.pt'},
        "pythia-410m":{"m_class":"","model_dir":'models/torchscript/pythia-410m.pt'},
        "tinyllama":{"m_class":"","model_dir":'models/torchscript/tinyllama.pt'}
    },
}