"""
How to deploy in TF serving
- Load model
- save it in SavedModel Fortmat of tf, 

tensorflow
transformers

nohup tensorflow_model_server \
  --rest_api_port=8501 \
  --model_name=bert \
  --model_base_path=models/tf/bert >server.log 2>&1
  
  pip install tensorflow-serving-api
  


setup for linux https://www.tensorflow.org/tfx/serving/setup
apt-get install tensorflow-model-server
  
apt-get upgrade tensorflow-model-server

https://colab.research.google.com/github/huggingface/blog/blob/main/notebooks/10_tf_serving.ipynb#scrollTo=iZLv4FCQ7Tzq

verify the model is loaded correctly
http://localhost:8501/v1/models/bert

"""

# [bert]
#from transformers import BertTokenizer, BertForMaskedLM
from transformers import  TFAutoModelForCausalLM, TFAutoModelForSeq2SeqLM, TFT5ForConditionalGeneration, TFAutoModel

# [t5, codet5p_220m, codet5-base]
#from transformers import T5Tokenizer, T5ForConditionalGeneration
# [codegen, codeparrot]
#from transformers import AutoModelForCausalLM, AutoTokenizer
# [pythia-70m]
#from transformers import GPTNeoXForCausalLM, AutoTokenizer

#from transformers import T5ForConditionalGeneration, AutoTokenizer

#model_checkpoints = ["bert-base-uncased", 't5-small', "Salesforce/codegen-350M-mono","EleutherAI/pythia-70m", "Salesforce/codet5p-220m"]
#ort_model = ORTModelForSeq2SeqLM.from_pretrained(checkpoint, export=True, use_cache = True) # t5, 
#ort_model = ORTModelForCausalLM.from_pretrained(checkpoint, export=True, use_cache = True) # codegen, gpt-neo, pythia (gptneox), codeparrot

models = [ 'codet5-base', 'codet5p-220', 'codegen-350-mono', 'gpt-neo-125m', 'codeparrot-small', 'pythia-410m'] # bloom, pythia
model_checkpoint = {'codet5-base':"Salesforce/codet5-base", 'codet5p-220':'Salesforce/codet5p-220m', 
                    'codegen-350-mono':"Salesforce/codegen-350M-mono", 'gpt-neo-125m':"EleutherAI/gpt-neo-125M",
                    'codeparrot-small':'codeparrot/codeparrot-small', 'pythia-410m':"EleutherAI/pythia-410m"} # model:checkpoint

model_name = models[-2]
checkpoint = model_checkpoint[model_name]

print(f"Saving {model_name} ...\n")
print(f"Checkpoint {checkpoint} ...\n")

#model = TFAutoModelForSeq2SeqLM.from_pretrained(checkpoint)
#from_pt=True: Specify `from_pt=True` to convert a checkpoint from PyTorch to TensorFlow:
# TFAutoModelForSeq2SeqLM
model = TFT5ForConditionalGeneration.from_pretrained(checkpoint,from_pt=True) #t5
#model = TFAutoModelForCausalLM.from_pretrained(checkpoint,from_pt=True)
#model = TFAutoModel.from_pretrained(checkpoint,from_pt=True)


#model = TFAutoModel.from_pretrained(checkpoint,from_pt=True)
print(model.config)

saving = True
if saving:
  temp_model_dir = f'models/tf/{model_name}'
  model.save_pretrained(temp_model_dir, saved_model=True)