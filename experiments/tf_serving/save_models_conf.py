"""
How to save HF models into savedmodel tf format, with the required configuration to serve it

"""

from transformers import  TFAutoModelForCausalLM, TFAutoModelForSeq2SeqLM, TFT5ForConditionalGeneration, TFAutoModel
import tensorflow as tf
from transformers import AutoTokenizer

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

model_name = models[4]
checkpoint = model_checkpoint[model_name]

print(f"Saving {model_name} ...\n")
print(f"Checkpoint {checkpoint} ...\n")



# Choose the right class to load the model, check model card if not sure about the class

#model = TFAutoModelForSeq2SeqLM.from_pretrained(checkpoint)
#from_pt=True: Specify `from_pt=True` to convert a checkpoint from PyTorch to TensorFlow:
# TFAutoModelForSeq2SeqLM
#model = TFT5ForConditionalGeneration.from_pretrained(checkpoint,from_pt=True,use_cache = False) #t5
model = TFAutoModelForCausalLM.from_pretrained(checkpoint,from_pt=True,use_cache = False) # codeparrot
#model = TFAutoModel.from_pretrained(checkpoint,from_pt=True)

# declare signature function for declaring the input shape of our data  to the servable model
# define model will accept two inputs, input_ids and attention_mask
MAX_SEQ_LEN = 100

callable = tf.function(model.call)
concrete_function = callable.get_concrete_function(
    [tf.TensorSpec([None, MAX_SEQ_LEN], tf.int32, name="input_ids"), 
    tf.TensorSpec([None, MAX_SEQ_LEN], tf.int32, name="attention_mask"),
    #tf.TensorSpec([None, MAX_SEQ_LEN], tf.int32, name="decoder_input_ids"),
    #tf.TensorSpec([None, MAX_SEQ_LEN], tf.int32, name="decoder_attention_mask"),
    #tf.TensorSpec([1, 4, None, 12,None,64], tf.int32, name="past_key_values"),
    ]
)

text = "def hello_world():"
tokenizer = AutoTokenizer.from_pretrained(checkpoint) # ./bert-base-uncased

inputs = tokenizer.encode(text, return_tensors="tf")
labels = tf.constant([1], dtype=tf.int32)
labels = tf.expand_dims(labels, axis=0)
#labels = tf.placeholder(tf.int32, 1)
print(labels.shape)
#labels = tensorflow.tensor([1]).unsqueeze(0) 
#outputs = model.generate(inputs, max_length=10,max_new_tokens = 30)
outputs = model.generate(inputs, max_new_tokens = 30,labels=labels) # labels dont change it 

print(f'outputs {outputs}')

model_name = 'codeparrot_nokeys'
temp_model_dir = f'models/tf/{model_name}'

model.save_pretrained(temp_model_dir, saved_model=True, )