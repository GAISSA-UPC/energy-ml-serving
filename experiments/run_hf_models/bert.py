# from transformers import BertTokenizer, BertModel
# tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
# model = BertModel.from_pretrained("bert-base-uncased")
# text = "Replace me by any text you'd like."
# encoded_input = tokenizer(text, return_tensors='pt')
# output = model(**encoded_input)

# To use it
from transformers import pipeline
import tensorflow as tf
from tensorflow import keras
import h5py

# h5, the weights of the model
# config.json, the model characteristics

MODELS_DIR = "../models"
BERT_MODEL = "tf_model.h5"
BERT_CONFIG_FILE = "config.json"

#Using the model
#unmasker = pipeline('fill-mask', model='bert-base-uncased')

#output = unmasker("The man worked as a [MASK].")
#print(output)


#model = tf.keras.Model(inputs=[tokens,attention], outputs=x)
#model = tf.keras.Model()
#model.load_weights(f'{MODELS_DIR}/{BERT_MODEL}')

# from tensorflow.keras.models import load_model
# loaded_model = load_model("models/tf_model.h5")
# loaded_model.predict("i love machine learning and google")

from transformers import BertConfig, BertModel
# if model is on hugging face Hub
#model = BertModel.from_pretrained("bert-base-uncased")
# from local folder


from transformers import BertTokenizer, BertModel
tokenizer = BertTokenizer.from_pretrained('./bert-model')
#model = BertModel.from_pretrained("bert-base-uncased")
model = BertModel.from_pretrained("./bert-model/")

text = "Replace me by any text you'd like."
encoded_input = tokenizer(text, return_tensors='pt')
output = model(**encoded_input)
print(output)

from transformers import pipeline
unmasker = pipeline('fill-mask', model='./bert-model')
output = unmasker("The man worked as a [MASK].")
print(output)


# Clone model (repo from Hugging Face)
# Code to use the model
# Define input to the API endpoint
# Create endpoint