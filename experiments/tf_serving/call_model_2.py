
import requests
from transformers import AutoTokenizer, RobertaTokenizer
import json
import torch
import tensorflow as tf
import numpy as np

model_checkpoint = {'codet5-base':"Salesforce/codet5-base", 'codet5p-220':'Salesforce/codet5p-220m', 
                    'codegen-350-mono':"Salesforce/codegen-350M-mono", 'gpt-neo-125m':"EleutherAI/gpt-neo-125M",
                    'codeparrot-small':'codeparrot/codeparrot-small', 'pythia-410m':"EleutherAI/pythia-410m"} # model:checkpoint

tokenizer = AutoTokenizer.from_pretrained(model_checkpoint['codet5-base']) 
#print(tokenizer)
#AutoTokenizer.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")text = "I like you. I love you"
text = 'hello <extra_id_0>!'
#encoded_input = tokenizer(text, pad_to_max_length=MAX_SEQ_LEN, max_length=MAX_SEQ_LEN)#TF Serve endpoint
#encoded_input = tokenizer(text)#TF Serve endpoint
encoded_input = tokenizer(text) #TF Serve endpoint

print(f'encoded_input: {encoded_input}')

#print(encoded_input.shape)
labels = torch.tensor([1]).unsqueeze(0)
print(labels.shape)
print(labels)

print(f'encoded_input: {encoded_input}')

url = "http://localhost:8503/v1/models/codet52:predict"


def postprocess(input_ids):
    return tokenizer.decode(input_ids, skip_special_tokens=True)

# output = postprocess(post)
# print(output)
payload={"instances": [{"input_ids": encoded_input['input_ids'], "attention_mask": encoded_input['attention_mask'],
                        "decoder_attention_mask": encoded_input['attention_mask'],"decoder_input_ids": encoded_input['attention_mask'] }]}

json_data = {"signature_name": "serving_default", "instances": [{"input_ids": encoded_input['input_ids'], "attention_mask": encoded_input['attention_mask'],
                        "decoder_attention_mask": encoded_input['attention_mask'],"decoder_input_ids": encoded_input['attention_mask'] }]}

resp = requests.post(url, json=json_data).json()
print('response: ',resp)

predictions = resp['predictions'][-1]['logits']

# The returned results are probabilities, that can be positive/negative hence we take their absolute value
abs_scores = np.abs(predictions)
# Take the argmax that correspond to the index of the max probability.
label_id = np.argmax(abs_scores)
# Print the proper LABEL with its index

print(f'label_id {label_id}')

next_token_logits = predictions[ :]
next_token_scores = tf.nn.log_softmax(next_token_logits, axis=-1)
next_token_id = tf.argmax(next_token_scores, axis=-1)
next_token_id = tf.cast(
    tf.expand_dims(next_token_id, axis=0), dtype="int32"
)
print("next_token_id",next_token_id)
#print(f"response.text: {response['predictions'][0]}")

post = next_token_id[0]
#prediction = tokenizer.decod

generated = postprocess(post)
print(generated)