import requests
from transformers import AutoTokenizer, RobertaTokenizer
import json
import torch
import tensorflow as tf

model_checkpoint = {'codet5-base':"Salesforce/codet5-base", 'codet5p-220':'Salesforce/codet5p-220m', 
                    'codegen-350-mono':"Salesforce/codegen-350M-mono", 'gpt-neo-125m':"EleutherAI/gpt-neo-125M",
                    'codeparrot-small':'codeparrot/codeparrot-small', 'pythia-410m':"EleutherAI/pythia-410m"} # model:checkpoint

tokenizer = AutoTokenizer.from_pretrained(model_checkpoint['codeparrot-small']) 
#print(tokenizer)
#AutoTokenizer.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")text = "I like you. I love you"
text = "def get_higher_nums():"
#encoded_input = tokenizer(text, pad_to_max_length=MAX_SEQ_LEN, max_length=MAX_SEQ_LEN)#TF Serve endpoint
encoded_input = tokenizer(text)#TF Serve endpoint

print(f'encoded_input: {encoded_input}')

#print(encoded_input.shape)
labels = torch.tensor([1]).unsqueeze(0)
print(labels.shape)
print(labels)

print(f'encoded_input: {encoded_input}')

url = "http://localhost:8503/v1/models/codeparrot:predict"




payload={"instances": [{"input_ids": encoded_input['input_ids'], "attention_mask": encoded_input['attention_mask'],
                        "token_type_ids": encoded_input['attention_mask'],}]}
                        #"decoder_input_ids": encoded_input['input_ids'] }]}

print(f'payload: {payload} ')
#print(f"input_ids: {encoded_input['input_ids'].dim_size(0)}")

#>> { "input_ids": [101, 1045, 2066, 2017, 1012, 1045, 2293, 2017,  102, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], "attention_mask": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] }
headers = {
  'Content-Type': 'application/json'
}

response = requests.request("POST", url, headers=headers, data=json.dumps(payload)).json()

print(f'response: {response}')
#print(f"response.text: {response['predictions']}")


predictions = response['predictions']
next_token_logits = predictions[ :]
next_token_scores = tf.nn.log_softmax(next_token_logits, axis=-1)
next_token_id = tf.argmax(next_token_scores, axis=-1)
next_token_id = tf.cast(
    tf.expand_dims(next_token_id, axis=0), dtype="int32"
)
print("next_token_id",next_token_id)
#print(f"response.text: {response['predictions'][0]}")

post = next_token_id[0][0]
#prediction = tokenizer.decode(outputs[0], skip_special_tokens=True)

#print(json.loads(response.text)['predictions'])

def postprocess(input_ids):
    return tokenizer.decode(input_ids, skip_special_tokens=True)

output = postprocess(post)
print(output)