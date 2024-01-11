# from transformers import pipeline

# unmasker = pipeline('fill-mask', model='bert-base-uncased')

# response = unmasker("Hello I'm a [MASK] model.")

# print(response)

from transformers import BertTokenizer, BertModel
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained("bert-base-uncased")
text = "Replace me by any text you'd like."
encoded_input = tokenizer(text, return_tensors='pt')
print(encoded_input)
output = model(**encoded_input)
print(output)
print(output[0].shape)

#TOKENIZER BERT
from transformers import BertTokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
# You can use this tokenizer for all the various bert models

# return_tensor - pytorch
# add_especial_tokens - beggining and end of sentence
encoding = tokenizer.encode_plus(text, add_special_tokens = True,    truncation = True, padding = "max_length", return_attention_mask = True, return_tensors = "pt")


mask_token = tokenizer.mask_token