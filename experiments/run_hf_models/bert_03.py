from transformers import T5Tokenizer, T5ForConditionalGeneration
from transformers import BertTokenizer, BertForMaskedLM
from torch.nn import functional as F
import torch


#text = input("Enter input ")
tokenizer = BertTokenizer.from_pretrained(pretrained_model_name_or_path = './bert-base-uncased')
# bert model for masked language modelling
model = BertForMaskedLM.from_pretrained(pretrained_model_name_or_path = './bert-base-uncased', return_dict = True)
text = "The capital of Mexico contains the Mayan "

#tokenizer = T5Tokenizer.from_pretrained("t5-small")
#model = T5ForConditionalGeneration.from_pretrained("t5-small", low_cpu_mem_usage=True)

# inputTokens = tokenizer(text, return_tensors="pt")
# outputs = model.generate(inputTokens['input_ids'], attention_mask=inputTokens['attention_mask'], max_new_tokens=50)
# print("Output:", tokenizer.decode(outputs[0], skip_special_tokens=True))


def infer_bert(text):
    input_ids = tokenizer(text, return_tensors="pt").input_ids
    outputs = model.generate(input_ids,max_new_tokens=10)

    return tokenizer.decode(outputs[0], skip_special_tokens=False)
    
print(infer_bert(text))

# for prompt in ["Hello, How are you?", "My name is Francisco"]:
#     print("Input:", prompt)
#     inputTokens = tokenizer("translate English to French: {}".format(prompt), return_tensors="pt")
#     outputs = model.generate(inputTokens['input_ids'], attention_mask=inputTokens['attention_mask'], max_new_tokens=50)
#     print("Output:", tokenizer.decode(outputs[0], skip_special_tokens=True))

def infer_t5(text):
    input_ids = tokenizer(text, return_tensors="pt").input_ids
    outputs = model.generate(input_ids)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)