from transformers import BertTokenizer, BertForMaskedLM
from torch.nn import functional as F
import torch



low_cpu = True

tokenizer = BertTokenizer.from_pretrained(pretrained_model_name_or_path = './bert-base-uncased')
# bert model for masked language modelling
model = None
if low_cpu:
   #model = BertForMaskedLM.from_pretrained(pretrained_model_name_or_path = './bert-base-uncased', return_dict = True, low_cpu_mem_usage=True)
   model = BertForMaskedLM.from_pretrained(pretrained_model_name_or_path = './bert-base-uncased', return_dict = True, device_map="auto")
   # ValueError: BertForMaskedLM does not support `device_map='auto'` yet.
else:
   # auto type of torch, optimal
   model = BertForMaskedLM.from_pretrained(pretrained_model_name_or_path = './bert-base-uncased', return_dict = True, torch_dtype="auto")
   #model = BertForMaskedLM.from_pretrained(pretrained_model_name_or_path = './bert-base-uncased', return_dict = True, torch_dtype=torch.float16)
   
print(model)



device = torch.device("cpu")
#model.to(device)
# return_dict True to use mask token
text = "I work as " + tokenizer.mask_token + "."
#text = "The most important tech company in Mexico is " + "['MASK']" + "."

input = tokenizer.encode_plus(text, return_tensors = "pt")
#print("device ",input.device)
#print(input.shape)
#input = input[0].cpu()
print("type ",type(input))
#print("device ",input.device)

print("input1: ", input)
print("input2: ", input["input_ids"][0])
# get index of the mask
mask_index = torch.where(input["input_ids"][0] == tokenizer.mask_token_id)
#print("device ",mask_index.device)
print(mask_index)
#mask_index = mask_index[0].cpu()  # Move mask_index to CPU
#print("device ",mask_index.device)

#mask_word = softmax[0, mask_index, :]


print("input_ids device ",input['input_ids'].device)
print("device ",model.device)

#print("input 1", *input)
#print(input["input_ids"])
#model.to(device)
with torch.no_grad():
   output = model(input["input_ids"])
print("output, ", output)
# logits are output of BERT model before softmax activation
logits = output.logits
print(logits)
print(logits.shape)
print("Here")

softmax = F.softmax(logits, dim = -1)
print(softmax)
print(softmax.shape)

mask_word = softmax[0, mask_index, :]
print("top 10")

top_10 = torch.topk(mask_word, 10, dim = 1)[1][0]
print(top_10)
print("Decoding convert ids to tokens")
# problem with decode
print(tokenizer.convert_ids_to_tokens(top_10))
print("Decoding")

