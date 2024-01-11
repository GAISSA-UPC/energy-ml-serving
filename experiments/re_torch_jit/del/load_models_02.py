from transformers import BertModel, BertTokenizer, BertConfig
import torch
#import torch.neuron

from transformers import BertModel, BertTokenizer, BertConfig
import torch
from transformers import AutoTokenizer
from transformers import AutoModelForCausalLM, AutoTokenizer

from transformers import T5ForConditionalGeneration

models = [ 'codet5-base', 'codet5p-220', 'codegen-350-mono', 'gpt-neo-125m', 'codeparrot-small', 'pythia-410m'] # bloom, pythia


model_checkpoint = {'codet5-base':"Salesforce/codet5-base", 'codet5p-220':'Salesforce/codet5p-220m', 
                    'codegen-350M-mono':"Salesforce/codegen-350M-mono", 'gpt-neo-125m':"EleutherAI/gpt-neo-125M",
                    'codeparrot-small':'codeparrot/codeparrot-small', 'pythia-410m':"EleutherAI/pythia-410m"} # model:checkpoint

# select checkpoint
name = models[-2]
checkpoint = model_checkpoint[name]
model = name



tokenizer = AutoTokenizer.from_pretrained(checkpoint)
#enc = BertTokenizer.from_pretrained("bert-base-uncased")


# Tokenizing input text
text = "def hello_world():"
inputs = tokenizer.tokenize(text, return_tensors="pt", truncation=True)
print(inputs)
indexed_tokens = tokenizer.convert_tokens_to_ids(inputs)
#tokenized_text = model.tokenize(text)
print(indexed_tokens)

tokens_tensor = torch.tensor([indexed_tokens])
print(tokens_tensor)


loaded_model = torch.jit.load(f"models/torchscript/{name}.pt")
loaded_model.eval()

# Input text for inference
input_text = "def hello_world():'"
input_ids = tokenizer.encode(input_text, return_tensors="pt")
print(input_ids)

print(len(input_ids))



# Generate output
with torch.no_grad():
    output = loaded_model(input_ids)

print(type(output[0]))
print(type(output))

# Decode and print the output
output_text = tokenizer.decode(output[0].tolist()   , skip_special_tokens=True)
print("Result:")
print(output_text)

#all_encoder_layers, pooled_output = loaded_model(*dummy_input)

# res = loaded_model(tokens_tensor)

# print(res)
# print(type(res))
# print(res[0].shape)
# print(len(res))


# input_token = tokenizer.decode(res[0], skip_special_tokens=True)

# #input_token = tokenizer.decode(res[0], return_tensors="pt").input_ids

# print(input_token)

# #torch.jit.trace(model, [tokens_tensor, segments_tensors])
# #torch.neuron.trace(model, [token_tensor, segments_tensors])