from transformers import BertModel, BertTokenizer, BertConfig
import torch
from transformers import AutoTokenizer, AutoConfig
from transformers import AutoModelForCausalLM


from transformers import T5ForConditionalGeneration
from transformers import T5Tokenizer, RobertaTokenizer

models = [ 'codet5-base', 'codet5p-220', 'codegen-350M-mono', 'gpt-neo-125m', 'codeparrot-small', 'pythia-410m'] # bloom, pythia


model_checkpoint = {'codet5-base':"Salesforce/codet5-base", 'codet5p-220':'Salesforce/codet5p-220m', 
                    'codegen-350M-mono':"Salesforce/codegen-350M-mono", 'gpt-neo-125m':"EleutherAI/gpt-neo-125M",
                    'codeparrot-small':'codeparrot/codeparrot-small', 'pythia-410m':"EleutherAI/pythia-410m"} # model:checkpoint


# select checkpoint
name = models[2]
checkpoint = model_checkpoint[name]
model = name

# change to your directory
save_directory = f"models/torchscript/{model}/"

print(f"Saving {model} ...\n")
print(f"Checkpoint {checkpoint} ...\n")

config = AutoConfig.from_pretrained(checkpoint,torchscript=True)

model = AutoModelForCausalLM.from_pretrained(checkpoint, torchscript =True)
#model = T5ForConditionalGeneration.from_pretrained(checkpoint, torch_dtype = 'auto') # t5 models
#tokenizer = AutoTokenizer.from_pretrained(checkpoint)

tokenizer = AutoTokenizer.from_pretrained(checkpoint, torchscript=True)

tokenizer.add_special_tokens({'pad_token': '[PAD]'})

dummy_input = "def hello_world():"
#inputs = tokenizer.encode_plus(dummy_input,max_length = int(20),pad_to_max_length = True, add_special_tokens = True, return_tensors = 'pt')
inputs = tokenizer.encode_plus(dummy_input,max_length = int(20),padding = True, add_special_tokens = True, return_tensors = 'pt',truncation=True)

input_ids = inputs["input_ids"]
traced_model = torch.jit.trace(model, [input_ids])

save_directory = f"models/torchscript/{name}_2.pt"
torch.jit.save(traced_model,save_directory)

# # Tokenizing input text
# text = "def hello_world():"
# inputs = tokenizer.tokenize(text, return_tensors="pt", truncation=True)
# print(inputs)
# indexed_tokens = tokenizer.convert_tokens_to_ids(inputs)
# #tokenized_text = model.tokenize(text)
# print(indexed_tokens)

# tokens_tensor = torch.tensor([indexed_tokens])
# print(tokens_tensor)

# input_tuple = (tokens_tensor, tokens_tensor)

# #traced_model = torch.jit.trace(model, [tokens_tensor,tokens_tensor])
# traced_model = torch.jit.trace(model, tokens_tensor) # encoded decoded models

# #tokens = ort_model.generate(**inputs)
# #print(tokens)

# #change
# save_directory = f"models/torchscript/{name}_2.pt"
# traced_model.save(save_directory)
# #model.save_pretrained(save_directory, saved_model=True)


# loaded_model = torch.jit.load(f"models/torchscript/{name}_2.pt")
# loaded_model.eval()

# all_encoder_layers, pooled_output = loaded_model(*input_tuple)

# print(all_encoder_layers)
# print(pooled_output)
