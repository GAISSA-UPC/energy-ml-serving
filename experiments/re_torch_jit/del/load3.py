import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


models = [ 'codet5-base', 'codet5p-220', 'codegen-350-mono', 'gpt-neo-125m', 'codeparrot-small', 'pythia-410m'] # bloom, pythia
model_checkpoint = {'codet5-base':"Salesforce/codet5-base", 'codet5p-220':'Salesforce/codet5p-220m', 
                    'codegen-350M-mono':"Salesforce/codegen-350M-mono", 'gpt-neo-125m':"EleutherAI/gpt-neo-125M",
                    'codeparrot-small':'codeparrot/codeparrot-small', 'pythia-410m':"EleutherAI/pythia-410m"} # model:checkpoint

# select checkpoint
name = models[-2]
checkpoint = model_checkpoint[name]
model = name

print("------------------------ Loading model")
# Load the TorchScript model
loaded_model = torch.jit.load(f"models/torchscript/{name}_2.pt")

print(loaded_model.code)

loaded_model.eval()  # Set the model to evaluation mode

# Load the tokenizer corresponding to the model
#tokenizer_checkpoint = "path/to/your/tokenizer"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)


input_text = "def hello_world():'"

tokenizer.add_special_tokens({'pad_token': '[PAD]'})

#dummy_input = "def hello_world():"
#inputs = tokenizer.encode_plus(dummy_input,max_length = int(20),pad_to_max_length = True, add_special_tokens = True, return_tensors = 'pt')
inputs = tokenizer.encode_plus(input_text,max_length = int(20),padding = True, add_special_tokens = True, return_tensors = 'pt',truncation=True)

input_ids = inputs["input_ids"]

#input_ids = tokenizer.encode(input_text, return_tensors="pt")
print("--------------Encoded inputs---------")
print(input_ids)
print(len(input_ids))
print(input_ids[0])
print(len(input_ids[0]))


# Generate predictions from the model
with torch.no_grad():
    output = loaded_model(input_ids,)  # Adjust max_length as needed

print(output[0])
print(output[0].shape)

# Convert the output tensor to token IDs
predicted_token_ids = torch.argmax(output[0], dim=-1)

# Convert the tensor to a list of lists
predicted_token_ids = predicted_token_ids.tolist()

print("predicted tokens:",predicted_token_ids)
#print(output.shape)
# Decode and print the output
output_text = tokenizer.decode(predicted_token_ids[0], skip_special_tokens=True,predict_with_generate=True)
print("Result:")
print(output_text)
