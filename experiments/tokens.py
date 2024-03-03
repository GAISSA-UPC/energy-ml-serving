import torch
from transformers import AutoModelForCausalLM, AutoTokenizer,AutoModelForSeq2SeqLM

# torch.set_default_device("cuda")

model_name = "codeparrot/codeparrot-small"
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto", )
tokenizer = AutoTokenizer.from_pretrained(model_name, )

inputs = tokenizer('''def print_prime(n):
   """
   Print all primes between 1 and n
   """''', return_tensors="pt", )
#def rolling_max(numbers: List[int]) ->
inputs = tokenizer('def rolling_max(numbers: List[int]) ->', return_tensors="pt", )

outputs = model.generate(**inputs, max_length=128) 
text = tokenizer.decode(outputs[0])#[0]
print(f'text2: {text}')


outputs_2 = model.generate(**inputs,max_new_tokens=128) 
text_2 = tokenizer.decode(outputs_2[0])#[0]
print(f'text2: {text_2}')



prompt_tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
output_tokens_1 = tokenizer.convert_ids_to_tokens(outputs[0])
output_tokens_2 = tokenizer.convert_ids_to_tokens(outputs_2[0])
num_prompt_tokens = len(prompt_tokens)
num_output_tokens = len(output_tokens_1)
num_output_tokens_2 = len(output_tokens_2)
print("Number of tokens in prompt:", num_prompt_tokens)
print("Number of tokens from max_length output:", num_output_tokens)
print("Number of tokens from max_new_tokens output:", num_output_tokens_2)