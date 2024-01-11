""" Simple script to generate code using codegen model
    
    https://huggingface.co/docs/transformers/model_doc/codegen
    
"""
from transformers import AutoModelForCausalLM, AutoTokenizer


#checkpoint = "Salesforce/codegen-350M-nl"
#checkpoint = "Salesforce/codegen-350M-multi"
checkpoint = "Salesforce/codegen-350M-mono"

model = AutoModelForCausalLM.from_pretrained(checkpoint)

tokenizer = AutoTokenizer.from_pretrained(checkpoint)

text = "def print_list(a):"
text = "def get_random_element(dictionary):"

completion = model.generate(**tokenizer(text, return_tensors="pt"))
#completion = model.generate(**tokenizer(text, return_tensors="pt"),max_new_tokens =25)
print(tokenizer.decode(completion[0]))