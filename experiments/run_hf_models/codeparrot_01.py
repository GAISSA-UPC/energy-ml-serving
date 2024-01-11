""" Simple script to generate code using codeparrot model
    
"""
from transformers import AutoTokenizer, AutoModelWithLMHead
  
tokenizer = AutoTokenizer.from_pretrained("codeparrot/codeparrot")
model = AutoModelWithLMHead.from_pretrained("codeparrot/codeparrot")

inputs = tokenizer("def hello_world():", return_tensors="pt")
outputs = model(**inputs)

print(outputs)

