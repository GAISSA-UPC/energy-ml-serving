from transformers import pipeline

unmasker = pipeline('fill-mask', model='bert-base-uncased', device_map="auto")

response = unmasker("Hello I'm a [MASK] model.")
print(response)