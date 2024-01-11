from transformers import GPT2Tokenizer, GPT2Model
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2Model.from_pretrained('gpt2')
text = input("Replace me by any text you'd like. ")
encoded_input = tokenizer(text, return_tensors='pt')
output = model(**encoded_input)
print(output)

print(tokenizer.decode(output[0], skip_special_tokens=True))