from transformers import T5ForConditionalGeneration, AutoTokenizer

checkpoint = "Salesforce/codet5p-220m"
device = "cpu" # for GPU usage or "cpu" for CPU usage

tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = T5ForConditionalGeneration.from_pretrained(checkpoint).to(device)

text = "def get_random_element(my_dictionary):<extra_id_0>"
inputs = tokenizer.encode(text, return_tensors="pt").to(device)
#outputs = model.generate(inputs, max_length=10,max_new_tokens = 30)
outputs = model.generate(inputs, max_new_tokens = 30)

print(tokenizer.decode(outputs[0], skip_special_tokens=True))
# ==> print "Hello World"
