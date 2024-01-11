from transformers import T5Tokenizer, T5ForConditionalGeneration
from codecarbon import track_emissions

tokenizer = T5Tokenizer.from_pretrained("t5-small")
model = T5ForConditionalGeneration.from_pretrained("t5-small", low_cpu_mem_usage=True)

@track_emissions
def infer_t5(text):
    input_ids = tokenizer(text, return_tensors="pt").input_ids
    outputs = model.generate(input_ids)

    return tokenizer.decode(outputs[0], skip_special_tokens=True)

text = "Hello men, this is a test to get the energy metrics of a inference using a machine learning model"

infer_t5(text)