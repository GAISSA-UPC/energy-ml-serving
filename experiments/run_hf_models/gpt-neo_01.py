from transformers import GPTNeoForCausalLM, GPT2Tokenizer

"""
    Too large
"""
model = GPTNeoForCausalLM.from_pretrained("EleutherAI/gpt-neo-1.3B")

tokenizer = GPT2Tokenizer.from_pretrained("EleutherAI/gpt-neo-1.3B")

prompt = ("def get_random(dictionary):")

input_ids = tokenizer(prompt, return_tensors="pt").input_ids

gen_tokens = model.generate(

    input_ids,

    do_sample=True,

    temperature=0.9,

    max_length=100,

)

gen_text = tokenizer.batch_decode(gen_tokens)[0]
print(gen_text)