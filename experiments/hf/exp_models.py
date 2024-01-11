# Check to run models of millions of parameters
# path of models D:\GAISSA\cloud-api\app_env\lib\site-packages\transformers

# file:///C:/Users/fjdur/Downloads/entropy-25-00888-v2.pdf
#./home/fjdur/.local/lib/python3.8/site-packages/onnxruntime/transformers
#./home/fjdur/.local/lib/python3.8/site-packages/transformers

models = ['codeparrot_1B', 't5-base', 'santacoder','gpt-neo','incoder-1b', 'codegen','bert-large',
          'bloom-560', 'codegen2-1b', 'pythia-1b'
          ]
base_model = models[3]

print(f'Running {base_model} ---------------------------')

if base_model == 'codeparrot_1B': # needs to download 6.6GB 
    from transformers import AutoTokenizer, AutoModelWithLMHead
    
    tokenizer = AutoTokenizer.from_pretrained("codeparrot/codeparrot")
    model = AutoModelWithLMHead.from_pretrained("codeparrot/codeparrot")

    inputs = tokenizer("def hello_world():", return_tensors="pt")
    outputs = model(**inputs)

elif base_model == 't5-base': # 890M, dowloaded, and running
    from transformers import T5Tokenizer, T5ForConditionalGeneration

    tokenizer = T5Tokenizer.from_pretrained("t5-base")

    model = T5ForConditionalGeneration.from_pretrained("t5-base")

    input_ids = tokenizer("translate English to German: The house is wonderful.", return_tensors="pt").input_ids

    outputs = model.generate(input_ids)

    print(tokenizer.decode(outputs[0], skip_special_tokens=True))
elif base_model == 'gpt-neo': # 5gb
    from transformers import pipeline
    # generator = pipeline('text-generation', model='EleutherAI/gpt-neo-125M')
    # outputs = generator("EleutherAI has", do_sample=True, min_length=20)
    
    generator = pipeline('text-generation', model='EleutherAI/gpt-neo-1.3B')
    output = generator("def hello_world", do_sample=True, min_length=50)
    print("----------")
    print(output)

elif base_model == 'santacoder':
    from transformers import AutoModelForCausalLM, AutoTokenizer

    checkpoint = "bigcode/santacoder"
    device = "cuda" # for GPU usage or "cpu" for CPU usage

    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    model = AutoModelForCausalLM.from_pretrained(checkpoint, trust_remote_code=True).to(device)

    inputs = tokenizer.encode("def print_hello_world():", return_tensors="pt").to(device)
    outputs = model.generate(inputs)
    print(tokenizer.decode(outputs[0]))

elif base_model == 'incoder-1b': # 2.2 gb
    from transformers import AutoModelForCausalLM, AutoTokenizer
    model = AutoModelForCausalLM.from_pretrained("facebook/incoder-1B")
    tokenizer = AutoTokenizer.from_pretrained("facebook/incoder-1B")
    output = tokenizer.decode(tokenizer.encode("from ."), clean_up_tokenization_spaces=False)
    print(output)
elif base_model == 'codegen': # 350-multi
    from transformers import AutoTokenizer, AutoModelForCausalLM
    tokenizer = AutoTokenizer.from_pretrained("Salesforce/codegen-350M-multi")
    model = AutoModelForCausalLM.from_pretrained("Salesforce/codegen-350M-multi")

    text = "def hello_world():"
    input_ids = tokenizer(text, return_tensors="pt").input_ids

    generated_ids = model.generate(input_ids, max_length=128)
    print(tokenizer.decode(generated_ids[0], skip_special_tokens=True))
elif base_model == 'bert-large': # 350-multi
    from transformers import BertTokenizer, BertModel
    tokenizer = BertTokenizer.from_pretrained('bert-large-uncased')
    model = BertModel.from_pretrained("bert-large-uncased")
    text = "Replace me by any text you'd like."
    encoded_input = tokenizer(text, return_tensors='pt')
    output = model(**encoded_input)
    print(output)
    print(tokenizer.decode(output))

elif base_model == 'bloom-560': # 560
    from transformers import pipeline
    pipe = pipeline("text-generation", model="bigscience/bloom-560m")
    print(pipe('def hello_world('))

elif base_model == 'codegen2-1b': #4.13gb
    from transformers import AutoTokenizer, AutoModelForCausalLM
    tokenizer = AutoTokenizer.from_pretrained("Salesforce/codegen2-1B")
    model = AutoModelForCausalLM.from_pretrained("Salesforce/codegen2-1B", trust_remote_code=True, revision="main")

    text = "def hello_world():"
    input_ids = tokenizer(text, return_tensors="pt").input_ids
    generated_ids = model.generate(input_ids, max_length=128)
    print(tokenizer.decode(generated_ids[0], skip_special_tokens=True))

elif base_model == 'pythia-1b': # 2-09gb
    from transformers import GPTNeoXForCausalLM, AutoTokenizer

    model = GPTNeoXForCausalLM.from_pretrained(
        "EleutherAI/pythia-1b",
        revision="step143000",
        cache_dir="./pythia-1b/step143000",
    )

    tokenizer = AutoTokenizer.from_pretrained(
    "EleutherAI/pythia-1b",
    revision="step143000",
    cache_dir="./pythia-1b/step143000",
    )

    inputs = tokenizer("Hello, I am", return_tensors="pt")
    tokens = model.generate(**inputs)
    output = tokenizer.decode(tokens[0])
    print(output)

