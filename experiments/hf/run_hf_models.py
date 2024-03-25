
'''
Running the models just with HF
codet5-large 770M
codet5p-770 770M
codegen-350-mono 350M
gpt-neo-125m 125M
gpt-neo-1.3B 1.3B
codeparrot-small 110M

models downloaded  ... tokenizer.json
'''
import torch
models = ['codet5-large', 'codet5p-770', 'codegen-350-mono', 'gpt-neo-125m', 'gpt-neo-1.3B', 'codeparrot-small',
          'pythia-410m'
          ]
models = [ 'codet5-base', 'codet5p-220', 'codegen-350-mono', 
          'gpt-neo-125m', 'codeparrot-small', 'pythia-410m'] # bloom, pythia
models = ['tinyllama']

base_model = 'tinyllama'

for model in models:
    base_model = model
    print(f'Running {base_model} ---------------------------\n')

    #home/fjdur/.cache/huggingface/hub/models--Salesforce--codet5-large/snapshots/7430ce16cc8c0f8db091c561a925047507734575/config.json
    if base_model == 'codet5-large': 
        from transformers import AutoTokenizer, T5ForConditionalGeneration
        tokenizer = AutoTokenizer.from_pretrained("Salesforce/codet5-large")
        model = T5ForConditionalGeneration.from_pretrained("Salesforce/codet5-large")
        text = "def greet(user): print(f'hello <extra_id_0>!')"
        input_ids = tokenizer(text, return_tensors="pt").input_ids

        # simply generate a single sequence
        generated_ids = model.generate(input_ids, max_length=8)
        print(tokenizer.decode(generated_ids[0], skip_special_tokens=True))

    if base_model == 'codet5-base': 
        from transformers import AutoTokenizer, T5ForConditionalGeneration
        tokenizer = AutoTokenizer.from_pretrained("Salesforce/codet5-base")
        model = T5ForConditionalGeneration.from_pretrained("Salesforce/codet5-base")
        text = "def greet(user): print(f'hello <extra_id_0>!')"
        input_ids = tokenizer(text, return_tensors="pt").input_ids

        # simply generate a single sequence
        generated_ids = model.generate(input_ids, max_length=8)
        print(tokenizer.decode(generated_ids[0], skip_special_tokens=True))

    elif base_model == 'codet5p-770': # 890M, dowloaded, and running
        from transformers import T5ForConditionalGeneration, AutoTokenizer

        checkpoint = "Salesforce/codet5p-770m"
        device = "cpu" # for GPU usage or "cpu" for CPU usage

        tokenizer = AutoTokenizer.from_pretrained(checkpoint)
        model = T5ForConditionalGeneration.from_pretrained(checkpoint).to(device)

        inputs = tokenizer.encode("def print_hello_world():<extra_id_0>", return_tensors="pt").to(device)
        outputs = model.generate(inputs, max_length=10)
        print(tokenizer.decode(outputs[0], skip_special_tokens=True))
        # ==> print "Hello World"

    elif base_model == 'codet5p-220': # 890M, dowloaded, and running
        from transformers import T5ForConditionalGeneration, AutoTokenizer

        checkpoint = "Salesforce/codet5p-220m"
        device = "cpu" # for GPU usage or "cpu" for CPU usage

        tokenizer = AutoTokenizer.from_pretrained(checkpoint)
        model = T5ForConditionalGeneration.from_pretrained(checkpoint).to(device)

        inputs = tokenizer.encode("def print_hello_world():<extra_id_0>", return_tensors="pt").to(device)
        outputs = model.generate(inputs, max_length=10)
        print(tokenizer.decode(outputs[0], skip_special_tokens=True))
        # ==> print "Hello World"

    elif base_model == 'codegen-350-multi': # 350-multi
        from transformers import AutoTokenizer, AutoModelForCausalLM
        tokenizer = AutoTokenizer.from_pretrained("Salesforce/codegen-350M-multi")
        model = AutoModelForCausalLM.from_pretrained("Salesforce/codegen-350M-multi")

        text = "def hello_world():"
        input_ids = tokenizer(text, return_tensors="pt").input_ids

        generated_ids = model.generate(input_ids, max_length=128)
        print(tokenizer.decode(generated_ids[0], skip_special_tokens=True))

    elif base_model == 'codegen-350-mono': # 350-multi
        from transformers import AutoTokenizer, AutoModelForCausalLM
        tokenizer = AutoTokenizer.from_pretrained("Salesforce/codegen-350M-mono")
        model = AutoModelForCausalLM.from_pretrained("Salesforce/codegen-350M-mono")

        text = "def hello_world():"
        input_ids = tokenizer(text, return_tensors="pt").input_ids

        generated_ids = model.generate(input_ids, max_length=128)
        print(tokenizer.decode(generated_ids[0], skip_special_tokens=True))

    elif base_model == 'gpt-neo-125m': 
        from transformers import pipeline
        generator = pipeline('text-generation', model='EleutherAI/gpt-neo-125M')
        output = generator("EleutherAI has", do_sample=True, min_length=20)
        print(output)

    elif base_model == 'gpt-neo-1.3B': 
        from transformers import pipeline
        # generator = pipeline('text-generation', model='EleutherAI/gpt-neo-125M')
        # outputs = generator("EleutherAI has", do_sample=True, min_length=20)
        
        generator = pipeline('text-generation', model='EleutherAI/gpt-neo-1.3B')
        output = generator("def hello_world", do_sample=True, min_length=50)
        print("----------")
        print(output)

    elif base_model == 'codeparrot-small': 
        # from transformers import AutoTokenizer, AutoModelWithLMHead
    
        # tokenizer = AutoTokenizer.from_pretrained("codeparrot/codeparrot-small")
        # model = AutoModelWithLMHead.from_pretrained("codeparrot/codeparrot-small")

        # inputs = tokenizer("def hello_world():", return_tensors="pt")
        # outputs = model(**inputs)
        # #print(outputs)
        # print(tokenizer.decode(outputs[0]))
        
        from transformers import pipeline

        pipe = pipeline("text-generation", model="codeparrot/codeparrot-small")
        outputs = pipe("def hello_world():")

        print(outputs)
        
    elif base_model == 'pythia-410m': 

        from transformers import GPTNeoXForCausalLM, AutoTokenizer

        model = GPTNeoXForCausalLM.from_pretrained(
        "EleutherAI/pythia-410m",
        revision="step3000",
        cache_dir="./pythia-410m/step3000",
        )

        tokenizer = AutoTokenizer.from_pretrained(
        "EleutherAI/pythia-410m",
        revision="step3000",
        cache_dir="./pythia-410m/step3000",
        )

        inputs = tokenizer("def hello_world", return_tensors="pt")
        tokens = model.generate(**inputs)
        print(tokenizer.decode(tokens[0]))
    elif base_model == 'tinyllama': 
        
        from transformers import pipeline
        pipe = pipeline("text-generation", model="TinyLlama/TinyLlama-1.1B-Chat-v1.0", torch_dtype=torch.bfloat16, device_map="auto")

        #pipe = pipeline("text-generation", model="codeparrot/codeparrot-small")
        outputs = pipe("Write a sum function")

        print(outputs)