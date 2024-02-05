import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# 'codegen-350M-mono',
models = [ 'codet5-base', 'codet5p-220',  'gpt-neo-125m', 'codeparrot-small', 'pythia-410m'] # bloom, pythia
models = [ 'codet5-base', 'codet5p-220', 'codeparrot-small', 'pythia-410m'] # bloom, pythia

model_checkpoint = {'codet5-base':"Salesforce/codet5-base",
                    'codet5p-220':'Salesforce/codet5p-220m',
                    'codegen-350M-mono':"Salesforce/codegen-350M-mono",
                    'gpt-neo-125m':"EleutherAI/gpt-neo-125M",
                    'codeparrot-small':'codeparrot/codeparrot-small',
                    'pythia-410m':"EleutherAI/pythia-410m"}

def infer(text: str, model, tokenizer) -> str:
    #text = "def greet(user): print(f'hello <extra_id_0>!')"
    input_ids = tokenizer(text, return_tensors="pt",max_length=20,padding='max_length').input_ids

    # simply generate a single sequence
    #generated_ids = model.generate(input_ids, max_length=8)
    generated_ids = model.generate(input_ids,)
    
    # decode
    prediction = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    
    return prediction

if 2<3:
    for model in models[:2]:
        # select checkpoint
        model_name = model
        checkpoint = model_checkpoint[model_name]

        print(f"------------------------ Loading model: {checkpoint} ------------------------")

        # Load the TorchScript model
        loaded_model = torch.jit.load(f"models/torchscript2/{model_name}.pt")
        #print(loaded_model.code)

        loaded_model.eval()  # Set the model to evaluation mode, turn off gradients computation

        # Load the tokenizer corresponding to the model
        tokenizer = AutoTokenizer.from_pretrained(checkpoint)


        input_text = "def hello_world():"
        input_text = "if condition:\n            return condition\n        else:\n            return None\n\n    def _get_condition(self"
        input_text = "def hello_world(): print("
        #tokenizer.add_special_tokens({'pad_token': '[PAD]'})

        #dummy_input = "def hello_world():"
        #inputs = tokenizer.encode_plus(dummy_input,max_length = int(20),pad_to_max_length = True, add_special_tokens = True, return_tensors = 'pt')
        #inputs = tokenizer.encode_plus(input_text,max_length = int(30),padding = True, add_special_tokens = True, return_tensors = 'pt',truncation=True)
        inputs = tokenizer.encode_plus(input_text, return_tensors = 'pt',max_length=20,padding='max_length')
        #input_ids = tokenizer(input_text, return_tensors="pt",max_length=20,padding='max_length').input_ids
        print("input_ids: ",inputs)
        input_ids = inputs["input_ids"]
    
        attention_mask = inputs["attention_mask"]
        input_tuple = [input_ids,attention_mask,input_ids] # decoder_input_ids["input_ids"]

        #input_ids = tokenizer.encode(input_text, return_tensors="pt")
        # print("--------------Encoded inputs---------")
        # print(input_ids)
        # print(len(input_ids))
        # print(input_ids[0])
        # print(len(input_ids[0]))


        # Generate predictions from the model
        with torch.no_grad():
            
            if model_name in [ 'codet5-base', 'codet5p-220']:
                print(attention_mask)
                output = loaded_model(input_ids,attention_mask = attention_mask, decoder_input_ids = input_ids)  #t5
            else:
                output = loaded_model(input_ids,)  # Adjust max_length as needed


        print(output[0])
        print(output[0].shape)

        # Convert the output tensor to token IDs
        predicted_token_ids = torch.argmax(output[0], dim=-1)

        # Convert the tensor to a list of lists
        predicted_token_ids = predicted_token_ids.tolist()

        print("predicted tokens:",predicted_token_ids)
        #print(output.shape)
        # Decode and print the output
        output_text = tokenizer.decode(predicted_token_ids[0], skip_special_tokens=True,predict_with_generate=True)
        print("Result:")
        print(output_text)

# for model in models[1:2]:
#     # select checkpoint
#     model_name = model
#     checkpoint = model_checkpoint[model_name]

#     print(f"------------------------ Loading model 2: {checkpoint} name {model_name}------------------------")

#     # Load the TorchScript model
#     loaded_model = torch.jit.load(f"models/torchscript/{model_name}.pt")
#     #print(loaded_model.code)

#     loaded_model.eval()  # Set the model to evaluation mode, turn off gradients computation

#     # Load the tokenizer corresponding to the model
#     tokenizer = AutoTokenizer.from_pretrained(checkpoint)


#     input_text = "def hello_world():"

#     prediction = None
#     #with torch.no_grad():
#     prediction = infer(input_text,loaded_model,tokenizer)
#     print(prediction)
    