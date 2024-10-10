"""_summary_

Get dataset. 
Random tokens

Takes one problem of HumanEval and select two random position of the prompt string, 
create a new string of it, tokenize it, check if number of tokens is n or less 
and if it meets condition save this new string

Selection:
- for each prompt:
    - select random line
    - tokenize line
    - then select tokens from start to the end of line (the end token is random)
    - decode and save new prompt

Original features:
    task_id
    prompt
    entry_point
    canonical_solution
    test

Saved features:
    task_id:
    selection:

ToDo: 
[] To select n continued lines
"""
import json
import random
import torch
from transformers import AutoTokenizer

#input_ids = tokenizer(text, return_tensors="pt",).input_ids
with open('./HumanEval.jsonl', 'r') as json_file:
    json_list = list(json_file)

problems = []

for json_str in json_list:
    result = json.loads(json_str)
    print(f"result: {result}")
    print(isinstance(result, dict))
    problems.append(result)

#list of prompts, each list is a list of all the lines
prompts_by_lines = []
for problem in problems:
    print(problem['task_id'],"--------------------")
    prompt = problem['prompt']
    lines = prompt.split('\n')
    #print(lines)
    prompts_by_lines.append(lines)

print(f"all: {prompts_by_lines}")
print(f"all len: {len(prompts_by_lines)}")



tokenizer = AutoTokenizer.from_pretrained("Salesforce/codet5-base")
min_n = 10
max_n = 15
len_tokens = 0
i=0
for problem in prompts_by_lines:
    meets_criteria = False

    print(f"\n\nPROBLEM -> {i}")

    new_json = {}

    new_json["task_id"] = "RandomTokens/"+str(i)

    
    #prompt = problem["prompt"]

    while not meets_criteria:
        selected_line = random.choice(problem)
        #selected_line = problem[i][selected_n_line]
        print(f"selected_line: {selected_line}")
        line_tokens = tokenizer(selected_line, return_tensors="pt",).input_ids[0]
        print(f"line_tokens: {line_tokens}")
        print(f"line_tokens: {len(line_tokens)}")

        # Select two random positions within the prompt string
        # start_index = random.randint(0, len(prompt) - 1 -1 )
        # end_index = random.randint(start_index , len(prompt)-1)
        #start_token = random.randint(0, len(prompt) - 1 -1 )
        try:
            end_index_token = random.randint( min_n-1,len(line_tokens)-1)
            print("index_token:",end_index_token)
        except:
            continue
        # Create a new string from the selected portion
        new_line_tokens = line_tokens[:end_index_token]
        #new_string = prompt[start_index:end_index]
        #print(f"new string: '{new_string}'")
        print(f"new_line_tokens: '{new_line_tokens}'")

        # Tokenize the new string
        #tokenizer = torch.hub.load("huggingface/transformers", "tokenizer", "gpt2")  # Use a suitable tokenizer
        #tokens = tokenizer(new_string, return_tensors="pt",).input_ids
        #print(f"len tokens: {len(tokens[0])}")

        # Check if the number of tokens meets the condition (n or less)
        if len(new_line_tokens) <= max_n and len(new_line_tokens) >= min_n:
            meets_criteria = True
            len_tokens += len(new_line_tokens) # for statistics
            # Save the new string if it meets the condition
            new_prompt = tokenizer.decode(new_line_tokens, skip_special_tokens=True)
            print(f"-- original: {selected_line}")
            print(f"-- then: {new_prompt}")
            new_json["new_prompt"] = new_prompt
            new_json_object = json.dumps(new_json)
            print("typee: ",type(new_json_object))
            with open("RandomHumanEval.jsonl", "a") as f:
                json.dump(new_json, f)
                f.write('\n')
                #f.write(f'\n{str(new_json)}')
        else:
            print("New string has too many or too few tokens (len:", len(new_line_tokens), "; Max:", max_n, "; Min:", min_n, ")")

    i+=1

print(f"Average of tokens len: {len_tokens/len(prompts_by_lines)}")