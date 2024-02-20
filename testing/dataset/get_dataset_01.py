"""_summary_

Get dataset. 
Random tokens

Takes one problem of HumanEval and select two random position of the prompt string, 
create a new string of it, tokenize it, check if number of tokens is n or less 
and if it meets condition save this new string

Original features:
    task_id
    prompt
    entry_point
    canonical_solution
    test

Saved features:
    task_id:
    selection:
"""
import json
import random
import torch
from transformers import AutoTokenizer

def separe_by_lines(d):
    # separe prompt by lines, return list of lines
    print("-")

#input_ids = tokenizer(text, return_tensors="pt",).input_ids
with open('./HumanEval.jsonl', 'r') as json_file:
    json_list = list(json_file)

problems = []

for json_str in json_list:
    result = json.loads(json_str)
    print(f"result: {result}")
    print(isinstance(result, dict))
    problems.append(result)

# list of prompts, each list is a list of all the lines
# prompts_by_lines = []
# for problem in problems:
#     print(problem['task_id'],"--------------------")
#     prompt = problem['prompt']
#     lines = prompt.split('\n')
#     #print(lines)
#     prompts_by_lines


tokenizer = AutoTokenizer.from_pretrained("Salesforce/codet5-base")
min_n = 10
max_n = 15
len_tokens = 0
for problem in problems:
    meets_criteria = False

    print(f"\n\nPROBLEM -> {problem['task_id']}")

    new_json = {}

    new_json["task_id"] = "RandomTokens/"+problem["task_id"]
    prompt = problem["prompt"]

    while not meets_criteria:
        # Select two random positions within the prompt string
        start_index = random.randint(0, len(prompt) - 1 -1 )
        end_index = random.randint(start_index , len(prompt)-1)

        # Create a new string from the selected portion
        new_string = prompt[start_index:end_index]
        print(f"new string: '{new_string}'")
        # Tokenize the new string
        #tokenizer = torch.hub.load("huggingface/transformers", "tokenizer", "gpt2")  # Use a suitable tokenizer
        tokens = tokenizer(new_string, return_tensors="pt",).input_ids
        print(f"len tokens: {len(tokens[0])}")

        # Check if the number of tokens meets the condition (n or less)
        if len(tokens[0]) <= max_n and len(tokens[0]) >= min_n:
            meets_criteria = True
            len_tokens += len(tokens[0]) # for statistics
            # Save the new string if it meets the condition
            new_json["selection"] = new_string
            new_json_object = json.dumps(new_json)
            print("typee: ",type(new_json_object))
            with open("RandomHumanEval.jsonl", "a") as f:
                json.dump(new_json, f)
                f.write('\n')
                #f.write(f'\n{str(new_json)}')
        else:
            print("New string has too many or too few tokens (len:", len(tokens[0]), "; Max:", max_n, "; Min:", min_n, ")")

print(f"Average of tokens len: {len_tokens/len(problems)}")