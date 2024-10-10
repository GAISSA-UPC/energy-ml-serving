"""_summary_

"""
import json
import random

# #input_ids = tokenizer(text, return_tensors="pt",).input_ids
# with open('./RandomHumanEval.jsonl', 'r') as json_file:
#     json_list = list(json_file)

# problems = []

# for json_str in json_list:
#     result = json.loads(json_str)
#     print(f"result: {result}")
#     print(isinstance(result, dict))
#     problems.append(result)

# #list of prompts, each list is a list of all the lines
# prompts_by_lines = []
# for problem in problems:
#     print(problem['task_id'],"--------------------")
#     prompt = problem['prompt']
#     lines = prompt.split('\n')
#     #print(lines)
#     prompts_by_lines.append(lines)
import json

# Read JSONL file and save to TXT file
def jsonl_to_txt(jsonl_filename, txt_filename):
    with open(jsonl_filename, 'r') as jsonl_file, open(txt_filename, 'w') as txt_file:
        for line in jsonl_file:
            # Parse JSON from each line
            json_data = json.loads(line)
            
            # Convert JSON data to string (you can customize this based on your needs)
            txt_data = str(json_data["new_prompt"])
            
            # Write to TXT file
            txt_file.write(txt_data + '\n')

# Example usage
jsonl_filename = 'RandomHumanEval.jsonl'
txt_filename = 'RandomHumanEval.txt'
jsonl_to_txt(jsonl_filename, txt_filename)


