import requests

# Define the endpoint URL
endpoints = {
  "bert" : "/huggingface_models/bert",
  "t5" : "/huggingface_models/t5",
  "codegen" : "/huggingface_models/CodeGen",
  "pythia" : "/huggingface_models/Pythia_70m",
  "codet5p" : "/huggingface_models/Codet5p_220m",
  "cnn" : "/h5_models/cnn_fashion"
}


examples = {
    "bert" : [
        "I am from [MASK].",
    ],
    "t5" : [
        "translate English to German: Hello, how are you?",
    ],
    "codegen" : [
        "def get_random_element(dictionary):",
    ],
    "pythia" : [
        "def get_random_element(dictionary):",
    ],
    "codet5p" : [
        "def get_random_element(my_dictionary):<extra_id_0>",
    ],
    "cnn" : [
        "101233",
    ],
    
}


model = 'bert'

url = 'http://localhost:8000'

url = url + endpoints["bert"]
# Define the request payload (data to be sent in the POST request)
payload = {
  "input_text": examples[model][0]
}

# Send the POST request
# response = requests.post(url, json=payload)

# # Check the response status code
# if response.status_code == 200:
#     # Request was successful
#     print('API call succeeded')
#     print('Response:', response.json())
# else:
#     # Request failed
#     print('API call failed')
#     print('Response:', response.text)

def experiment_bert():
    n = 5
    for i in range(n):
        response = requests.post(url, json=payload)
        print(response.text)
        assert response.status_code == 200
        
experiment_bert()

