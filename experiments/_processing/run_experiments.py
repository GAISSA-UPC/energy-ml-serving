import requests
import json
import argparse
import datetime

parser = argparse.ArgumentParser(description='Description of your script.')

# Add command line arguments
parser.add_argument('-r', '--reps', type=int, default=1, help='repetitions')
parser.add_argument('-m', '--model', type=str, default=None, help='model')
parser.add_argument('-e', '--engine', type=str, default=None, help='runtime engine')

# Parse the command line arguments
args = parser.parse_args()


# Access the values of the arguments
reps = args.reps
model = args.model
engine = args.engine


# Define the endpoint URL
models = [ 'codet5-base', 'codet5p-220', 'codegen-350M-mono', 'gpt-neo-125m', 'codeparrot-small', 'pythia-410m'] # bloom, pythia

endpoints2 = {
  "codet5-base" : "/huggingface_models/codet5-base",
  "t5" : "/huggingface_models/t5",
  "codegen" : "/huggingface_models/CodeGen",
  "pythia" : "/huggingface_models/Pythia_70m",
  "codet5p" : "/huggingface_models/Codet5p_220m",
  "cnn" : "/h5_models/cnn_fashion"
}

endpoints = {
    'codet5-base':"/huggingface_models/codet5-base/",
    "codet5p" : "/huggingface_models/codet5p-220/",
    'gpt-neo-125m':"/huggingface_models/gpt-neo-125m/",
    'codeparrot-small':"/huggingface_models/codeparrot-small/",
    'pythia-410m':"/huggingface_models/pythia-410m/",
}
#models = [ 'codet5-base', 'codet5p-220', 'codegen-350M-mono', 'gpt-neo-125m', 'codeparrot-small', 'pythia-410m'] # bloom, pythia

runtime_engines = ['none','onnx','ov','torchscript']

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

examples['codet5-base'] = ["def get_random_element(dictionary):",]
examples['gpt-neo-125m'] = ["def get_random_element(dictionary):",]
examples['codeparrot-small'] = ["def get_random_element(dictionary):",]
examples['pythia-410m'] = ["def get_random_element(dictionary):",]


models = endpoints.keys()
print(f'Models: {models}')

url = 'http://localhost:8000'

#url = url + endpoints["bert"]
# Define the request payload (data to be sent in the POST request)


def experiment_model(model:str = '', n:int = 5, engine:str=''):
    """_summary_ Do 'n' API calls to make a prediction using the 'model'

    Args:
        model (str): _description_
        n (int, optional): _description_. Defaults to 5.
    """
    #print(f'Experiment using {model} model...')
    print(f'Total reps: {n}')
    
    payload = {
        "input_text": examples[model][0]
    }
    
    for i in range(n):
        print(f'--------- {i}:')

        endpoint = url + endpoints[model] + engine
        print(f"Endpoint --> {endpoint}")
        #print(f"Engine --> {engine}")
        #print(f"Model --> {model}")
        response = requests.post(endpoint, json=payload)
        data = response.json()
        #print(data)
        data = data['data']
        print(data)        
        assert response.status_code == 200
        assert isinstance(data['prediction'], dict), 'not a dictionary'
        assert isinstance(data['prediction']['prediction'],str), 'not a string'
        assert len(data['prediction']['prediction']) > 0, 'prediction len is 0'

def experiment_all(reps : int = 1, models = models, engines = runtime_engines):
    """_summary_ Experiment with all the models.

    Args:
        reps (int, optional): _description_. Number of calls for each model. Defaults to 1.
    """
    for engine in runtime_engines:
        print('-----------------------------------------------------------------------------------')
        print(f'Experiment -> Runtime Engine -> {engine}\n')
        for model in models:
            print('------------------------------------------------------------')
            print(f'Experiment -> Model -> {model}\n')
            #print(f'Experiment using {model} model...')
            #print(f'Total reps: {reps}')
            experiment_model(model, reps, engine)
        
        
        
# model = 'bert'
#experiment_model(model,1)
ct = datetime.datetime.now()
print('\n\n\n++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
print(f'Running experiment at: {ct}')
print(f'command-line arguments: reps - {reps}, model - {model}, engine - {engine}')

if model is not None:
    models = [model]
if engine is not None:
    runtime_engines = [engine]

experiment_all(reps=reps, models = models, engines =runtime_engines)

# if model is None:
#     #experiment_all(reps=reps)
#     if engine is None:
#         experiment_all(reps=reps)
#     elif engine in runtime_engines:
#         experiment_all(reps=reps, engines = [engine])
# elif model in models:
#     #experiment_model(model, reps)
#     if engine is None:
#         experiment_model(model, reps,engines = [engine])
#     elif engine in runtime_engines:
#         experiment_all(model, reps=reps, engines = [engine])
#     if engine in runtime_engines:
