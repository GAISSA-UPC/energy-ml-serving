"""


run_experiments()

analysis()

Experiment:
    ServingInfrastructure
    Model
    Dataset to make inferences

Analysis:
    File for each model and ServingInfrastructure

"""
#import gc
import random

from inference_requests import  inference_fastapi
from utils import *

script_name = os.path.basename(__file__)
def print(*args, **kwargs):
    builtins.print(f"[{script_name}] ",*args, **kwargs)

if TESTING:
    COOLDOWN_REP = 0
    WARM_UP = False
    print(f"** WARNING: TESTING mode ** ")

parser = argparse.ArgumentParser(description='Make inferences.')

# Add command line arguments
parser.add_argument('-i', '--infrastructure', type=str, default=None, help='Example: ["torch","onnx","ov","torchscript", "torchserve", "sagemaker"]')
parser.add_argument('-m', '--models', type=str, default=None, help="Example: [ 'codet5-base', 'codet5p-220', 'gpt-neo-125m', 'codeparrot-small', 'pythia-410m']")
parser.add_argument('-r', '--reps', type=int, default=1, help="Number of repetitions")


# Parse the command line arguments
args = parser.parse_args()

# Access the values of the arguments
infrastructure = args.infrastructure
models = args.models
reps = args.reps

def check_server(url, timeout=30, interval=5):
    """
    Check if the Uvicorn server is up by making a GET request to the specified URL.

    :param url: The URL to check for server availability.
    :param timeout: The request timeout in seconds.
    :param interval: The interval between checks in seconds.
    """
    while True:
        try:
            response = requests.get(url, timeout=timeout)
            if response.status_code == 200:
                print("Server is up!")
                break
        except requests.exceptions.RequestException as e:
            print(f"Server is not up yet: {e}")
        time.sleep(interval)

def warm_up(models, serving_infrastructure, dataset = DATASET_WARM_UP):
    """_summary_ Run once the inferences (using the inputs from dataset) using each model

    Args:
        models (_type_): _description_
        serving_infrastructure (_type_): _description_
        dataset (_type_): _description_
    """
    print(f"++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n")
    
    print(f"warm_up(): Start -> {serving_infrastructure}, models -> {models}, dataset -> {dataset}, reps -> 1\n")
    print(f"------------------------------\n")

    for model in models:
        make_inferences(model, serving_infrastructure, dataset) if not TEST_SCRIPTS_FLOW else print("**TEST_SCRIPTS_FLOW**")
    
    print(f"Waiting {COOLDOWN_REP} seconds to cooldown")
    time.sleep(COOLDOWN_REP)
    print(f"warm_up(): End")

def run_experiment(model, serving_infrastructure, dataset, reps):
    """_summary_ Run experiments using dataset for an specific model and serving infrastructure

    Args:
        model (_type_): _description_
        serving_infrastructure (_type_): _description_
        dataset (_type_): _description_
    """
    print(f"++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n")
    
    print(f"Running experiment infrastructure -> {serving_infrastructure}, model -> {model}, dataset -> {dataset}, reps -> {reps}\n")

    # Initialize the server
    print(f"------------------------------\n")
    #set_up_infrastructure(model, serving_infrastructure)
    
    # Run inference using dataset through API POST requests
    for i in range(reps):
        #print(f'gc: {gc.collect()}')

        print(f"---------------- Rep {i+1} out of {reps}: \n")
        
        make_inferences(model, serving_infrastructure, dataset)
        if COOLDOWN_REP > 0:
            print(f"Waiting {COOLDOWN_REP} seconds to cooldown")
            time.sleep(COOLDOWN_REP)
    # Stop server, delete resources...
    # mainly for cloud services, where you have to delete resources
    print(f"------------------------------\n")
    #finish_infrastructure(model, serving_infrastructure)


def make_inferences(model, serving_infrastructure, dataset):
    print(f"Running POST requests infrastructure -> {serving_infrastructure}, model -> {model}, dataset -> {dataset}\n")

    with open(dataset) as my_file:
        examples = my_file.read().splitlines()
    
    # Randomize the order of examples
    random.shuffle(examples)
    print(f'Dataset (randomized): {examples}')

    
    if serving_infrastructure in ['torch','onnx','torchscript','ov']:
        print("-------------------------------------------------")
        inference_fastapi(model,serving_infrastructure,examples)
    elif serving_infrastructure == 'torchserve':
        print("-------------------------------------------------")
        inference_torchserve(model,serving_infrastructure,examples)
    elif serving_infrastructure == 'sagemaker':
        print("-------------------------------------------------")
        inference_sagemaker(model,serving_infrastructure,examples)
    else:
        print("Error: Infrastructure is wrong")


def end():
    print(f"_________________________________________________________________")
    print(f"                    finished inferencing")
    print(f"_________________________________________________________________")

    if WARM_UP:
        dataset = DATASET_PATH
        with open(dataset) as my_file:
            examples = my_file.read().splitlines()

        print(f"*Note: Remember to remove the warm_up() inferences in csv files, number of lines: {len(examples)} for each runtime engine")


if __name__ == "__main__":

    check_server(CHECK_URL)

    serving_infrastructures = ["torch","onnx", "ov", "torchscript"]
    serving_infrastructure = infrastructure
    assert serving_infrastructure in serving_infrastructures, f"'{serving_infrastructure}' not registered, you must use one of these: {serving_infrastructures}"

    print(f'models in args: {models}')

    if models is None :
        models_list = MODELS
        
    else:
        if "," in models:
            models_list = models.split(",")
        else:
            models_list = [models,]
        for model in models_list:
            print(model)
            assert model in endpoints.keys()

    dataset = DATASET_PATH
    print(f"++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n")

    print(f"INFRASTRUCTURE -> {serving_infrastructure}\n")
    print(f"MODELS -> {models_list}\n")

    
    warm_up(models_list, serving_infrastructure,) if WARM_UP else print("** no warm_up() **")

    # Be sure the serving infrastructure is set up
    print(f"++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n")
    print(f"STARTING EXPERIMENT \n")
    model_counter = 0
    for model in models_list:

        #print(f'gc: {gc.collect()}')
        print(f"---------------- Model {model_counter+1} out of {len(models_list)}: \n")

        run_experiment(model, serving_infrastructure, dataset, reps) if not TEST_SCRIPTS_FLOW else print("**TEST_SCRIPTS_FLOW**")
        model_counter+=1

    #run final steps
    end()
    