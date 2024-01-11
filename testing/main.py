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
import argparse

#from inference_requests import inference_sagemaker, inference_fastapi, inference_torchserve, endpoints
from inference_requests import  inference_fastapi,  endpoints

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


# Input
DATASET_PATH = "testing/inputs.txt"
# Output 
RESULTS_PATH = "results"


def run_experiment(model, serving_infrastructure, dataset, reps):
    """_summary_ Run experiments using dataset for an specific model and serving infrastructure

    Args:
        model (_type_): _description_
        serving_infrastructure (_type_): _description_
        dataset (_type_): _description_
    """
    print(f"+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n")
    
    print(f"Running experiment infrastructure -> {serving_infrastructure}, model -> {model}, dataset -> {dataset}, reps -> {reps}\n")

    # Initialize the server
    print(f"------------------------------\n")
    #set_up_infrastructure(model, serving_infrastructure)
    
    # Run inference using dataset through API POST requests
    for i in range(reps):
        print(f"---------------- Rep {i+1} out of {reps}: \n")
        make_inferences(model, serving_infrastructure, dataset)
    
    # Stop server, delete resources...
    # mainly for cloud services, where you have to delete resources
    print(f"------------------------------\n")
    #finish_infrastructure(model, serving_infrastructure)


def make_inferences(model, serving_infrastructure, dataset):
    print(f"Running POST requests infrastructure -> {serving_infrastructure}, model -> {model}, dataset -> {dataset}\n")

    with open(dataset) as my_file:
        examples = my_file.read().splitlines()
    print(f'Dataset: {examples}')
    
    if serving_infrastructure == 'torch':
        print("---")
        inference_fastapi(model,serving_infrastructure,examples)
    elif serving_infrastructure == 'onnx':
        print("---")
        inference_fastapi(model,serving_infrastructure,examples)
    elif serving_infrastructure == 'ov':
        print("---")
        inference_fastapi(model,serving_infrastructure,examples)
    elif serving_infrastructure == 'torchscript':
        print("---")
        inference_fastapi(model,serving_infrastructure,examples)
    elif serving_infrastructure == 'torchserve':
        print("---")
        inference_torchserve(model,serving_infrastructure,examples)
    elif serving_infrastructure == 'sagemaker':
        print("---")
        inference_sagemaker(model,serving_infrastructure,examples)
    else:
        print("Error: Infrastructure is wrong")


# def finish_infrastructure(model, serving_infrastructure):
#     print(f"Finishing infrastructure -> {serving_infrastructure} model -> {model}\n")


if __name__ == "__main__":

    serving_infrastructures = ["torch","onnx", "ov", "torchscript"]
    serving_infrastructure = infrastructure

    print(models)

    if models is None :
        #models_list = [ 'codet5-base', 'codet5p-220',  'codeparrot-small',]#'gpt-neo-125m',
        #models_list = [ 'codet5-base', 'codet5p-220', 'gpt-neo-125m', 'codeparrot-small', 'pythia-410m'] 
        models_list = [ 'codet5-base', 'codet5p-220', 'codeparrot-small', 'pythia-410m']  #'gpt-neo-125m',
        models_list = [ 'pythia-410m']  #'gpt-neo-125m',

        #models_list = [ 'codet5-base']  #'gpt-neo-125m',
         
    else:
        if "," in models:
            models_list = models.split(",")
        else:
            models_list = [models,]
        for model in models_list:
            print(model)
            assert model in endpoints.keys()


    dataset = DATASET_PATH


    print(f"+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n")
    print(f"INFRASTRUCTURE -> {serving_infrastructure}\n")
    # Be sure the serving infrastructure is set up
    for model in models_list:
        #print(model, serving_infrastructure, dataset)
        run_experiment(model, serving_infrastructure, dataset, reps)
    