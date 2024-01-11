"""
Script to set up ML Serving infrastructure

ToDo: One script per infrastructure to avoid loading more modules

"""
# Sagemaker
import sagemaker
import boto3
from sagemaker.huggingface import HuggingFaceModel

import os
import argparse

parser = argparse.ArgumentParser(description='Serve ML models.')

# Add command line arguments
parser.add_argument('-i', '--infrastructure', type=str, default=None, help='["torch","onnx", "torchserve", "sagemaker"]')
parser.add_argument('-m', '--models', type=str, default=None, help='["torch","onnx", "torchserve", "sagemaker"]')


# Parse the command line arguments
args = parser.parse_args()

# Access the values of the arguments
infrastructure = args.infrastructure
models = args.models


model_checkpoint = {'codet5-base':"Salesforce/codet5-base", 'codet5p-220':'Salesforce/codet5p-220m', 
                        'codegen-350-mono':"Salesforce/codegen-350M-mono", 'gpt-neo-125m':"EleutherAI/gpt-neo-125M",
                        'codeparrot-small':'codeparrot/codeparrot-small', 'pythia-410m':"EleutherAI/pythia-410m"} # model:checkpoint

def set_up_infrastructure(models, serving_infrastructure):
    print(f"Setting up infrastructure -> {serving_infrastructure} models -> {models}\n")
    try:
        if serving_infrastructure == 'torch':
            run_fastapi(serving_infrastructure)
        elif serving_infrastructure == 'onnx':
            run_fastapi(serving_infrastructure)
        elif serving_infrastructure == 'torchserve':
            set_up_torchserve(models)
        elif serving_infrastructure == 'sagemaker':
            set_up_sagemaker(models)
    except Exception as e:
        print(type(e))
        print(f'Exception: {e}')

def run_fastapi(serving_infrastructure):
    #uvicorn.run("app.api_code:app", host="0.0.0.0", port=8080, reload=True)
    os.system('uvicorn app.api_code:app  --host 0.0.0.0 --port 8000  --reload --reload-dir app')

def set_up_torchserve(models):
    model = models[0]
    # Package models before
    print(f"TorchServe not implemented yet")
    #model = models[0]
    mar_files_path = 'models/torch_m_02/'
    os.system(f'nohup torchserve --start --ncs --model-store {mar_files_path}  --models models/torch_m_02/{model}.mar  2>&1 | tee server.log')
    

def set_up_sagemaker(models):
    
    for model in models:
        print(f'Deploying {model} model on SageMaker')

        try:
            role = sagemaker.get_execution_role()
        except ValueError:
            iam = boto3.client('iam')
            role = iam.get_role(RoleName='sagemaker_role')['Role']['Arn']

        print(f"sagemaker role arn: {role}")
        #models = [ 'codet5-base', 'codet5p-220', 'codegen-350-mono', 'gpt-neo-125m', 'codeparrot-small', 'pythia-410m'] # bloom, pythia
        

        model_name = model
        checkpoint = model_checkpoint[model_name]
        print(f'checkpoint: {checkpoint}')

        # Hub Model configuration. https://huggingface.co/models
        hub = {
        'HF_MODEL_ID' : checkpoint, # model_id from hf.co/models
        'HF_TASK' : 'text-generation' # NLP task you want to use for predictions
        }

        # create Hugging Face Model Class
        huggingface_model = HuggingFaceModel(
        env=hub,
        role=role, # iam role with permissions to create an Endpoint
        transformers_version="4.26", # transformers version used
        pytorch_version="1.13", # pytorch version used
        py_version="py39", # python version of the DLC
        )

        print(huggingface_model)

        # deploy model to SageMaker Inference
        predictor = huggingface_model.deploy(
            endpoint_name = model_name,
            initial_instance_count=1,
            instance_type="ml.m5.xlarge" # xlarge free tier
        )


if __name__ == "__main__":

    #models = [ 'codet5-base', 'codet5p-220', 'codegen-350-mono', 'gpt-neo-125m', 'codeparrot-small', 'pythia-410m'] # bloom, pythia
    #models = [ 'codet5-base', 'codet5p-220', 'gpt-neo-125m', 'codeparrot-small',]
    #models = models[:2]
    serving_infrastructures = ["torch","onnx", "torchserve", "sagemaker"]
    serving_infrastructure = infrastructure

    if models is None :
        models_list = [ 'codet5-base', 'codet5p-220', 'gpt-neo-125m', 'codeparrot-small',]
    else:
        if "," in models:
            models_list = models.split(",")
        else:
            models_list = [models,]
        for model in models_list:
            print(model)
            assert model in model_checkpoint.keys()
    print(f'Models: {models_list}')
    set_up_infrastructure(models_list, serving_infrastructure)
    