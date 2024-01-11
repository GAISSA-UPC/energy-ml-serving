"""
Deploys a HF model, makes inference and delete resources

"""
import sagemaker
import boto3


RESULTS_DIR = '/home/fjdur/cloud-api/results/'


try:
    role = sagemaker.get_execution_role()
except ValueError:
    iam = boto3.client('iam')
    role = iam.get_role(RoleName='sagemaker_role')['Role']['Arn']

print(f"sagemaker role arn: {role}")

models = [ 'codet5-base', 'codet5p-220', 'codegen-350-mono', 'gpt-neo-125m', 'codeparrot-small', 'pythia-410m'] # bloom, pythia
model_checkpoint = {'codet5-base':"Salesforce/codet5-base", 'codet5p-220':'Salesforce/codet5p-220m', 
                    'codegen-350-mono':"Salesforce/codegen-350M-mono", 'gpt-neo-125m':"EleutherAI/gpt-neo-125M",
                    'codeparrot-small':'codeparrot/codeparrot-small', 'pythia-410m':"EleutherAI/pythia-410m"} # model:checkpoint

model_name = models[1]
checkpoint = model_checkpoint[model_name]
print(f'checkpoint: {checkpoint}')

# Code sagemaker.huggingface
#https://github.com/aws/sagemaker-python-sdk/blob/c3a5fb01827fdd2cdad66a2b659a2a9a574153a2/src/sagemaker/huggingface/model.py
from sagemaker.huggingface import HuggingFaceModel

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
   initial_instance_count=1,
   instance_type="ml.t3.medium"
)



# example request, you always need to define "inputs"
data = {
"inputs":  "def hello_world():"
}

# request
from codecarbon import track_emissions

#response = predictor.predict(data)

@track_emissions(project_name = "codet5p-220_sm", output_file = RESULTS_DIR + "emissions_codet5p-220.csv")
def infer(predictor, data):
    return predictor.predict(data)

response = infer(predictor, data)

print(response)

#predictor.delete_model()
#predictor.delete_endpoint()

# delete endpoint
predictor.delete_model()
predictor.delete_endpoint()
predictor.delete_predictor()