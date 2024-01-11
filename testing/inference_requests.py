import json
import requests
#import boto3 # for aws
#from codecarbon import track_emissions

RESULTS_DIR = 'results/'

# FastAPI endpoints
endpoints = {
  "codet5-base" : "/huggingface_models/codet5-base",
  "codet5p-220" : "/huggingface_models/codet5p-220",
  "codegen-350-mono" : "/huggingface_models/codegen-350-mono",
  "gpt-neo-125m" : "/huggingface_models/gpt-neo-125m",
  "codeparrot-small" : "/huggingface_models/codeparrot-small",
  "pythia-410m" : "/huggingface_models/pythia-410m",
}

model_checkpoint = {'codet5-base':"Salesforce/codet5-base", 'codet5p-220':'Salesforce/codet5p-220m', 
                        'codegen-350-mono':"Salesforce/codegen-350M-mono", 'gpt-neo-125m':"EleutherAI/gpt-neo-125M",
                        'codeparrot-small':'codeparrot/codeparrot-small', 'pythia-410m':"EleutherAI/pythia-410m"} # model:checkpoint


    
def inference_fastapi(model, serving_infrastructure, dataset):
    """_summary_ Run inference using dataset

    Args:
        model (_type_): _description_
        serving_infrastructure (_type_): _description_
        dataset (_type_): _description_

    Returns:
        _type_: _description_
    """
    print(f"Running POST requests infrastructure -> {serving_infrastructure}, model -> {model}, dataset -> {dataset}\n")

    url = 'http://localhost:8000'
    

    #@track_emissions(project_name = f"{model}_{serving_infrastructure}", output_file = RESULTS_DIR + f"emissions_{model}.csv")
    def infer(engine):
        # Make the POST request using the SageMaker runtime client

        print(f"Endpoint: {url}{endpoints[model]}/{engine}")
        response = requests.post(f"{url}{endpoints[model]}/{engine}"  , json=payload)
        
        return response

    for line in dataset:
        print("___________________________________")
        # Define the data you want to send as a JSON payload
        payload = {
            "input_text": line,
            # Add more data as needed
        }

        # Convert the payload to JSON
        #payload_json = json.dumps(payload)

        # Define the headers (usually "Content-Type: application/json")
        headers = {"Content-Type": "application/json"}

        response = infer(serving_infrastructure)

        data = response.json()

        print(data)        
        
        # Check the response
        if data["status-code"] == 200:
            #result = json.loads(response["Body"].read().decode("utf-8"))
            print("Inference successful!")
            print(data['data']['prediction'])
            # Process the result as needed
        else:
            print(
                f"Request failed with status code {data['status-code']}: {data}"
            )

    print (f"\n\nCodeCarbon Results: {RESULTS_DIR}emissions_{model}.csv")


def inference_torchserve(model, serving_infrastructure, dataset):
    """_summary_ Run inference using dataset

    Args:
        model (_type_): _description_
        serving_infrastructure (_type_): _description_
        dataset (_type_): _description_

    Returns:
        _type_: _description_
    """
    print(f"Running POST requests infrastructure -> {serving_infrastructure}, model -> {model}, dataset -> {dataset}\n")
    model = 'codeparrot'
    url = f'http://127.0.0.1:8080/predictions/{model}'
    

    #@track_emissions(project_name = f"{model}_{serving_infrastructure}", output_file = RESULTS_DIR + f"emissions_{model}.csv")
    def infer(engine):
        # Make the POST request using the SageMaker runtime client

        print(f"Endpoint: {url}")
        response = requests.post(f"{url}"  , data=line)
        
        return response

    for line in dataset:
        print("___________________________________")
        print(f'input: {line}')
        # Define the data you want to send as a JSON payload
        payload = {
            line,
            # Add more data as needed
        }

        # Define the headers (usually "Content-Type: application/json")
        headers = {"Content-Type": "application/json"}

        response = infer(serving_infrastructure)
        #print(response)
        #print(response.text)

        #print(type(response))
        #print(response.status_code)
        
        #output = response.json()
        #print(output)
        
        # Check the response
        if response.status_code == 200:
            #result = json.loads(response["Body"].read().decode("utf-8"))
            print("Inference successful!")
            print(f'Output: {response.text}')
            # Process the result as needed
        else:
            print(
                f"Request failed with status code {response.status_code}: {response.text}"
            )

    #print (f"\n\nCodeCarbon Results: {RESULTS_DIR}emissions_{model}.csv")


def inference_sagemaker(model, serving_infrastructure, dataset, endpoint_name = None):
        
    print(f"Running POST requests infrastructure -> {serving_infrastructure}, model -> {model}, dataset -> {dataset}\n")

    
    # Define your AWS region
    aws_region = "eu-west-3"  # Replace with your AWS region

    # Initialize the SageMaker runtime client
    # We initialize the boto3 SageMaker runtime client, which automatically handles authentication using your AWS credentials.
    sagemaker_runtime = boto3.client("sagemaker-runtime", region_name=aws_region)

    
    # Define the SageMaker endpoint name
    if endpoint_name is None:
        endpoint_name = model

    # Just approximate by using GCP and same country and region than the sagemaker endpoint
    # https://github.com/mlco2/codecarbon/blob/master/codecarbon/data/cloud/impact.csv
    @track_emissions(project_name = f"{model}_sm", output_file = RESULTS_DIR + f"emissions_{model}.csv", 
                     offline=True, country_iso_code = 'FRA',cloud_region='europe-west9', cloud_provider='gcp')
    def infer():

        # Make the POST request using the SageMaker runtime client
        response = sagemaker_runtime.invoke_endpoint(
            EndpointName=endpoint_name,
            Body=payload_json,
            ContentType="application/json",
        )
        return response

    for line in dataset:
        # Define the data you want to send as a JSON payload
        payload = {
            "inputs": line,
            # Add more data as needed
        }

        # Convert the payload to JSON
        payload_json = json.dumps(payload)

        # Define the headers (usually "Content-Type: application/json")
        headers = {"Content-Type": "application/json"}

        response = infer()

        # Check the response
        if response["ResponseMetadata"]["HTTPStatusCode"] == 200:
            result = json.loads(response["Body"].read().decode("utf-8"))
            print("Inference successful!")
            print(result)
            # Process the result as needed
        else:
            print(
                f"Request failed with status code {response['ResponseMetadata']['HTTPStatusCode']}: {response['Body'].read().decode('utf-8')}"
            )

    print (f"\n\nCodeCarbon Results: {RESULTS_DIR}emissions_{model}.csv")