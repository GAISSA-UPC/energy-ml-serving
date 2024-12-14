import json
import requests
import boto3

# Define your AWS region
aws_region = "eu-west-3"  # Replace with your AWS region

# Initialize the SageMaker runtime client
# We initialize the boto3 SageMaker runtime client, which automatically handles authentication using your AWS credentials.
sagemaker_runtime = boto3.client("sagemaker-runtime", region_name=aws_region)

# Define the SageMaker endpoint name
endpoint_name = 'huggingface-pytorch-inference'

# Define the data you want to send as a JSON payload
payload = {
    "inputs": "import pandas as",
    # Add more data as needed
}

# Convert the payload to JSON
payload_json = json.dumps(payload)

# Define the headers (usually "Content-Type: application/json")
headers = {"Content-Type": "application/json"}

# Make the POST request using the SageMaker runtime client
response = sagemaker_runtime.invoke_endpoint(
    EndpointName=endpoint_name,
    Body=payload_json,
    ContentType="application/json",
)

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