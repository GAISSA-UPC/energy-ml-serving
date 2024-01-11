
# SageMaker

It is needed to use SageMaker instances on AWS

- To build, train and deploy ML models
- connect to Amazon services
- support for all leading ML frameworks

- Free tier
  - Free Tier usage per month for the first 2 months

- MLaaS platform, Image Classification with Machine Learning as a Service:-A comparison between Azure, SageMaker, and Vertex AI
  - "Apart from being useful for data storage, training, and validating a machine learning
model, MLaaS can also be used to deploy the model as a web service endpoint
for production use"
- <https://research.spec.org/icpe_proceedings/2021/companion/p49.pdf>

  <https://github.com/huggingface/notebooks/blob/main/sagemaker/11_deploy_model_from_hf_hub/deploy_transformer_model_from_hf_hub.ipynb>

https://github.com/huggingface/notebooks/blob/main/sagemaker/13_deploy_and_autoscaling_transformers/sagemaker-notebook.ipynb

1. Install
  <https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html>

  python3 -m pip install  "sagemaker>=2.48.0" --upgrade
2. Create User
  Save access keys
3. Configure access keys in your local environment
  ```/usr/local/bin/aws configure```
   - You should have your keys in home/.aws/credentials and the region info in home/.aws/config
   - config file example:
      ```
      [default]
      region = eu-west-3
      ```
   - credentials example:
      ```
      [default]
      aws_access_key_id = ABC
      aws_secret_access_key = ABC
      ``` 
1. Create role
  <https://docs.aws.amazon.com/sagemaker/latest/dg/sagemaker-roles.html>

  copy ARN arn:aws:iam::150660304444:role/sagamaker_role
5. Create group and add IAM permissions
  check_role.py

## Deploy model in sagemaker
https://github.com/huggingface/notebooks/blob/main/sagemaker/11_deploy_model_from_hf_hub/deploy_transformer_model_from_hf_hub.ipynb
https://huggingface.co/docs/sagemaker/inference#deploy-a-model-from-the-hub
- Notebooks: https://github.com/huggingface/notebooks/tree/main/sagemaker


## models, endpoints

An endpoint configuration has a model attached to it,
We can reuse the endpoint of the notebook and check


## Workshop https://www.youtube.com/watch?v=whwlIEITXoY
- HF in sagemaker, supported by HF
- https://github.com/philschmid/huggingface-sagemaker-workshop-series/blob/main/workshop_2_going_production/lab3_autoscaling.ipynb

--user ABCD:1234 --aws-sigv4 "aws:amz:us-east-1:execute-api"
endpoint_url = 'https://runtime.sagemaker.eu-west-3.amazonaws.com/endpoints/huggingface-pytorch-inference-2023-09-14-08-26-53-037/invocations'


curl -X POST 'https://runtime.sagemaker.eu-west-3.amazonaws.com/endpoints/huggingface-pytorch-inference-2023-09-14-08-26-53-037/invocations' -d '{"inputs":"def hello_world():"}' --user "AKIAYHK3NOPPP44HKHH7:kR5HLdQlLBXfR4FO4rsBvLnoBVyuBV29MOUs7llj" --aws-sigv4 "aws:amz:eu-west-3:execute-api"

$ curl -X POST "<ENDPOINT>" -d <data> --user <AWS_ACCESS_KEY>:<AWS_SECRET_KEY> --aws-sigv4 "aws:amz:<REGION>:<SERVICE>"
$ curl -X POST "https://1234WXYZ.execute-api.us-east-1.amazonaws.com/stage/lambda_proxy" -d '{"x":"y"}' --user ABCD:1234 --aws-sigv4 "aws:amz:us-east-1:execute-api"