import sagemaker
import boto3

try:
    role = sagemaker.get_execution_role()
except ValueError:
    iam = boto3.client('iam')
    role = iam.get_role(RoleName='sagemaker_role')['Role']['Arn']

print(f"sagemaker role arn: {role}")