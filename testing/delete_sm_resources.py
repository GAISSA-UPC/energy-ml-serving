"""
This script is used to delete SageMaker resources:
- Delete SageMaker model
- Delete SageMaker endpoint

Be sure you have configured your SageMaker configuration
"""
import sagemaker

endpoint_names = [ 'codet5-base', 'codet5p-220', 'gpt-neo-125m', 'codeparrot-small',]
#models = models[:2]

for endpoint_name in endpoint_names:
    predictor=sagemaker.predictor.Predictor(
        endpoint_name=endpoint_name,
        #sagemaker_session=sagemaker.Session(),
        #serializer=sagemaker.serializers.CSVSerializer()
    )

    # delete endpoint
    try:
        predictor.delete_model()
        predictor.delete_endpoint()
        print(f'Resources for {endpoint_name} succesfully deleted!')
    except Exception as e:
        print(f'Error while deleting {endpoint_name} resources.')