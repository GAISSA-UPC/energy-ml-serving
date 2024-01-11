
# How to deploy a ML model in Vertex AI?


Online predictions - synchronous requests made to a model endpoint
Batch predictions - asynchronous requests, you request a batchPredictionsJob directly from the model resource without needing to deploy the model to an endpoint. 

- Upload model
  - Create bucket to save model, model storage service of GCP
    -  models_bucket_upc 

- From https://huggingface.co/blog/deploy-vertex-ai:
  - Upload (SavedModel) model to Vertex AI Model Registry
  - Deploy model from registry to the Vertex AI endpoint
  - Requests to endpoint

https://www.linkedin.com/pulse/how-measure-carbon-footprint-python-vertex-ai-joseph-v-thomas/

```code```

  
## Pricing

https://cloud.google.com/vertex-ai/pricing

Model Registry 
- There is no cost associated with having your models in the Model Registry. Cost is only incurred when you deploy the model to an endpoint or perform a batch prediction on the model. This cost is determined by the type of model you are deploying.

You pay for these actions
- Training the model
- Deploying the model to an endpoint
- Using the model to make predictions

You pay for each model deployed to an endpoint, even if no prediction is made. You must undeploy your model to stop incurring further charges. Models that are not deployed or have failed to deploy are not charged.



## Problems

- WHen trying to import a model from the model registry
  - Error:
    - Estado: 400 Código de error: 9
  - Why:
  - Solution: