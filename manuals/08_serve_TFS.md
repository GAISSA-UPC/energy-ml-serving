# How to deploy trained models through TensorFlow Serving
1. Verify you have installed tensorflow
1. Transform the model into SavedModel format
   - experiments/tf_serving/save_models.py
      - model auto mapping from HF to TF: https://github.com/huggingface/transformers/blob/main/src/transformers/models/auto/modeling_auto.py
2. Follow instructions in https://www.tensorflow.org/tfx/serving/setup  to install TensorFlow Serving

- In windows we need to install it with docker
echo "deb [arch=amd64] http://storage.googleapis.com/tensorflow-serving-apt stable tensorflow-model-server tensorflow-model-server-universal" |  tee /etc/apt/sources.list.d/tensorflow-serving.list && \
curl https://storage.googleapis.com/tensorflow-serving-apt/tensorflow-serving.release.pub.gpg |  apt-key add -

3. Serve the model
   1. https://www.tensorflow.org/tfx/serving/serving_basic
   2. check inspect the serving signature of the SavedModel
      saved_model_cli show --dir models/tf/codet5-base/saved_model/1 --tag_set serve --signature_def serving_default

The given SavedModel SignatureDef contains the following input(s):
  inputs['attention_mask'] tensor_info:
      dtype: DT_INT32
      shape: (-1, -1)
      name: serving_default_attention_mask:0
  inputs['decoder_attention_mask'] tensor_info:
      dtype: DT_INT32
      shape: (-1, -1)
      name: serving_default_decoder_attention_mask:0
  inputs['decoder_input_ids'] tensor_info:
      dtype: DT_INT32
      shape: (-1, -1)
      name: serving_default_decoder_input_ids:0
  inputs['input_ids'] tensor_info:
      dtype: DT_INT32
      shape: (-1, -1)
      name: serving_default_input_ids:0
The given SavedModel SignatureDef contains the following output(s):
  outputs['encoder_last_hidden_state'] tensor_info:
      dtype: DT_FLOAT
      shape: (-1, -1, 768)
      name: StatefulPartitionedCall_1:0
  outputs['logits'] tensor_info:
      dtype: DT_FLOAT
      shape: (-1, -1, 32100)
      name: StatefulPartitionedCall_1:1
  outputs['past_key_values'] tensor_info:
      dtype: DT_FLOAT
      shape: (11, 4, -1, 12, -1, 64)
      name: StatefulPartitionedCall_1:2
Method name is: tensorflow/serving/predict

   3. https://huggingface.co/blog/tf-serving-vision
   https://colab.research.google.com/github/huggingface/blog/blob/main/notebooks/10_tf_serving.ipynb
   1. ```shell
       nohup tensorflow_model_server   --rest_api_port=8501   --model_name=bert   --model_base_path=/home/fjdur/cloud-api/models/tf/bert/saved_model  2>&1 | tee server.log
     ```
     curl http://localhost:8501/v1/models/bert
4. Test  
   1. Use tools like cURL, Postman, or Python's requests library to send POST requests to the server's REST API endpoint (http://<server_address>:8501/v1/models/my_model:predict).
   2. Prepare the request payload with the input data in the expected format defined by your model.
   3. Parse the response to obtain the model's predictions.
5. How to do inference:

https://towardsdatascience.com/use-pre-trained-huggingface-models-in-tensorflow-serving-d2761f7e69f6
https://huggingface.co/blog/tf-serving-vision
https://github.com/dimitreOliveira/hf_tf_serving_examples
https://colab.research.google.com/github/huggingface/blog/blob/main/notebooks/10_tf_serving.ipynb


## Errors

"error": "Tensor name: past_key_values has inconsistent batch size: 11 expecting: 1"