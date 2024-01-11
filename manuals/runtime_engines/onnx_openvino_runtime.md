# How to run models using ONNXruntime

1. Export ONNX or OpenVINO models using transformers or fetch model from HF repository when doing inference
see script: converter_onnx.py,converter_ov.py
2. Run API with the configuration done for the model
3. Do inference!

## Using optimum library from Hugging Face

- We are using optimum library to convert from HF to onnx format and then use onnxruntime
- How to convert:
    - Using script converter.py, loads model in onnxruntime and save it
    - Using cli
        - optimum-cli export onnx --model local_path --task question-answering distilbert_base_uncased_squad_onnx/

## Using onnxruntime

- options to get model
- export it
  - Export with optimum https://huggingface.co/docs/optimum/exporters/onnx/overview
- huggingface repo
- onnx model zoo

- Options to run the inference from your_model.onnx
  - onnxruntime InferenceSession 
    - session.run()
    - example: 02_deploy_onnx_bert.py
    - https://github.com/microsoft/onnxruntime/blob/main/onnxruntime/python/tools/transformers/notebooks/PyTorch_Bert-Squad_OnnxRuntime_CPU.ipynb
  - optimum library
    - 

- https://onnxruntime.ai/docs/

Optimum an up and coming solution leveraging Intel NC and ONNX Runtime
https://blog.ml6.eu/bert-is-eating-your-cash-quantization-and-onnxruntime-to-save-ea6dc84dcd88
https://github.com/huggingface/optimum

Intel OpenVINO 	python -m pip install optimum[openvino,nncf]
ONNX runtime 	python -m pip install optimum[onnxruntime]

1. Install optimum

#python -m pip install onnxruntime==1.11
pip install optimum[exporters]

optimum-cli export onnx --model model-huggingface dir/

optimum-cli export onnx --model gpt2 gpt2_onnx/

optimum-cli export onnx --model local_path --task question-answering distilbert_base_uncased_squad_onnx/

tasks: https://huggingface.co/docs/optimum/exporters/task_manager
fill-mask, ORTModelForMaskedLM,
text-generation   ORTModelForCausalLM 
encoder decoder models like t5  -   AutoModelForSeq2SeqLM

Available architectures to export
https://huggingface.co/docs/optimum/exporters/onnx/overview

https://github.com/huggingface/notebooks/blob/main/examples/onnx-export.ipynb

# save all in a directory
python -m transformers.onnx --model=t5-small dir

https://github.com/huggingface/transformers/blob/main/src/transsformers/models/t5/convert_t5x_checkpoint_to_pytorch.py

if the model is encoder and decoder use ORTModelForSeq2SeqLM
https://github.com/huggingface/transformers/issues/16006

Script to export ONNX and openVINO models using optimum:

experiments/re_onnxruntime/converter_onnx.py
experiments/re_onnxruntime/converter_openvino.py

## onnx runtime
https://onnxruntime.ai/docs/get-started/with-python.html

https://github.com/PierrickPochelu/inference_framework_benchmark/tree/main

guide using onnxruntime with python
https://onnxruntime.ai/docs/get-started/with-python.html

## ONNX models

- Visualization https://netron.app/

## Steps
1. Install onnxruntime
   1. pip install onnxruntime
2. Export model to onnx
3. Load the onnx model with onnx.load
4. Create inference session using ort.InferenceSession
5. get input and output metadata
6. preprocess, tokenize
7. Run
8. postprocess

 INVALID_ARGUMENT : Invalid rank for input:

 https://github.com/huggingface/transformers/issues/16006


 supported models to export from huggingface to onnx
   https://huggingface.co/docs/optimum/exporters/onnx/overview
   bert
   bart
   t5
   codegen

According to the supported models:
   - masked lm: bert
   - seq2seq: t5
   - causal lm: codegen


- available tasks for huggingface models:

from optimum.exporters.tasks import TasksManager

distilbert_tasks = list(TasksManager.get_supported_tasks_for_model_type("distilbert", "onnx").keys())
print(distilbert_tasks)

## Running model without optimum
- Not much information and not a stable option

https://github.com/microsoft/onnxruntime/blob/main/onnxruntime/python/tools/transformers/notebooks/Inference_GPT2_with_OnnxRuntime_on_CPU.ipynb

https://github.com/microsoft/onnxruntime/blob/main/onnxruntime/python/tools/transformers/convert_generation.py

https://github.com/huggingface/notebooks/blob/main/examples/onnx-export.ipynb

## Export from hugging face to onnx
Whan trying to export onnx model:
   optimum-cli export onnx --task text-generation-with-past --model "Salesforce/codegen-350M-mono" --batch_size 1 --sequence_length 10  models/onnx/onnx_codegen_4
onnxruntime::Model::Model Unsupported model IR version: 9, max supported IR version: 8

## Last updates for project:
- We are using optimum library to convert from HF to onnx format and then use onnxruntime
- How to convert:
    - Using script converter.py, loads model in onnxruntime and save it
    - Using cli
        - optimum-cli export onnx --model local_path --task question-answering distilbert_base_uncased_squad_onnx/

## Using the onnx models and onnxruntime




## OpenVino 
A python API to the C++ openvino runtime
https://huggingface.co/docs/optimum/intel/inference

https://docs.openvino.ai/2022.3/openvino_docs_OV_UG_Model_Representation.html

steps when doing inference using optimum.openvino
  https://docs.openvino.ai/2022.3/openvino_docs_OV_UG_Integrate_OV_with_your_application.html

github:
  https://github.com/huggingface/optimum-intel/