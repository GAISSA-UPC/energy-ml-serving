# How to deploy trained models through TorchServe

- TorchServe repository and documentation: https://github.com/pytorch/serve/tree/master
https://github.com/pytorch/serve/blob/master/docs/server.md
- [Example for ONNX integration](https://github.com/pytorch/serve/blob/b260776b55283ab6080f00fec62bfcddf766e97d/test/pytest/test_onnx.py)

- bert using torch 
https://www.slideshare.net/nidhinpattaniyil/serving-bert-models-in-production-with-torchserve
https://github.com/npatta01/pytorch-serving-workshop/blob/main/serving/handler.py

1. Verify requirements: requirements_onnx.txt
2. Install TorchServe

```
git clone https://github.com/pytorch/serve.git
- requirements_onnx
# Install dependencies
# cuda is optional
python ./ts_scripts/install_dependencies.py --cuda=cu102

# Latest release
pip install torchserve torch-model-archiver torch-workflow-archiver

# Nightly build
pip install torchserve-nightly torch-model-archiver-nightly torch-workflow-archiver-nightly
```
Optional: Run the ONNX example to check you have installed it correctly

3. experiments/torch_converter.py  to save as Torch molddde

2. From ONNX model

- https://github.com/pytorch/serve/blob/master/test/pytest/test_onnx.py

   1. Define TorchServe handler 


   2. Convert ONNX model into TorchServe MAR file

    - Package model
    - Package all the artifacts of the loadable models

    See also: https://medium.com/analytics-vidhya/deploy-huggingface-s-bert-to-production-with-pytorch-serve-27b068026d18

    Converting the trained checkpoint to TorchServe MAR file

    TorchServe uses a format called MAR (Model Archive) to package models and version them inside its model store. To make it accessible from TorchServe, we need to convert our trained BERT checkpoint to this format and attach our handler above.

    ONNX

    torch-model-archiver -f --model-name onnx --version 1.0 --serialized-file models/onnx/onnx_bert/model.onnx --export-path torchserve_m/ --handler torchserve_handlers/bert_handler.py

    torch-model-archiver --model-name "bert" --version 1.0 --serialized-file models/onnx/onnx_bert/model.onnx --extra-files "./models/onnx/onnx_bert/config.json,./models/onnx/onnx_bert/vocab.txt" --handler torchserve_handlers/bert_handler.py --export-path torchserve_m/

    codet5
        torch-model-archiver -f --model-name codet5 --version 1.0 --serialized-file models/torch/codet5-base/pytorch_model.bin --export-path models/torch_m/ --handler experiments/torch_serve/t5_handler_codecarbon.py  --extra-files "models/torch/codet5-base/config.json,models/torch/codet5-base/generation_config.json"

        torch-model-archiver -f --model-name codet5 --version 1.0 --serialized-file models/torch/codet5-base/pytorch_model.bin --export-path models/torch_m_02/ --handler experiments/torch_serve/t5_handler_codecarbon_2.py  --extra-files "models/torch/codet5-base/config.json,models/torch/codet5-base/generation_config.json"
    codeparrot
        torch-model-archiver -f --model-name codeparrot --version 1.0 --serialized-file models/torch/codeparrot-small/pytorch_model.bin --export-path models/torch_m/ --handler experiments/torch_serve/causal_lm_h.py  --extra-files "models/torch/codeparrot-small/config.json,models/torch/codeparrot-small/generation_config.json"

        torch-model-archiver -f --model-name codeparrot --version 1.0 --serialized-file models/torch/codeparrot-small/pytorch_model.bin --export-path models/torch_m_02/ --handler experiments/torch_serve/causal_lm_2.py  --extra-files "models/torch/codeparrot-small/config.json,models/torch/codeparrot-small/generation_config.json"
    codegen
        torch-model-archiver -f --model-name codegen --version 1.0 --serialized-file models/torch/codegen-350-mono/pytorch_model.bin --export-path models/torch_m/ --handler experiments/torch_serve/codegen_h.py  --extra-files "models/torch/codegen-350-mono/config.json,models/torch/codegen-350-mono/generation_config.json"
    codet5p
        torch-model-archiver -f --model-name codet5p --version 1.0 --serialized-file models/torch/codet5p-220/pytorch_model.bin --export-path models/torch_m/ --handler experiments/torch_serve/t5_handler.py  --extra-files "models/torch/codet5p-220/config.json,models/torch/codet5p-220/generation_config.json"

        torch-model-archiver -f --model-name codet5p --version 1.0 --serialized-file models/torch/codet5p-220/pytorch_model.bin --export-path models/torch_m_02/ --handler experiments/torch_serve/t5_handler_codecarbon_2.py  --extra-files "models/torch/codet5p-220/config.json,models/torch/codet5p-220/generation_config.json"
    gpt-neo
        torch-model-archiver -f --model-name gpt-neo --version 1.0 --serialized-file models/torch/gpt-neo-125m/pytorch_model.bin --export-path models/torch_m/ --handler experiments/torch_serve/causal_lm_h.py  --extra-files "models/torch/gpt-neo-125m/config.json,models/torch/gpt-neo-125m/generation_config.json"

        torch-model-archiver -f --model-name gpt-neo --version 1.0 --serialized-file models/torch/gpt-neo-125m/pytorch_model.bin --export-path models/torch_m_02/ --handler experiments/torch_serve/causal_lm_2.py  --extra-files "models/torch/gpt-neo-125m/config.json,models/torch/gpt-neo-125m/generation_config.json"
    pythia-410m
        torch-model-archiver -f --model-name pythia-410m --version 1.0 --serialized-file models/torch/pythia-410m/pytorch_model.bin --export-path models/torch_m/ --handler experiments/torch_serve/causal_lm_h.py  --extra-files "models/torch/pythia-410m/config.json,models/torch/pythia-410m/generation_config.json"

        torch-model-archiver -f --model-name pythia-410m --version 1.0 --serialized-file models/torch/pythia-410m/pytorch_model.bin --export-path models/torch_m_02/ --handler experiments/torch_serve/causal_lm_2.py  --extra-files "models/torch/pythia-410m/config.json,models/torch/pythia-410m/generation_config.json"

4. Start server
mkdir model_store && mv bert.mar model_store && torchserve --start --model-store model_store --models bert=bert.mar

torchserve --start --ncs --model-store torchserve_m/ --models torchserve_m/bert1.mar

nohup torchserve --start --ncs --model-store torchserve_m/ --models torchserve_m/bert1.mar  2>&1 | tee server.log
nohup torchserve --start --ncs --model-store models/torch_m/ --models models/torch_m/codet5p.mar  2>&1 | tee server.log

Check status
curl http://localhost:8080/ping
curl http://localhost:8081/models
curl http://localhost:8081/models/bert2

You should see in the logs something like "WORKER_MODEL_LOADED"
5. After serving model
- Inference API, Management API, and Metrics API, deployed by default on localhost in ports 8080, 8081, and 8082, respectively.

6. Inference
curl -X POST http://127.0.0.1:8080/predictions/bert -T example.txt
- Verify your response
- Add the string in the .txt file



## Undeerstanding concepts

### torch-model-archiver
https://towardsdatascience.com/serving-pytorch-models-with-torchserve-6b8e8cbdb632
https://medium.com/@SrGrace_/a-practical-guide-to-torchserve-197ec913bbd
package all model artifacts into a single model archive file

it takes
- A model checkpoint file (pth, serialized PyTorch state dictionary) or a model definition and a state_dict file in case of eager mode
  - Model file: model architecture, inherits from torch.nn.Module
  - serialized: .pt or .pth and state_dict in case of eager mode
- Optional assets
  - https://github.com/pytorch/serve/tree/master/model-archiver
  - dictionary/json relations between ids and labels ...

- .mar file
  - the “ready to serve” archive of the model generated with torch-model-archiver.

  Torch Model Archiver is a tool used for creating archives of trained neural net models that can be consumed for TorchServe inference.

  https://github.com/pytorch/serve/tree/master/model-archiver

### Handlers
https://github.com/pytorch/serve/blob/master/docs/default_handlers.md

- initialize(): Initialize the Model object
- preprocess(): e.g. tokenize input
- inference()
- postprocess(): return a list with same length as batchsize

### Workers


### VIdeos

### Pytorch

When inferring, put models in eval mode, 

### Workflow
https://pytorch.org/serve/workflows.html

Only following output types are supported by workflow models/functions : String, Int, List, Dict of String, int, Json serializable objects, byte array and Torch Tensors

### AutoModels

tasks: https://huggingface.co/docs/optimum/exporters/task_manager
fill-mask, ORTModelForMaskedLM,
text-generation   ORTModelForCausalLM 
encoder decoder models like t5  -   AutoModelForSeq2SeqLM
