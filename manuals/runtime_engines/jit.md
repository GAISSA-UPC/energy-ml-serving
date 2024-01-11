# How to run models using torch jit

ToDo
1. Export model, see script: export_models.py
2. Run API with the configuration done for the model
3. Do inference!

TorchScript is a way to create serializable and optimizable models from PyTorch code. Any TorchScript program can be saved from a Python process and loaded in a process where there is no Python dependency.

# PyTorch
https://lernapparat.de/jit-optimization-intro/
https://www.slideshare.net/perone/pytorch-under-the-hood


## Torchscript, JIT
https://github.com/pytorch/serve/blob/cf19d7c5d80a4445fd7eec4c0448613f45f910e7/examples/Huggingface_Transformers/torchscript.md

- provide example input
- use same lenght for example and in inference
- torchscript flag in config for pretrained model

https://pytorch.org/tutorials/recipes/torchscript_inference.html

## Export a HuggingFace model into torchscript

Exporting a model requires two things:

- model instantiation with the torchscript flag
- a forward pass with dummy inputs


We recommended you trace the model with a dummy input size at least as large as the largest input that will be fed to the model during inference. Padding can help fill the missing values. However, since the model is traced with a larger input size, the dimensions of the matrix will also be large, resulting in more calculations.


https://stackoverflow.com/questions/76264623/how-to-create-handler-for-huggingface-model-deployment-using-torchserve

Script to export torchscript models:

experiments/re_torch_jit/export_models.py

## Model format

https://stackoverflow.com/questions/59095824/what-is-the-difference-between-pt-pth-and-pwf-extentions-in-pytorch


## Problems
https://github.com/huggingface/transformers/issues/5647


## Errors

- torch.jit.frontend.UnsupportedNodeError: function definitions aren't supported:
  - If you use torch.jit.script, Torch.jit.script same but automatically converts python code from module.forward without example input.
  - Sometimes this is not supported