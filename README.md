# Energy consumption of ML serving infrastructure: runtime engines
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Summary
Guidelines to serve models using runtime engine (+ web framework API) as serving infrastructure

## Repository Structure

The repository is structured as follows:

<pre/>
- app
  | API, schemas
- models
  | This folder contains our trained or pretrained models
- notebooks
  | This folder contains the jupyter notebooks
- reports
  | Generated PDFs, graphics and figures to be used in reporting
- utils
  | Python functions
- manuals
  | self-contained manuals
- requirements.txt: The dependencies of our implementation
</pre>

Guide:
1. [API creation](manuals/01_create_api.md). Guide to create an API to deploy ML models.
2. [Add pretrained model](manuals/02_add_models.md). Guide to add pretrained ML models (from HuggingFace, hdf5 format, pickle format) to do inferences through an API.
3. [Deploy ML models in a cloud provider (General)](manuals/03_deploy_general.md). Guide to deploy ML models using an API in a cloud provider.

# Replication package

1. Edit files with experiment parameters (time,files,...), check CONSTANTS in scripts:
  - runall.sh
  - testing/main.py
  - models_code.py

# ML Serving

Guidelines to serve models using runtime engine (+ web framework API) as serving infrastructure

## Runtime Engine + Web framework
### ONNX Runtime
### OpenVINO
### Torch.jit
### No runtime (Using only ML framework)


## Models*
\* Initial proposed models

- BERT model
- T5
- CodeGen
- Pythia-70m
- CNN model
- Codet5p-220m

### Code Generation
- CodeGen
  - https://huggingface.co/Salesforce/codegen-350M-mono
- Pythia-70m
  - https://huggingface.co/EleutherAI/pythia-70m
- Codet5p-220m
  - https://huggingface.co/Salesforce/codet5p-220m



## Energy tracking metrics
- codecarbon
- ...

## Data collection
Dataset:
[testing/inputs.txt](testing/inputs.txt)

```bash
Run server:
uvicorn app.api_code:app  --host 0.0.0.0 --port 8000  --reload --reload-dir app

Make inferences:
python3 testing/main.py -i torch -r 5 | tee -a results/out_torch.log
python3 testing/main.py -i onnx -r 5 | tee -a results/out_onnx.log
python3 testing/main.py -i ov -r 5 | tee -a results/out_ov.log
python3 testing/main.py -i torchscript -r 5 | tee -a results/out_torchscript.log

python3 testing/main.py -i torch -r 2 | tee -a results/out_torch.log;
python3 testing/main.py -i onnx -r 2 | tee -a results/out_onnx.log;
python3 testing/main.py -i ov -r 2 | tee -a results/out_ov.log;
python3 testing/main.py -i torchscript -r 2 | tee -a results/out_torchscript.log;


```
Results are saved in [results/](results/)

- check steps in repo:
https://github.com/MLOps-essi-upc/MLOps2023Course-demo/blob/main/docs/deployment/01_deploy_general.md

Manual to configure proxy server
- check service is up
service nginx status

## References
See manuals/references


-------------------

Machine information for runtime engine experiments:
