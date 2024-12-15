# Energy consumption of code small language models serving with runtime engines and execution providers


[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.14484001.svg)](https://doi.org/10.5281/zenodo.14484001)
[![arXiv](https://img.shields.io/badge/arXiv-0000.00000-b31b1b.svg)](https://arxiv.org/abs/0000.00000)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Summary

### Contributions
- Actionable guidelines for practitioners
- Measuring the impact of deep learning serving configurations on energy and performance
- An analysis of deep learning serving configurations

### DL serving configurations
A duplet of a runtime engine and an execution provider: <[Runtime engine], [Execution provider]>
- Runtime engines
  - Default Torch (TORCH)
  - ONNX Runtime engine (ONNX)
  - OpenVINO runtime (OV)
  - Torch JIT (JIT)
- Execution providers
  - CPU Execution Provider (CPU)
  - CUDA Execution Provider (CUDA)

## Repository Structure

The repository is structured as follows:

<pre/>
- app
  | API, schemas
- dataset
  | input dataset generation
- experiments
  | Notebooks and scripts to process profilers datasets
- manuals
  | Self-contained manuals related to the serving infrastructure
- model_selection
  | This folder contains models selection scripts and metadata
- scripts
  | Environment scripts and bash scripts for automated experiments
- testing
  | Scripts to send request to server
- requirements.txt: The dependencies of our implementation
- runall_update.sh: Bash script to start server and run experiments
- code_slm_selection.csv: Selection of used code SLM
</pre>


## Replication package

### 1. Data management

- Needed: HumanEval dataset
- Output: New input dataset
- files: 
  - ```dataset/*```

### 2. Modelling

- Needed: Selection criteria
- Output: Selected models
- files:
  - ```code_slm_selection.csv```

### 3. Development

- Needed: Development of serving infrastructure, selected models
- Output: Serving infrastructure
- files:
  - ```app/```

### 4. Operation

- Needed: Deployed serving infrastructure
- Output: results (profilers datasets)
- files:
  - ```testing/```

1. Edit experiment parameters (time,files,...):
   1. server settings
      - ```app/models_code_load.py```: Model classes
        - MAX_LENGTH tokens
   2. experiment settings
      - ```testing/utils.py```: experiment settings, python script
        - input dataset
      - ```repeat.sh```: repeat n experiments runs or just execute runall
        - ```runall_update.sh```: experiment settings, bash script
          - run server
          - run experiments for each runtime engine
2. Run server and experiments: runall.sh
  ```bash
  nohup ./repeat.sh > repeat.out 2>&1 &
  ```

  Or:
  ```bash
  nohup ./runall.sh > results/runall.out 2>&1 &
  ```
3. Obtain ```results/*```

### 5. Research output

- Needed: Profilers datasets
- Output: Research output, data analysis and, support files to answer RQs
- files:
  - ```experiments/```
  - figures
  - tables
  - statistical results

Files in `experiments/`

- `visualize_{profiler}` - Visualization of raw data obtained from profilers.
- `01_get_info_{profiler}` - Preprocessing of raw data obtained from profilers (script).
- `02_get_time_marks` - Get time marks of inferences done during experiment (script)
- `03_analysis_{execution_provider}` - Process data for analysis (notebook).
- `04_aggregation` - Aggregated data (notebook).
- `05_aggregated_plots` - Box plots (notebook).
- `06_tests` - Obtaining statistical results of used statistical tests (script).
- `07_tests_merge` - Merge test results, organized by dependent variable (notebook).
- `08_analysis` - Notebook to analyze results (notebook).
- `09_result_tables` - Creation of paper table (notebook).


## Models

- [codeparrot-small](https://huggingface.co/codeparrot/codeparrot-small)
- [tiny_starcoder](https://huggingface.co/bigcode/tiny_starcoder_py)
- [pythia-410m](https://huggingface.co/EleutherAI/pythia-410m)
- [bloomz-560m](https://huggingface.co/bigscience/bloom-560m)
- [starcoderbase-1b](https://huggingface.co/bigcode/starcoderbase-1b)
- [bloomz-1b1](https://huggingface.co/bigscience/bloomz-1b1)
- [tinyllama](https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0)
- [pythia-1.4b](https://huggingface.co/EleutherAI/pythia-1.4b)
- [codegemma-2b](https://huggingface.co/google/codegemma-2b)
- [phi2](https://huggingface.co/microsoft/phi-2)
- [stablecode-3b](https://huggingface.co/stabilityai/stablecode-instruct-alpha-3b)
- [stablecode-3b-completion](https://huggingface.co/stabilityai/stablecode-completion-alpha-3b-4k)

## Energy tracking tools
- [EnergiBridge](https://github.com/tdurieux/EnergiBridge)
- [nvidia-smi](https://developer.nvidia.com/nvidia-system-management-interface)
- [Wattmeter](https://vitriko.eu/regleta-inteligente-netio-powerbox-4kf)


## Help

### Testing
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

```
Results are saved in [results/](results/)

### Useful Guides
1. [API creation](manuals/01_create_api.md). Guide to create an API to deploy ML models.
2. [Add pretrained model](manuals/02_add_models.md). Guide to add pretrained ML models (from HuggingFace, hdf5 format, pickle format) to do inferences through an API.
3. [Deploy ML models in a cloud provider (General)](manuals/03_deploy_general.md). Guide to deploy ML models using an API in a cloud provider.
4. [See more](manuals/)

### Other repos
1. https://madewithml.com, API
2. https://github.com/se4ai2122-cs-uniba/SE4AI2021Course_FastAPI-demo, API
3. https://github.com/MLOps-essi-upc


## Citation
Please use the following BibTex entry:

```bibtex
@article{duran2024serving,
  title={Identifying architectural design decisions for achieving green ML serving},
  author={Dur{\'a}n, Francisco and Martinez, Matias and Lago, Patricia and Mart{\'\i}nez-Fern{\'a}ndez, Silverio},
  journal={arXiv preprint arXiv:},
  year={2024}
}
```
