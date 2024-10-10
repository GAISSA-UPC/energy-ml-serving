# Energy consumption of code small language models serving with runtime engines and execution providers
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Summary
The contributions of this work are:

- Actionable guidelines for practitioners
- Measuring the impact of deep learning serving configurations on energy and performance
- An analysis of deep learning serving configurations

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
- models
  | This folder contains pretrained models
- results
  | Generated datasets, PDFs, graphics and figures to be used in reporting
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
- `01_get_info_{profiler}` - Preprocessing of raw data obtained from profilers.
- `02_get_time_marks` - Get time marks of inferences done during experiment.
- `03_analysis_{execution_provider}` - Data analysis.
- `04_test` - Obtaining statistical results of used statistical tests
- `05_test` - Merge test results, organized by dependent variable
- `06_analysis` - Notebook to analyze results
- `07_result_tables` - Creation of paper table


## Models

- [codeparrot-small](https://huggingface.co/codeparrot/codeparrot-small)
- [phi2](https://huggingface.co/codeparrot/)
- [pythia-410m](https://huggingface.co/codeparrot/)
- [pythia1-4b](https://huggingface.co/codeparrot/)
- [tinyllama](https://huggingface.co/codeparrot/)

## Energy tracking metrics
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
