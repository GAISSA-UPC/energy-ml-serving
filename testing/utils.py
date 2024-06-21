"""_summary_

Module with configurations to run inferences.

"""
import os
import argparse
import time
import builtins
import requests

TEST_SCRIPTS_FLOW = False # Default is False, If true, it will not do inferences. 
TESTING = False # default is False, if true: do not add time between reps and WARM_UP=false
WARM_UP = False # default is True

# wait time between each repetition
COOLDOWN_REP = 0 # 5*60=300 seconds
# for each line in dataset, wait time between each inference
WAIT_BETWEEN_INFERENCE = 0 # 60 sec

# Paths
DATASET_PATH = "testing/inputs.txt" # [CHANGE]
DATASET_WARM_UP = "testing/inputs_warm_up.txt"
RESULTS_DIR = "results/"

MODELS = [ 'codet5-base', 'codeparrot-small', 'pythia-410m', 'codet5p-220']  #'gpt-neo-125m', 'codet5p-220'
#MODELS = [ 'codet5-base','codet5p-220']  #'gpt-neo-125m', 'codet5p-220'
#MODELS=['codeparrot-small', 'pythia-410m',]
#MODELS=['slm',]

CHECK_URL = 'http://localhost:8000/'
# FastAPI endpoints
endpoints = {
  "codet5-base" : "/huggingface_models/codet5-base",
  "codet5p-220" : "/huggingface_models/codet5p-220",
  "codegen-350-mono" : "/huggingface_models/codegen-350-mono",
  "gpt-neo-125m" : "/huggingface_models/gpt-neo-125m",
  "codeparrot-small" : "/huggingface_models/codeparrot-small",
  "pythia-410m" : "/huggingface_models/pythia-410m",
  "tinyllama":"/huggingface_models/tinyllama",
  "phi2":"/huggingface_models/phi2",
  "pythia1-4b":"/huggingface_models/pythia1-4b",
  "slm":"/huggingface_models/slm",
}
