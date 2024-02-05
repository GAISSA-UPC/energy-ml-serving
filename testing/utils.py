"""_summary_

Module with configurations to run inferences.

"""
import os
import argparse
import time
import builtins

TEST_SCRIPTS_FLOW = False # If true, it will not do inferences, default is False
TESTING = False # default is False, if true: do not add time between reps and WARM_UP=false
WARM_UP = False # default is True

# time in seconds
# wait time between each repetition
COOLDOWN_REP = 10 
# for each line in dataset, wait time between each inference
WAIT_BETWEEN_INFERENCE = 0 

# Paths
DATASET_PATH = "testing/inputs.txt"
DATASET_WARM_UP = "testing/inputs_warm_up.txt"
RESULTS_DIR = "results/"

MODELS = [ 'codet5-base', 'codeparrot-small', 'pythia-410m', 'codet5p-220']  #'gpt-neo-125m', 'codet5p-220'

# FastAPI endpoints
endpoints = {
  "codet5-base" : "/huggingface_models/codet5-base",
  "codet5p-220" : "/huggingface_models/codet5p-220",
  "codegen-350-mono" : "/huggingface_models/codegen-350-mono",
  "gpt-neo-125m" : "/huggingface_models/gpt-neo-125m",
  "codeparrot-small" : "/huggingface_models/codeparrot-small",
  "pythia-410m" : "/huggingface_models/pythia-410m",
}

model_checkpoint = {'codet5-base':"Salesforce/codet5-base", 'codet5p-220':'Salesforce/codet5p-220m', 
                        'codegen-350-mono':"Salesforce/codegen-350M-mono", 'gpt-neo-125m':"EleutherAI/gpt-neo-125M",
                        'codeparrot-small':'codeparrot/codeparrot-small', 'pythia-410m':"EleutherAI/pythia-410m"} # model:checkpoint

