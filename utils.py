"""_summary_

This module contains constants with relative paths and some general values.

"""

RESULTS_DIR = 'results/'
REPORTS_DIR = 'reports/'

# Define the endpoint URL
# endpoints = {
#   "bert" : "/huggingface_models/bert",
#   "t5" : "/huggingface_models/t5",
#   "codegen" : "/huggingface_models/CodeGen",
#   "pythia" : "/huggingface_models/Pythia_70m",
#   "codet5p" : "/huggingface_models/Codet5p_220m",
#   "cnn" : "/h5_models/cnn_fashion"
# }

endpoints = {
  "codet5-base" : "/huggingface_models/codet5-base",
  "codet5p-220m" : "/huggingface_models/codet5p-220",
  "gpt-neo-125m" : "/huggingface_models/gpt-neo-125m",
  "codeparrot-small" : "/huggingface_models/codeparrot-small",
  "pythia-410m" : "/huggingface_models/pythia-410m",
}


models = endpoints.keys()