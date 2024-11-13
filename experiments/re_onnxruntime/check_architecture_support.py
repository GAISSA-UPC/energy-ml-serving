"""
script that performs the following tasks:

    Retrieves the model_type for a list of model IDs.
    Checks if each model_type is supported for ONNX export.
    Displays the results in a readable format.
    
    Retrieve Model Information:
        The script uses HfApi.list_models to fetch model details, including the model_type from the configuration.

    Check ONNX Support:
        It uses TasksManager.get_supported_tasks_for_model_type to determine if the model's model_type is supported for ONNX export.

    Error Handling:
        Exceptions are handled to avoid breaking the script if a model lacks a model_type or ONNX support is not found.

    Iterate Over Models:
        The script processes multiple models and provides clear output for each.

Usage

    Replace the models_names list with your desired model IDs.
    Run the script in a Python environment with huggingface_hub and optimum installed.
"""
from huggingface_hub import HfApi
from optimum.exporters import TasksManager

# Initialize the API
api = HfApi()

# Define the list of model names
models_names = [
    'google/gemma-2-2b', 'bigscience/bloomz-560m', 'bigscience/bloomz-1b1', 'bigcode/starcoder2-3b',
    'smallcloudai/Refact-1_6B-fim', 'google/recurrentgemma-2b', 'bigcode/tiny_starcoder_py',
    'google/codegemma-2b', 'bigcode/starcoderbase-1b', 'ibm/PowerLM-3b', 'ibm/PowerMoE-3b',
    'smallcloudai/Refact-1_6B-fim', 'ibm-granite/granite-3.0-2b-base', 'TechxGenus/starcoder2-15b-GPTQ',
    'TheBloke/starcoder-GPTQ', 'rtlabs/StableCode-3B'
]

# Iterate over the model names and check their support
for model_name in models_names:
    print(f"Checking model: {model_name}")
    try:
        # Fetch the full model info, including config
        models = api.list_models(
            model_name=model_name, full=True, fetch_config=True, task="text-generation",
            pipeline_tag="text-generation", limit=1
        )
        
        for model in models:
            print("-------------")
            print(f"Model ID: {model.id}")
            
            # Get the model_type from the config
            model_type = model.config.get('model_type', None)
            
            if model_type:
                print(f"Model Type: {model_type}")
                
                # Check if the model_type is supported for ONNX export
                try:
                    supported_architectures = TasksManager.get_supported_tasks_for_model_type(model_type, 'onnx')
                    if supported_architectures:
                        print(f"Supported for ONNX Export. Tasks: {list(supported_architectures.keys())}")
                    else:
                        print("Not supported for ONNX export.")
                except Exception as e:
                    print(f"Error checking support for ONNX: {e}")
            else:
                print("Model type not found in the config.")
    except Exception as e:
        print(f"Error fetching model info for {model_name}: {e}")
    print("\n")
