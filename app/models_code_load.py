""" models

This module defines the classes of each model used in the API.

To add a new model:
    1. Add Models_names
    2. Add ML_task
    3. Create new class:
        def class NewModel(Model):
    4. Create schema in schemas
    5. Add endpoint in api
    
ToDo:
- Add max_new_tokens parameter

Models:
    - codet5-base
    - codet5p-220
    - codegen-350-mono
    - gpt-neo-125m
    - codeparrot-small
    - pythia-410m
    
"""

# External

import subprocess
import time
import os
import builtins


import threading
import requests
import json
import csv
from datetime import datetime

from codecarbon import track_emissions
from enum import Enum

# Required to run CNN model
#import numpy as np
#import random
#from torch.nn import functional as F


from app.utils_api import *


MAX_LENGTH = 128

#os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Constants
RESULTS_DIR = 'results/'

GPU_RESULTS = 'results/gpu_results.csv'
GPU_SMI_SLEEP = 1
GPU_SMI_SEC = 0.1
GPU_SMI_MS = 100
GPU_ID = 0

models = [ 'codet5-base', 'codet5p-220', 'codegen-350-mono', 
          'gpt-neo-125m', 'codeparrot-small', 'pythia-410m',
          'tinyllama', 'slm','pythia1-4b', 'phi2',
          'bloomz-560m', 'stablecode-3b',
          'codegemma-2b', 'tiny_starcoder',
          'starcoderbase-1b','bloomz-1b1','stablecode-3b-completion'] # bloom, pythia [ADD]

runtime_engines = ['onnx','ov','torchscript'] # torch not considered in this script as 'runtime_engine'

wattmeter = False # global variable, default: False
response = None

script_name = os.path.basename(__file__)
def print(*args, **kwargs):
    builtins.print(f"[{script_name}] ",*args, **kwargs)


print(f" - Using device: {device} -")    
nvidia_smi_command = f"nvidia-smi -i {GPU_ID} --query-gpu=timestamp,gpu_name,utilization.gpu,utilization.memory,memory.total,memory.used,power.draw,power.max_limit,temperature.gpu --format=csv -lms {GPU_SMI_MS} -f {GPU_RESULTS}"

def get_wattmeter_data(task):
    filename = RESULTS_DIR + f'wattmeter_{task}.csv'
    error_log = RESULTS_DIR + f'wattmeter_error_log_{task}.txt'
    fieldnames = ['Wattmetter Timestamp', 'True timestamp', 'ID', 'Name', 'State', 'Action', 'Delay',
                  'Current', 'PowerFactor', 'Phase', 'Energy', 'ReverseEnergy', 'EnergyNR', 'ReverseEnergyNR', 'Load']
    url = "http://147.83.72.195/netio.json"
    
    while wattmeter:
        with open(filename, mode='a', newline='') as file:
            try:
                response = requests.get(url,) # timeout=100
                print(f"response:{response}")
                # Parse JSON data
                data = json.loads(response.text)

                timestamp = data['Agent']['Time']
                print(timestamp)

                # Extract data for output ID 1
                output_1_data = None
                for output in data['Outputs']:
                    if output['ID'] == 1:
                        output_1_data = output
                        break

                if output_1_data:
                    writer = csv.DictWriter(file, fieldnames=fieldnames)

                    if file.tell() == 0:
                        writer.writeheader()

                    output_1_data['Wattmetter Timestamp'] = timestamp
                    output_1_data['True timestamp'] = datetime.fromtimestamp(
                        time.time())
                    writer.writerow(output_1_data)
                else:
                    error_message = f"{datetime.now()} - Output ID 1 data not found in the JSON.\n"
                    with open(error_log, 'a') as error_file:
                        error_file.write(error_message)
            except Exception as e:
                error_message = f"{datetime.now()} - {e}\n"
                with open(error_log, 'a') as error_file:
                    error_file.write(error_message)
            finally: # lets you execute code, regardless of the result of the try-except block
                time.sleep(0.5)  # Pause execution for 0.5 seconds before next call

# not updated
class ML_task(Enum):
    MLM = 1 # Masked Language Modeling
    TRANSLATION = 2
    CV = 3 # Computer Vision
    CODE = 4

# not updated
# class models_names(Enum):
#     Codet5_base = 1
#     Codet5p_220m = 2
#     CodeGen_350m_mono = 3
#     GPT_Neo_125m = 4
#     CodeParrot_small = 5
#     Pythia_410m = 6
#     Tinyllama = 7
#     SLM = 8

class Model:
    """
    Creates a default model
    """
    def __init__(self, model_name : str = None, ml_task : ML_task = None):
        self.name = model_name
        # Masked Language Modeling - MLM
        self.ml_task = ml_task.name
    
    def predict(self, user_input : str) -> dict:
        # Do prediction
        prediction = "Not defined yet "
        response = {
            "prediction" : prediction
        }
        return response
    
    
    def infer(self, text : str, model, tokenizer) -> str:
        """_summary_ Infer function to track

        Args:
            text (str): _description_
            model (_type_): _description_
            tokenizer (_type_): _description_

        Returns:
            str: _description_
        """
        
        #input_ids = tokenizer(text, return_tensors="pt").input_ids
        #outputs = model.generate(input_ids)
        #return tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        response = None
        return response
        
# not updated, last: may
class CodeT5_BaseModel(Model):
    """
    Creates a T5 model. Inherits from Model()
    """
    def __init__(self):
        super().__init__('codet5-base', ML_task.CODE)
        
    def predict(self, user_input: str, engine = None,model=None, tokenizer=None):
        
        print(f'Runtime engine: {engine}')
        if engine not in runtime_engines:
            #tokenizer = AutoTokenizer.from_pretrained("Salesforce/codet5-base")
            #model = T5ForConditionalGeneration.from_pretrained("Salesforce/codet5-base")
            #model = AutoModelForSeq2SeqLM.from_pretrained("Salesforce/codet5-base",)
            decorator_to_use = self.track_no_runtime
        elif engine == 'onnx':
            model_dir = 'models/onnx/codet5-base'
            #tokenizer = AutoTokenizer.from_pretrained(model_dir) 
            #model = ORTModelForSeq2SeqLM.from_pretrained(model_dir,    return_dict = True, use_cache = True)            
            #model = ORTModelForSeq2SeqLM.from_pretrained(model_dir,    return_dict = True, use_cache = False)
            decorator_to_use = self.track_onnx
        elif engine == 'ov':
            model_dir = 'models/ov/codet5-base'
            #tokenizer = AutoTokenizer.from_pretrained('models/onnx/codet5-base') 
            #model = ORTModelForSeq2SeqLM.from_pretrained(model_dir,    return_dict = True, use_cache = True)            
            #model = OVModelForSeq2SeqLM.from_pretrained(model_dir,    return_dict = True, use_cache = False)
            decorator_to_use = self.track_ov
        elif engine == 'torchscript':
            model_dir = 'models/torchscript/codet5-base.pt'
            #tokenizer = AutoTokenizer.from_pretrained('models/onnx/codet5-base') 
            # Load the TorchScript model
            #model = torch.jit.load(model_dir)
            #print(loaded_model.code)
            #model.eval()  # Set the model to evaluation mode, turn off gradients computation
            decorator_to_use = self.track_torchscript
        
    
        #@track_emissions(project_name = "codet5-base", output_file = RESULTS_DIR + "emissions_codet5-base.csv")
        @decorator_to_use
        def infer(text: str, model, tokenizer, engine) -> str:
            prediction = None
            if(engine != 'torchscript'):
                #text = "def greet(user): print(f'hello <extra_id_0>!')"
                #input_ids = tokenizer(text, return_tensors="pt",max_length=MAX_LENGTH,padding='max_length').input_ids
                input_ids = tokenizer(text, return_tensors="pt",).input_ids
                # simply generate a single sequence
                #generated_ids = model.generate(input_ids, max_length=8)
                input_ids = input_ids.to(device)
                model.to(device)
                generated_ids = model.generate(input_ids, max_length=MAX_LENGTH) 
                
                # decode
                prediction = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
                
            elif (engine == 'torchscript'):
                #inputs = tokenizer.encode_plus(text, return_tensors = 'pt')
                inputs = tokenizer(text, return_tensors="pt", max_length=MAX_LENGTH,padding='max_length')#padding='max_length'
                print("check-inputs: ", inputs)
                input_ids = inputs["input_ids"]
                print("check-inputs: ", input_ids)

                attention_mask = inputs["attention_mask"]
                input_tuple = [input_ids,attention_mask,input_ids] # decoder_input_ids["input_ids"]

                input_ids = input_ids.to(device)
                model.to(device)
                # Generate predictions from the model
                with torch.no_grad():
                    output = model(input_ids,attention_mask = attention_mask, decoder_input_ids = input_ids)  #t5

                # Convert the output tensor to token IDs
                predicted_token_ids = torch.argmax(output[0], dim=-1)

                # Convert the tensor to a list of lists
                predicted_token_ids = predicted_token_ids.tolist()

                #print("predicted tokens:",predicted_token_ids)
                # Decode and print the output
                prediction = tokenizer.decode(predicted_token_ids[0], skip_special_tokens=True,predict_with_generate=True)
                print("Result:")
                print(prediction)

            return prediction
        
        if device.startswith('cuda'):
            print(f"nvidia-smi process started: {GPU_RESULTS}")
            nvidiaProfiler = subprocess.Popen(nvidia_smi_command.split())
            time.sleep(GPU_SMI_SLEEP)
            response = {
                "prediction" : infer(user_input, model, tokenizer, engine)
            }
            time.sleep(GPU_SMI_SLEEP)
            nvidiaProfiler.terminate()   
            print(f"nvidia-smi process terminated: {GPU_RESULTS}")
        else:
            response = {
                "prediction" : infer(user_input, model, tokenizer, engine)
            }
        
        return response
    
    @track_emissions(project_name = "codet5-base_none", output_file = RESULTS_DIR + "emissions_codet5-base.csv")
    def track_no_runtime(self,func):
        def wrapper(*args, **kwargs):
            result = func(*args, **kwargs)
            return result
        return wrapper
        
    @track_emissions(project_name = "codet5-base_onnx", output_file = RESULTS_DIR + "emissions_codet5-base.csv")
    def track_onnx(self, func):
        def wrapper(*args, **kwargs):
            result = func(*args, **kwargs)
            return result
        return wrapper

    @track_emissions(project_name = "codet5-base_ov", output_file = RESULTS_DIR + "emissions_codet5-base.csv")
    def track_ov(self, func):
        def wrapper(*args, **kwargs):
            result = func(*args, **kwargs)
            return result
        return wrapper

    @track_emissions(project_name = "codet5-base_torchscript", output_file = RESULTS_DIR + "emissions_codet5-base.csv")
    def track_torchscript(self, func):
        def wrapper(*args, **kwargs):
            result = func(*args, **kwargs)
            return result
        return wrapper

# not updated, last: may
class Codet5p_220mModel(Model):
    """_summary_ Creates a Codet5p_220m model. Inherits from Model()

    Args:
        Model (_type_): _description_
    """

    def __init__(self):
        super().__init__('codet5p-220m', ML_task.CODE)
        
    def predict(self, user_input: str, engine = None,model = None, tokenizer = None):
        
        checkpoint = "Salesforce/codet5p-220m"
        #device = "cpu" # for GPU usage or "cpu" for CPU usage

        

        print(f'Runtime engine: {engine}')
        if engine not in runtime_engines:
            #tokenizer = AutoTokenizer.from_pretrained(checkpoint)
            # torch_dtype = 'auto' not implemented
            #model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint,)
            #model = T5ForConditionalGeneration.from_pretrained(checkpoint, device_map = 'auto')
            decorator_to_use = self.track_no_runtime
        elif engine == 'onnx':
            model_dir = 'models/onnx/codet5p-220'
            #tokenizer = AutoTokenizer.from_pretrained(model_dir) 
            #model = ORTModelForSeq2SeqLM.from_pretrained(model_dir,    return_dict = True, use_cache = True)            
            #model = ORTModelForSeq2SeqLM.from_pretrained(model_dir,    return_dict = True, use_cache = False)
            decorator_to_use = self.track_onnx
        elif engine == 'ov':
            model_dir = 'models/ov/codet5p-220'
            #tokenizer = AutoTokenizer.from_pretrained('models/onnx/codet5p-220') 
            #model = OVModelForSeq2SeqLM.from_pretrained(model_dir,    return_dict = True, use_cache = False)
            decorator_to_use = self.track_ov
        elif engine == 'torchscript':
            model_dir = 'models/torchscript/codet5p-220.pt'
            #tokenizer = AutoTokenizer.from_pretrained('models/onnx/codet5p-220') 
            # Load the TorchScript model
            #model = torch.jit.load(model_dir)
            #print(loaded_model.code)
            #model.eval()  # Set the model to evaluation mode, turn off gradients computation
            decorator_to_use = self.track_torchscript
            
        #@track_emissions(project_name = "codet5p-220m", output_file = RESULTS_DIR + "emissions_codet5p-220m.csv")
        @decorator_to_use
        def infer( text: str, model, tokenizer, engine) -> str:
            prediction = None
            if (engine != 'torchscript'):
                
                #inputs = tokenizer.encode(text, return_tensors="pt",max_length=MAX_LENGTH,padding='max_length').to('cpu')
                inputs = tokenizer.encode(text, return_tensors="pt",)#.to('cpu')
                
                #outputs = model.generate(inputs, max_length=10,max_new_tokens = 30)
                #outputs = model.generate(inputs, max_new_tokens = 30)

                inputs = inputs.to(device)
                model.to(device)

                outputs = model.generate(inputs, max_length=MAX_LENGTH)
                
                prediction = tokenizer.decode(outputs[0], skip_special_tokens=True)
            elif(engine == 'torchscript'):
                #inputs = tokenizer.encode_plus(text, return_tensors = 'pt')
                inputs = tokenizer(text, return_tensors="pt",max_length=MAX_LENGTH,padding='max_length',) # padding='max_length'
                #inputs = tokenizer.encode_plus(text, return_tensors = 'pt')

                input_ids = inputs["input_ids"]

                attention_mask = inputs["attention_mask"]
                input_tuple = [input_ids,attention_mask,input_ids] # decoder_input_ids["input_ids"]

                input_ids = input_ids.to(device)
                attention_mask = attention_mask.to(device)
                #model.to(device)

                # Generate predictions from the model
                with torch.no_grad():
                    output = model(input_ids,attention_mask = attention_mask, decoder_input_ids = input_ids)  #t5
                
                # Convert the output tensor to token IDs
                predicted_token_ids = torch.argmax(output[0], dim=-1)

                # Convert the tensor to a list of lists
                predicted_token_ids = predicted_token_ids.tolist()

                #print("predicted tokens:",predicted_token_ids)
                # Decode and print the output
                prediction = tokenizer.decode(predicted_token_ids[0], skip_special_tokens=True,predict_with_generate=True)
                print("Result:")
                print(prediction)
            
            return prediction
        
        if device.startswith('cuda'):
            print(f"nvidia-smi process started: {GPU_RESULTS}")
            nvidiaProfiler = subprocess.Popen(nvidia_smi_command.split())
            time.sleep(GPU_SMI_SLEEP)
            response = {
                "prediction" : infer(user_input, model, tokenizer, engine)
            }
            time.sleep(GPU_SMI_SLEEP)
            nvidiaProfiler.terminate()   
            print(f"nvidia-smi process terminated: {GPU_RESULTS}")
        else:
            response = {
                "prediction" : infer(user_input, model, tokenizer, engine)
            }

        return response
    
    @track_emissions(project_name = "codet5p-220m_none", output_file = RESULTS_DIR + "emissions_codet5p-220m.csv")
    def track_no_runtime(self,func):
        def wrapper(*args, **kwargs):
            result = func(*args, **kwargs)
            return result
        return wrapper
        
    @track_emissions(project_name = "codet5p-220m_onnx", output_file = RESULTS_DIR + "emissions_codet5p-220m.csv")
    def track_onnx(self, func):
        def wrapper(*args, **kwargs):
            result = func(*args, **kwargs)
            return result
        return wrapper

    @track_emissions(project_name = "codet5p-220m_ov", output_file = RESULTS_DIR + "emissions_codet5p-220m.csv")
    def track_ov(self, func):
        def wrapper(*args, **kwargs):
            result = func(*args, **kwargs)
            return result
        return wrapper
    
    @track_emissions(project_name = "codet5p-220m_torchscript", output_file = RESULTS_DIR + "emissions_codet5p-220m.csv")
    def track_torchscript(self, func):
        def wrapper(*args, **kwargs):
            result = func(*args, **kwargs)
            return result
        return wrapper
    
# Cannot retrieve model in current machine
# not updated
class CodeGen_350mModel(Model):
    """_summary_ Creates a CodeGen model. Inherits from Model()

    Args:
        Model (_type_): _description_
    """

    def __init__(self):
        super().__init__(models_names.CodeGen_350m_mono, ML_task.CODE)
        
    def predict(self, user_input: str, engine = None,model = None, tokenizer = None):
        
        checkpoint = "Salesforce/codegen-350M-mono"

        print(f'Runtime engine: {engine}')
        if engine not in ['onnx','ov']:
            tokenizer = AutoTokenizer.from_pretrained(checkpoint)
            model = AutoModelForCausalLM.from_pretrained(checkpoint)
            decorator_to_use = self.track_no_runtime
        elif engine == 'onnx':
            model_dir = 'models/onnx/codegen-350M-mono/'
            tokenizer = AutoTokenizer.from_pretrained(model_dir) 
            #model = ORTModelForCausalLM.from_pretrained(model_dir, device_map = 'auto', torch_dtype = 'auto', use_cache=True)
            model = ORTModelForCausalLM.from_pretrained(model_dir,)
            decorator_to_use = self.track_onnx

        #@track_emissions(project_name = "codegen-350M-mono", output_file = RESULTS_DIR + "emissions_codegen-350M-mono.csv")
        @decorator_to_use
        def infer(text: str, model, tokenizer) -> str:
            input_ids = tokenizer(text, return_tensors="pt").input_ids

            #generated_ids = model.generate(input_ids, max_length=128)
            generated_ids = model.generate(input_ids, )
            prediction =  tokenizer.decode(generated_ids[0], skip_special_tokens=True)
            
            return prediction

        response = {
            "prediction" : infer(user_input,model,tokenizer),
        }
        
        return response

    @track_emissions(project_name = "codegen-350M-mono_none", output_file = RESULTS_DIR + "emissions_codegen-350M-mono.csv")
    def track_no_runtime(self,func):
        def wrapper(*args, **kwargs):
            result = func(*args, **kwargs)
            return result
        return wrapper
        
    @track_emissions(project_name = "codegen-350M-mono_onnx", output_file = RESULTS_DIR + "emissions_codegen-350M-mono.csv")
    def track_onnx(self, func):
        def wrapper(*args, **kwargs):
            result = func(*args, **kwargs)
            return result
        return wrapper
    
# not updated
class GPTNeo_125m(Model):
    """
    Creates a GPT-Neo model. Inherits from Model()
    """
    def __init__(self):
        super().__init__(models_names.GPT_Neo_125m, ML_task.CODE)
        
    def predict(self, user_input: str, engine = None,):
        
        model='EleutherAI/gpt-neo-125M'
        
        print(f'Runtime engine: {engine}')
        if engine not in runtime_engines:
            tokenizer = AutoTokenizer.from_pretrained(model)
            model = AutoModelForCausalLM.from_pretrained(model)
            decorator_to_use = self.track_no_runtime
        elif engine == 'onnx':
            model_dir = 'models/onnx/gpt-neo-125m'
            tokenizer = AutoTokenizer.from_pretrained(model_dir) 
            #model = ORTModelForCausalLM.from_pretrained(model_dir, device_map = 'auto', torch_dtype = 'auto', use_cache=True)
            model = ORTModelForCausalLM.from_pretrained(model_dir,)
            decorator_to_use = self.track_onnx
        elif engine == 'ov':
            model_dir = 'models/ov/gpt-neo-125m'
            tokenizer = AutoTokenizer.from_pretrained('models/onnx/gpt-neo-125m') 
            model = OVModelForCausalLM.from_pretrained(model_dir,)
            decorator_to_use = self.track_ov
        elif engine == 'torchscript':
            model_dir = 'models/torchscript/gpt-neo-125m.pt'
            tokenizer = AutoTokenizer.from_pretrained('models/onnx/gpt-neo-125m') 
            # Load the TorchScript model
            model = torch.jit.load(model_dir)
            #print(loaded_model.code)
            model.eval()  # Set the model to evaluation mode, turn off gradients computation
            decorator_to_use = self.track_torchscript


        @decorator_to_use
        def infer( text: str, model, tokenizer, engine) -> str:
            if(engine != 'torchscript'):
                
                inputs = tokenizer(text, return_tensors="pt")
                # generate
                tokens = model.generate(**inputs)
                
                # decode
                prediction = tokenizer.decode(tokens[0])
                return prediction
        
            elif(engine == 'torchscript'):
                #inputs = tokenizer.encode_plus(text, return_tensors = 'pt')
                inputs = tokenizer(text, return_tensors="pt",max_length=MAX_LENGTH,padding='max_length')
                #inputs = tokenizer.encode_plus(text, return_tensors = 'pt')
                
                input_ids = inputs["input_ids"]

                attention_mask = inputs["attention_mask"]
                input_tuple = [input_ids,attention_mask,input_ids] # decoder_input_ids["input_ids"]

                # Generate predictions from the model
                with torch.no_grad():
                    output = model(input_ids,)  # Adjust max_length as needed
                
                # Convert the output tensor to token IDs
                predicted_token_ids = torch.argmax(output[0], dim=-1)

                # Convert the tensor to a list of lists
                predicted_token_ids = predicted_token_ids.tolist()

                #print("predicted tokens:",predicted_token_ids)
                # Decode and print the output
                prediction = tokenizer.decode(predicted_token_ids[0], skip_special_tokens=True,predict_with_generate=True)
                print("Result:")
                print(prediction)
                return prediction

            

        response = {
            "prediction" : infer(user_input, model, tokenizer,engine)
        }
        return response
        
    @track_emissions(project_name = "gpt-neo-125M_none", output_file = RESULTS_DIR + "emissions_gpt-neo-125m.csv")
    def track_no_runtime(self,func):
        def wrapper(*args, **kwargs):
            result = func(*args, **kwargs)
            return result
        return wrapper
        
    @track_emissions(project_name = "gpt-neo-125M_onnx", output_file = RESULTS_DIR + "emissions_gpt-neo-125m.csv")
    def track_onnx(self, func):
        def wrapper(*args, **kwargs):
            result = func(*args, **kwargs)
            return result
        return wrapper
    
    @track_emissions(project_name = "gpt-neo-125M_ov", output_file = RESULTS_DIR + "emissions_gpt-neo-125m.csv")
    def track_ov(self, func):
        def wrapper(*args, **kwargs):
            result = func(*args, **kwargs)
            return result
        return wrapper

    @track_emissions(project_name = "gpt-neo-125M_torchscript", output_file = RESULTS_DIR + "emissions_gpt-neo-125m.csv")
    def track_torchscript(self, func):
        def wrapper(*args, **kwargs):
            result = func(*args, **kwargs)
            return result
        return wrapper

#running
class CodeParrot_smallModel(Model):
    """_summary_ Creates a CodeGen model. Inherits from Model()

    Args:
        Model (_type_): _description_
    """

    def __init__(self):
        super().__init__('codeparrot-small', ML_task.CODE)
        
    def predict(self, user_input: str, engine = None, model = None, tokenizer = None):
        #model = "codeparrot/codeparrot-small"
        #generator = pipeline("text-generation", model= model)
        
        print(f'Runtime engine: {engine}')
        if engine not in runtime_engines:
            #tokenizer = AutoTokenizer.from_pretrained(model)
            #model = AutoModelForCausalLM.from_pretrained(model)
            decorator_to_use = self.track_no_runtime
        elif engine == 'onnx':
            model_dir = 'models/onnx/codeparrot-small'
            #tokenizer = AutoTokenizer.from_pretrained(model_dir) 
            #model = ORTModelForCausalLM.from_pretrained(model_dir, device_map = 'auto', torch_dtype = 'auto', use_cache=True)
            #model = ORTModelForCausalLM.from_pretrained(model_dir,)
            decorator_to_use = self.track_onnx
        elif engine == 'ov':
            model_dir = 'models/ov/codeparrot-small'
            #tokenizer = AutoTokenizer.from_pretrained('models/onnx/codeparrot-small') 
            #model = OVModelForCausalLM.from_pretrained(model_dir, ov_config={"CACHE_DIR":""}) #use_cache=True # ov_config={"PERFORMANCE_HINT": "LATENCY", "CACHE_DIR":""}
            decorator_to_use = self.track_ov
        elif engine == 'torchscript':
            model_dir = 'models/torchscript/codeparrot-small.pt'
            #tokenizer = AutoTokenizer.from_pretrained('models/onnx/codeparrot-small') 
            # Load the TorchScript model
            #model = torch.jit.load(model_dir)
            #print(loaded_model.code)
            #model.eval()  # Set the model to evaluation mode, turn off gradients computation
            decorator_to_use = self.track_torchscript
        
        #@track_emissions(project_name = "codeparrot-small", output_file = RESULTS_DIR + "emissions_codeparrot-small.csv")
        #@decorator_to_use
        def infer( text: str, model, tokenizer, engine) -> str:
            prediction = None
            if(engine != 'torchscript'):
                #prediction = generator(text)
                #tokenizer.add_special_tokens({'pad_token': '[PAD]'})
                #inputs = tokenizer(text, return_tensors="pt",max_length=MAX_LENGTH,padding='max_length')
                inputs = tokenizer(text, return_tensors="pt")

                inputs = inputs.to(device)
                model.to(device)

                # generate
                tokens = model.generate(**inputs, max_length=MAX_LENGTH)
                
                # decode
                prediction = tokenizer.decode(tokens[0])
            elif (engine == 'torchscript'):
                #inputs = tokenizer.encode_plus(text, return_tensors = 'pt')
                #inputs = tokenizer(text, return_tensors="pt",truncation=True, max_length=MAX_LENGTH)
                #tokenizer.add_special_tokens({'pad_token': '[PAD]'})
                #tokenizer.add_special_tokens({'pad_token': '[PAD]'})
                #inputs = tokenizer(text, return_tensors="pt",max_length=MAX_LENGTH,padding='max_length' )

                tokenizer.pad_token = tokenizer.eos_token
                #inputs = tokenizer.encode_plus(text, return_tensors = 'pt')

                inputs = tokenizer.encode_plus(text,max_length = MAX_LENGTH, padding = 'max_length', return_tensors = 'pt',)
        
                input_ids = inputs["input_ids"]

                attention_mask = inputs["attention_mask"]
                input_tuple = [input_ids,attention_mask,input_ids] # decoder_input_ids["input_ids"]

                if device.startswith('cuda'): 
                    input_ids = input_ids.to(device)
                    model.to(device)
                
                # Generate predictions from the model
                with torch.no_grad():
                    output = model(input_ids,)  # Adjust max_length as needed
                    
                    # if model in [ 'codet5-base', 'codet5p-220']:
                    #     print(attention_mask)
                    #     output = model(input_ids,attention_mask = attention_mask, decoder_input_ids = input_ids)  #t5
                    # else:
                    #     output = model(input_ids,)  # Adjust max_length as needed
                
                # Convert the output tensor to token IDs
                predicted_token_ids = torch.argmax(output[0], dim=-1)

                # Convert the tensor to a list of lists
                predicted_token_ids = predicted_token_ids.tolist()

                #print("predicted tokens:",predicted_token_ids)
                # Decode and print the output
                prediction = tokenizer.decode(predicted_token_ids[0], skip_special_tokens=True,predict_with_generate=True)
                print("Result:")
                print(prediction)
            
            return prediction
 
        def wattmeter_task():
            global wattmeter
            global response

            if device.startswith('cuda'):
                #gpu_metrics = RESULTS_DIR + f"gpu-power_{self.name}_{engine}.csv"
                #nvidia_smi_command = f"nvidia-smi -i {GPU_ID} --query-gpu=timestamp,gpu_name,utilization.gpu,utilization.memory,memory.total,memory.used,power.draw,power.max_limit,temperature.gpu --format=csv -lms {GPU_SMI_MS} -f {gpu_metrics}"

                #nvidia_smi_command = f"nvidia-smi -i {GPU_ID} --query-gpu=timestamp,gpu_name,utilization.gpu,utilization.memory,memory.total,memory.used,power.draw,power.max_limit,temperature.gpu --format=csv -lms {GPU_SMI_MS} >> {gpu_metrics}"
                #print(f"nvidia-smi process started: {gpu_metrics}")
                #nvidiaProfiler = subprocess.Popen(nvidia_smi_command,shell=True)
                #nvidiaProfiler = subprocess.Popen(nvidia_smi_command.split())
                #time.sleep(GPU_SMI_SLEEP)
                
                response = {
                    "prediction" : infer(user_input, model, tokenizer, engine)
                }
                #time.sleep(GPU_SMI_SLEEP)
                #nvidiaProfiler.terminate()
                #nvidiaProfiler.wait()
                #print(f"nvidia-smi process terminated: {gpu_metrics}")
            else:
                
                response = {
                    "prediction" : infer(user_input, model, tokenizer, engine)
                }
            
            wattmeter = False


        # start wattmeter
        task_thread = threading.Thread(target=wattmeter_task)
        task_thread.start()

        global wattmeter
        wattmeter = True
        get_wattmeter_data(f'{self.name}_{engine}')

        # Wait for the wattmeter task thread to complete
        task_thread.join()
        print("wattmeter_task has completed.")
        
        return response

    @track_emissions(project_name = "codeparrot-small_none", output_file = RESULTS_DIR + "emissions_codeparrot-small.csv")
    def track_no_runtime(self,func):
        def wrapper(*args, **kwargs):
            result = func(*args, **kwargs)
            return result
        return wrapper
        
    @track_emissions(project_name = "codeparrot-small_onnx", output_file = RESULTS_DIR + "emissions_codeparrot-small.csv")
    def track_onnx(self, func):
        def wrapper(*args, **kwargs):
            result = func(*args, **kwargs)
            return result
        return wrapper
    
    @track_emissions(project_name = "codeparrot-small_ov", output_file = RESULTS_DIR + "emissions_codeparrot-small.csv")
    def track_ov(self, func):
        def wrapper(*args, **kwargs):
            result = func(*args, **kwargs)
            return result
        return wrapper

    @track_emissions(project_name = "codeparrot-small_torchscript", output_file = RESULTS_DIR + "emissions_codeparrot-small.csv")
    def track_torchscript(self, func):
        def wrapper(*args, **kwargs):
            result = func(*args, **kwargs)
            return result
        return wrapper

#running
class Pythia_410mModel(Model):
    """_summary_ Creates a Pythia model. Inherits from Model()

    Args:
        Model (_type_): _description_
    """

    def __init__(self):
        super().__init__('pythia-410m', ML_task.CODE)
        
    def predict(self, user_input: str, engine = None, model = None, tokenizer = None):
        
        print(f'Runtime engine: {engine}')

        if engine not in runtime_engines:
            # model = GPTNeoXForCausalLM.from_pretrained(
            # "EleutherAI/pythia-410m",
            # #revision="step3000",
            # #cache_dir="./pythia-410m/step3000",
            # )

            # tokenizer = AutoTokenizer.from_pretrained(
            # "EleutherAI/pythia-410m",
            # #revision="step3000",
            # #cache_dir="./pythia-410m/step3000",
            # )
            decorator_to_use = self.track_no_runtime
        elif engine == 'onnx':
            model_dir = 'models/onnx/pythia-410m'
            #tokenizer = AutoTokenizer.from_pretrained(model_dir) 
            #model = ORTModelForCausalLM.from_pretrained(model_dir, device_map = 'auto', torch_dtype = 'auto', use_cache=True)
            #model = ORTModelForCausalLM.from_pretrained(model_dir,)
            decorator_to_use = self.track_onnx
        elif engine == 'ov':
            model_dir = 'models/ov/pythia-410m'
            #tokenizer = AutoTokenizer.from_pretrained('models/onnx/pythia-410m') 
            #model = OVModelForCausalLM.from_pretrained(model_dir,)
            decorator_to_use = self.track_ov
        elif engine == 'torchscript':
            model_dir = 'models/torchscript/pythia-410m.pt'
            #tokenizer = AutoTokenizer.from_pretrained('models/onnx/pythia-410m') 
            # Load the TorchScript model
            #model = torch.jit.load(model_dir)
            #print(loaded_model.code)
            #model.eval()  # Set the model to evaluation mode, turn off gradients computation
            decorator_to_use = self.track_torchscript

        #@track_emissions(project_name = "pythia-410m", output_file = RESULTS_DIR + "emissions_pythia-410m.csv")
        #@decorator_to_use # commented because codecarbon is not used
        def infer( text: str, model, tokenizer, engine) -> str:
            prediction = None
            if(engine != 'torchscript' ):
                # tokenize
                #tokenizer.pad_token = tokenizer.eos_token
                #inputs = tokenizer(text, return_tensors="pt",max_length=MAX_LENGTH,padding='max_length')
                inputs = tokenizer(text, return_tensors="pt",)
                inputs = inputs.to(device)
                model.to(device)
                # generate
                tokens = model.generate(**inputs, max_length=MAX_LENGTH)
                # decode
                prediction = tokenizer.decode(tokens[0])
            elif (engine == 'torchscript'):
                #inputs = tokenizer.encode_plus(text, return_tensors = 'pt')
                #inputs = tokenizer(text, return_tensors="pt",truncation=True, max_length=MAX_LENGTH)
                #inputs = tokenizer(text, return_tensors="pt",max_length=MAX_LENGTH,padding='max_length')
                
                tokenizer.pad_token = tokenizer.eos_token # pythia tokenizer does not have pad token
                inputs = tokenizer.encode_plus(text,max_length = MAX_LENGTH, padding = 'max_length', return_tensors = 'pt',)

                input_ids = inputs["input_ids"]

                attention_mask = inputs["attention_mask"]
                input_tuple = [input_ids,attention_mask,input_ids] # decoder_input_ids["input_ids"]

                if device.startswith('cuda'):
                    input_ids = input_ids.to(device)
                    model.to(device)
                # Generate predictions from the model
                with torch.no_grad():
                    output = model(input_ids,)  # Adjust max_length as needed
                
                # Convert the output tensor to token IDs
                predicted_token_ids = torch.argmax(output[0], dim=-1)

                # Convert the tensor to a list of lists
                predicted_token_ids = predicted_token_ids.tolist()

                #print("predicted tokens:",predicted_token_ids)
                # Decode and print the output
                prediction = tokenizer.decode(predicted_token_ids[0], skip_special_tokens=True,predict_with_generate=True)
                print("Result:")
                print(prediction)
            
            return prediction


        def wattmeter_task():
            global wattmeter
            global response

            if device.startswith('cuda'):
                # gpu_metrics = RESULTS_DIR + f"gpu-power_{self.name}_{engine}.csv"
                # nvidia_smi_command = f"nvidia-smi -i {GPU_ID} --query-gpu=timestamp,gpu_name,utilization.gpu,utilization.memory,memory.total,memory.used,power.draw,power.max_limit,temperature.gpu --format=csv -lms {GPU_SMI_MS} -f {gpu_metrics}"
                # print(f"nvidia-smi process started: {gpu_metrics}")
                
                # nvidiaProfiler = subprocess.Popen(nvidia_smi_command.split())
                # time.sleep(GPU_SMI_SLEEP)
                response = {
                    "prediction" : infer(user_input, model, tokenizer, engine)
                }
                # time.sleep(GPU_SMI_SLEEP)
                # nvidiaProfiler.terminate()   
                # print(f"nvidia-smi process terminated: {gpu_metrics}")
            else:
                response = {
                    "prediction" : infer(user_input, model, tokenizer, engine)
                }

            wattmeter = False


        # start wattmeter
        task_thread = threading.Thread(target=wattmeter_task)
        task_thread.start()

        global wattmeter
        wattmeter = True
        get_wattmeter_data(f'{self.name}_{engine}')

        # Wait for the wattmeter task thread to complete
        task_thread.join()
        print("wattmeter_task has completed.")
        
        return response

    @track_emissions(project_name = "pythia-410m_none", output_file = RESULTS_DIR + "emissions_pythia-410m.csv")
    def track_no_runtime(self,func):
        def wrapper(*args, **kwargs):
            result = func(*args, **kwargs)
            return result
        return wrapper
        
    @track_emissions(project_name = "pythia-410m_onnx", output_file = RESULTS_DIR + "emissions_pythia-410m.csv")
    def track_onnx(self, func):
        def wrapper(*args, **kwargs):
            result = func(*args, **kwargs)
            return result
        return wrapper
    
    @track_emissions(project_name = "pythia-410m_ov", output_file = RESULTS_DIR + "emissions_pythia-410m.csv")
    def track_ov(self, func):
        def wrapper(*args, **kwargs):
            result = func(*args, **kwargs)
            return result
        return wrapper

    @track_emissions(project_name = "pythia-410m_torchscript", output_file = RESULTS_DIR + "emissions_pythia-410m.csv")
    def track_torchscript(self, func):
        def wrapper(*args, **kwargs):
            result = func(*args, **kwargs)
            return result
        return wrapper

#running
class SLMModel(Model):
    """_summary_ Creates a SLM model. Inherits from Model()

    Args:
        Model (_type_): _description_
    """

    def __init__(self):
        super().__init__('SLM', ML_task.CODE)
        
    def predict(self, user_input: str, engine = None, model = None, tokenizer = None):
        
        print(f'Runtime engine: {engine}')

        if engine not in runtime_engines:
            decorator_to_use = self.track_no_runtime
        elif engine == 'onnx':
            model_dir = f'models/onnx/{model}'
            #tokenizer = AutoTokenizer.from_pretrained(model_dir) 
            #model = ORTModelForCausalLM.from_pretrained(model_dir, device_map = 'auto', torch_dtype = 'auto', use_cache=True)
            #model = ORTModelForCausalLM.from_pretrained(model_dir,)
            decorator_to_use = self.track_onnx
        elif engine == 'ov':
            model_dir = f'models/ov/{model}'
            #tokenizer = AutoTokenizer.from_pretrained('models/onnx/pythia-410m') 
            #model = OVModelForCausalLM.from_pretrained(model_dir,)
            decorator_to_use = self.track_ov
        elif engine == 'torchscript':
            model_dir = f'models/torchscript/{model}.pt'
            #tokenizer = AutoTokenizer.from_pretrained('models/onnx/pythia-410m') 
            # Load the TorchScript model
            #model = torch.jit.load(model_dir)
            #print(loaded_model.code)
            #model.eval()  # Set the model to evaluation mode, turn off gradients computation
            decorator_to_use = self.track_torchscript

        #@track_emissions(project_name = "pythia-410m", output_file = RESULTS_DIR + "emissions_pythia-410m.csv")
        #@decorator_to_use
        def infer( text: str, model, tokenizer, engine) -> str:
            prediction = None
            if(engine != 'torchscript' ):
                # tokenize
                #tokenizer.pad_token = tokenizer.eos_token
                #inputs = tokenizer(text, return_tensors="pt",max_length=MAX_LENGTH,padding='max_length')
                inputs = tokenizer(text, return_tensors="pt",)
                inputs = inputs.to(device)
                
                
                #print(f"ort_model.main_input_name: {model.main_input_name}")
                #inputs = {key: value for key, value in inputs.items() if key in model.main_input_name}
                
                # Check if the model has 'main_input_name' (specific to ONNX Runtime)
                if hasattr(model, "main_input_name"):
                    inputs = {key: value for key, value in inputs.items() if key in model.main_input_name}
                else:
                    # For PyTorch models, no filtering is needed
                    inputs = {key: value for key, value in inputs.items()}
                
                model.to(device)
                
                # generate
                tokens = model.generate(**inputs, max_length=MAX_LENGTH)
                # decode
                prediction = tokenizer.decode(tokens[0])
            elif (engine == 'torchscript'):
                #inputs = tokenizer.encode_plus(text, return_tensors = 'pt')
                #inputs = tokenizer(text, return_tensors="pt",truncation=True, max_length=MAX_LENGTH)
                #inputs = tokenizer(text, return_tensors="pt",max_length=MAX_LENGTH,padding='max_length')
                
                tokenizer.pad_token = tokenizer.eos_token # tinyllama tokenizer does not have pad token
                if self.name in ['stablecode-3b-completion']: # 
                    print("Adding token")
                    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
                inputs = tokenizer.encode_plus(text,max_length = MAX_LENGTH, padding = 'max_length', return_tensors = 'pt',)

                input_ids = inputs["input_ids"]

                attention_mask = inputs["attention_mask"]
                input_tuple = [input_ids,attention_mask,input_ids] # decoder_input_ids["input_ids"]

                if device.startswith('cuda'):
                    input_ids = input_ids.to(device)
                    model.to(device)
                # Generate predictions from the model
                with torch.no_grad():
                    output = model(input_ids,)  # Adjust max_length as needed
                
                # Convert the output tensor to token IDs
                predicted_token_ids = torch.argmax(output[0], dim=-1)

                # Convert the tensor to a list of lists
                predicted_token_ids = predicted_token_ids.tolist()

                #print("predicted tokens:",predicted_token_ids)
                # Decode and print the output
                prediction = tokenizer.decode(predicted_token_ids[0], skip_special_tokens=True,predict_with_generate=True)
                print("Result:")
                print(prediction)
            
            return prediction

        def wattmeter_task():
            global wattmeter
            global response
        
            if device.startswith('cuda'):
                #gpu_metrics = RESULTS_DIR+ f"gpu-power_{self.name}_{engine}.csv"
                #nvidia_smi_command = f"nvidia-smi -i {GPU_ID} --query-gpu=timestamp,gpu_name,utilization.gpu,utilization.memory,memory.total,memory.used,power.draw,power.max_limit,temperature.gpu --format=csv -lms {GPU_SMI_MS} -f {gpu_metrics}"
                #print(f"nvidia-smi process started: {gpu_metrics}")
                #nvidiaProfiler = subprocess.Popen(nvidia_smi_command.split())
                #time.sleep(GPU_SMI_SLEEP)
                response = {
                    "prediction" : infer(user_input, model, tokenizer, engine)
                }
                #time.sleep(GPU_SMI_SLEEP)
                #nvidiaProfiler.terminate()   
                #print(f"nvidia-smi process terminated: {gpu_metrics}")
            else:
                response = {
                    "prediction" : infer(user_input, model, tokenizer, engine)
                }
            
            wattmeter = False


        # start wattmeter
        task_thread = threading.Thread(target=wattmeter_task)
        task_thread.start()

        global wattmeter
        wattmeter = True
        get_wattmeter_data(f'{self.name}_{engine}')

        # Wait for the wattmeter task thread to complete
        task_thread.join()
        print("wattmeter_task has completed.")
        
        return response

    @track_emissions(project_name = "slm_none", output_file = RESULTS_DIR + "emissions_slm.csv")
    def track_no_runtime(self,func):
        def wrapper(*args, **kwargs):
            result = func(*args, **kwargs)
            return result
        return wrapper
        
    @track_emissions(project_name = "slm_onnx", output_file = RESULTS_DIR + "emissions_slm.csv")
    def track_onnx(self, func):
        def wrapper(*args, **kwargs):
            result = func(*args, **kwargs)
            return result
        return wrapper
    
    @track_emissions(project_name = "slm_ov", output_file = RESULTS_DIR + "emissions_slm.csv")
    def track_ov(self, func):
        def wrapper(*args, **kwargs):
            result = func(*args, **kwargs)
            return result
        return wrapper

    @track_emissions(project_name = "slm_torchscript", output_file = RESULTS_DIR + "emissions_slm.csv")
    def track_torchscript(self, func):
        def wrapper(*args, **kwargs):
            result = func(*args, **kwargs)
            return result
        return wrapper
