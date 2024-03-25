"""Main script: it includes our API initialization and endpoints.


"""
#import pickle
from datetime import datetime
from functools import wraps
from http import HTTPStatus
from pathlib import Path
from typing import Dict, List

from fastapi import FastAPI, Request, HTTPException

from app.schemas_code import PredictCodeT5_Base, PredictCodet5p_220m, PredictCodeGen_350m, PredictGPTNeo_125m, PredictCodeParrot_small, PredictPythia_410m, PredictPayload
from app.schemas_code import PredictTinyllama
#from transformers import pipeline

# Local modules
#from app.models import LMBERTModel, Model, T5Model, CNNModel, CodeGenModel, Pythia_70mModel, Codet5p_220mModel
from app.models_code_load import  CodeT5_BaseModel, Codet5p_220mModel, CodeGen_350mModel, GPTNeo_125m, CodeParrot_smallModel, Pythia_410mModel
from app.models_code_load import TinyllamaModel

from fastapi.responses import FileResponse

import csv
import os
import builtins

from app.utils_api import *

script_name = os.path.basename(__file__)
def print(*args, **kwargs):
    builtins.print(f"[{script_name}] ",*args, **kwargs)

print("------------------------modules loaded!------------------------")

MODELS_DIR = Path("models/")
NAME_APP = "cloud-api"
model_wrappers_list: List[dict] = []

# Define application
app = FastAPI(
    title=NAME_APP,
    description="This API lets you make predictions on .. using a couple of simple models.",
    version="0.1",
)

loaded_model = None
loaded_tokenizer = None

def construct_response(f):
    """Construct a JSON response for an endpoint's results."""

    @wraps(f)
    def wrap(request: Request, *args, **kwargs):
        results = f(request, *args, **kwargs)

        # Construct response
        response = {
            "message": results["message"],
            "method": request.method,
            "status-code": results["status-code"],
            "timestamp": datetime.now().isoformat(),
            "url": request.url._url,
        }

        # Add data
        if "data" in results:
            response["data"] = results["data"]

        return response

    return wrap


# @app.on_event("startup")
# def _load_models():
#     """Loads all pickled models found in `MODELS_DIR` and adds them to `models_list`"""

#     model_paths = [
#         filename for filename in MODELS_DIR.iterdir() if filename.suffix == ".pkl"
#     ]

#     for path in model_paths:
#         with open(path, "rb") as file:
#             model_wrapper = pickle.load(file)
#             model_wrappers_list.append(model_wrapper)

    

@app.get("/", tags=["General"])  # path operation decorator
@construct_response
def _index(request: Request):
    """Root endpoint."""

    response = {
        "message": HTTPStatus.OK.phrase,
        "status-code": HTTPStatus.OK,
        "data": {"message": f"Welcome to {NAME_APP}! Please, read the `/docs`!"},
    }
    return response


@app.post("/load_model/{engine}/{model_name}")
async def load_model(model_name: str,engine:str):
    print("-------loading----------")
    global loaded_model, loaded_tokenizer
    try:
        # Load the model and tokenizer
        print(f'Runtime engine: {engine}')
        if engine not in ['onnx','ov','torchscript']:
            selected_class = info_models[engine][model_name]["m_class"]
            model_dir = info_models[engine][model_name]["model_dir"]
            tokenizer_dir = info_models[engine][model_name]["tokenizer_dir"]

            print(f"class: {selected_class} - model_dir {model_dir}")
            loaded_tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir)
            loaded_model = selected_class.from_pretrained(model_dir)

        elif engine == 'onnx':
            selected_class = info_models[engine][model_name]["m_class"]
            model_dir = info_models[engine][model_name]["model_dir"]
            tokenizer_dir = info_models['onnx'][model_name]["tokenizer_dir"] # using same than onnx

            print(f"class: {selected_class} - model_dir {model_dir}")
            loaded_tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir)
            #model = ORTModelForSeq2SeqLM.from_pretrained(model_dir,    return_dict = True, use_cache = True)            
            if model_name =='codeparrot-small':
                loaded_model = selected_class.from_pretrained(model_dir,)
            elif model_name == 'pythia-410m':
                loaded_model = selected_class.from_pretrained(model_dir,)
            elif model_name == 'tinyllama':
                loaded_model = selected_class.from_pretrained(model_dir,)
            else:
                loaded_model = selected_class.from_pretrained(model_dir,    return_dict = True, use_cache = False)
                
        elif engine == 'ov':
            # model_dir = 'models/ov/codet5-base'
            # loaded_tokenizer = AutoTokenizer.from_pretrained('models/onnx/codet5-base') 
            # loaded_model = OVModelForSeq2SeqLM.from_pretrained(model_dir,    return_dict = True, use_cache = False)
            selected_class = info_models[engine][model_name]["m_class"]
            model_dir = info_models[engine][model_name]["model_dir"]
            tokenizer_dir = info_models['onnx'][model_name]["tokenizer_dir"] # using same than onnx

            print(f"class: {selected_class} - model_dir {model_dir}")
            loaded_tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir) 
            
            if model_name =='codeparrot-small':
                loaded_model = selected_class.from_pretrained(model_dir, ov_config={"CACHE_DIR":""})
            elif model_name == 'pythia-410m':
                loaded_model = selected_class.from_pretrained(model_dir,)
            elif model_name == 'tinyllama':
                loaded_model = selected_class.from_pretrained(model_dir,)
            else:
                loaded_model = selected_class.from_pretrained(model_dir,return_dict = True, use_cache = False)
        elif engine == 'torchscript':
            #selected_class = info_models[engine][model_name]["m_class"]
            model_dir = info_models[engine][model_name]["model_dir"]
            tokenizer_dir = info_models['onnx'][model_name]["tokenizer_dir"] # using same than onnx
            loaded_tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir) 
            # Load the TorchScript model
            loaded_model = torch.jit.load(model_dir)
            #print(loaded_model.code)
            loaded_model.eval()  # Set the model to evaluation mode, turn off gradients computation
            #decorator_to_use = self.track_torchscript
        return {"message": f"Model {model_name} loaded successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/huggingface_models/codet5-base/{engine}", tags=["Hugging Face Models"])
@construct_response
def _predict_codet5_base(request: Request, payload: PredictCodeT5_Base, engine: str = None):
    """bert-base-uncased model."""
    
    input_text = payload.input_text 
    print("Input text")
    print(input_text)
    #model_wrapper = next((m for m in model_wrappers_list if m["type"] == type), None)

    model = CodeT5_BaseModel()
    print(f"Model: {model.name}")

    if input_text:
        prediction = model.predict(input_text, engine, loaded_model, loaded_tokenizer)
        
        response = {
            "message": HTTPStatus.OK.phrase,
            "status-code": HTTPStatus.OK,
            "data": {
                #"model-type": model_wrapper["type"],
                "model-type": model.name,
                "input_text": input_text,
                "prediction": prediction,
                #"predicted_type": predicted_type,
            },
        }
    else:
        response = {
            "message": "Model not found",
            "status-code": HTTPStatus.BAD_REQUEST,
        }
    return response


@app.post("/huggingface_models/codet5p-220/{engine}", tags=["Hugging Face Models"])
@construct_response
def _predict_codet5p_220(request: Request, payload: PredictCodet5p_220m, engine: str = None):
    """T5 model."""
    
    input_text = payload.input_text 
    print("Input text")
    print(input_text)
    #model_wrapper = next((m for m in model_wrappers_list if m["type"] == type), None)

    model = Codet5p_220mModel()
    print(f"Model: {model.name}")

    if input_text:
        prediction = model.predict(input_text, engine, loaded_model, loaded_tokenizer)
        
        response = {
            "message": HTTPStatus.OK.phrase,
            "status-code": HTTPStatus.OK,
            "data": {
                #"model-type": model_wrapper["type"],
                "model-type": model.name,
                "input_text": input_text,
                "prediction": prediction,
                #"predicted_type": predicted_type,
            },
        }
    else:
        response = {
            "message": "Model not found",
            "status-code": HTTPStatus.BAD_REQUEST,
        }
    return response





@app.post("/huggingface_models/gpt-neo-125m/{engine}", tags=["Hugging Face Models"])
@construct_response
def _predict_gpt_neo_125m(request: Request, payload: PredictGPTNeo_125m, engine: str = None):
    """T5 model."""
    
    input_text = payload.input_text 
    print("Input text")
    print(input_text)
    #model_wrapper = next((m for m in model_wrappers_list if m["type"] == type), None)

    model = GPTNeo_125m()
    print(f"Model: {model.name}")

    if input_text:
        prediction = model.predict(input_text, engine)
        
        response = {
            "message": HTTPStatus.OK.phrase,
            "status-code": HTTPStatus.OK,
            "data": {
                #"model-type": model_wrapper["type"],
                "model-type": model.name,
                "input_text": input_text,
                "prediction": prediction,
                #"predicted_type": predicted_type,
            },
        }
    else:
        response = {
            "message": "Model not found",
            "status-code": HTTPStatus.BAD_REQUEST,
        }
    return response


@app.post("/huggingface_models/codeparrot-small/{engine}", tags=["Hugging Face Models"])
@construct_response
def _predict_codeparrot_small(request: Request, payload: PredictCodeParrot_small, engine: str = None):
    """Codet5p_220m model."""
    
    input_text = payload.input_text 
    print("Input text")
    print(input_text)
    #model_wrapper = next((m for m in model_wrappers_list if m["type"] == type), None)

    model = CodeParrot_smallModel()
    print(f"Model: {model.name}")

    if input_text:
        prediction = model.predict(input_text, engine, loaded_model, loaded_tokenizer)
        
        response = {
            "message": HTTPStatus.OK.phrase,
            "status-code": HTTPStatus.OK,
            "data": {
                #"model-type": model_wrapper["type"],
                "model-type": model.name,
                "input_text": input_text,
                "prediction": prediction,
                #"predicted_type": predicted_type,
            },
        }
    else:
        response = {
            "message": "Model not found",
            "status-code": HTTPStatus.BAD_REQUEST,
        }
    return response


@app.post("/huggingface_models/pythia-410m/{engine}", tags=["Hugging Face Models"])
@construct_response
def _predict_pythia_410m(request: Request, payload: PredictPythia_410m, engine: str = None):
    """Codet5p_220m model."""
    
    input_text = payload.input_text 
    print("Input text")
    print(input_text)
    #model_wrapper = next((m for m in model_wrappers_list if m["type"] == type), None)

    model = Pythia_410mModel()
    print(f"Model: {model.name}")

    if input_text:
        prediction = model.predict(input_text, engine, loaded_model, loaded_tokenizer)
        
        response = {
            "message": HTTPStatus.OK.phrase,
            "status-code": HTTPStatus.OK,
            "data": {
                #"model-type": model_wrapper["type"],
                "model-type": model.name,
                "input_text": input_text,
                "prediction": prediction,
                #"predicted_type": predicted_type,
            },
        }
    else:
        response = {
            "message": "Model not found",
            "status-code": HTTPStatus.BAD_REQUEST,
        }
    return response

@app.post("/huggingface_models/tinyllama/{engine}", tags=["Hugging Face Models"])
@construct_response
def _predict_tinyllama(request: Request, payload: PredictTinyllama, engine: str = None):
    """Codet5p_220m model."""
    
    input_text = payload.input_text 
    print("Input text")
    print(input_text)
    #model_wrapper = next((m for m in model_wrappers_list if m["type"] == type), None)

    model = TinyllamaModel()
    print(f"Model: {model.name}")

    if input_text:
        prediction = model.predict(input_text, engine, loaded_model, loaded_tokenizer)
        
        response = {
            "message": HTTPStatus.OK.phrase,
            "status-code": HTTPStatus.OK,
            "data": {
                #"model-type": model_wrapper["type"],
                "model-type": model.name,
                "input_text": input_text,
                "prediction": prediction,
                #"predicted_type": predicted_type,
            },
        }
    else:
        response = {
            "message": "Model not found",
            "status-code": HTTPStatus.BAD_REQUEST,
        }
    return response


header = "timestamp,project_name,experiment_id,run_id,duration,emissions,emissions_rate,cpu_power,gpu_power,ram_power,cpu_energy,gpu_energy,ram_energy,energy_consumed,country_name,country_iso_code,region,cloud_provider,cloud_region,os,python_version,cpu_count,cpu_model,gpu_count,gpu_model,longitude,latitude,ram_total_size,tracking_mode,on_cloud"
example = "2022-11-26T10:32:27,codecarbon,cc2e23fa-52a8-4ea3-a4dc-f039451bcdc4,0.871192216873169,4.1067831054495705e-07,0.0004713980480897,7.5,0.0,1.436851501464844,1.8141875664393104e-06,0,3.472772259025685e-07,2.161464792341879e-06,Spain,ESP,catalonia,,,Linux-5.15.0-53-generic-x86_64-with-glibc2.35,3.10.6,4,AMD Ryzen 5 3500U with Radeon Vega Mobile Gfx,,,2.2586,41.9272,3.83160400390625,machine,N"

@app.get("/results", responses={200: {"description": "CSV file containing all of the information collected from each inference call made until now.", 
                                     "content": {"text/csv": {"example": header + "\n" + example}}
                                     }
                                }
        )
def results(file: str = "emissions.csv"):
    file_path = file
    if os.path.exists(file_path):
        return FileResponse(file_path, media_type="text/csv")
    
    
    return {"error" : "File not found!"}


