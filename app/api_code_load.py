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
from app.schemas_code import PredictTinyllama, PredictSLM
#from transformers import pipeline

# Local modules
#from app.models import LMBERTModel, Model, T5Model, CNNModel, CodeGenModel, Pythia_70mModel, Codet5p_220mModel
from app.models_code_load import  CodeT5_BaseModel, Codet5p_220mModel, CodeGen_350mModel, GPTNeo_125m, CodeParrot_smallModel, Pythia_410mModel
from app.models_code_load import SLMModel

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
    start_time = datetime.now()
    #print("Loading started at:", start_time)

    global loaded_model, loaded_tokenizer
    try:
        # Load the model and tokenizer
        print(f'Runtime engine: {engine}')
        if engine not in ['onnx','ov','torchscript']:
            selected_class = info_models[engine][model_name]["m_class"]
            model_dir = info_models[engine][model_name]["model_dir"]
            tokenizer_dir = info_models[engine][model_name]["tokenizer_dir"]

            print(f"class: {selected_class} - model_dir {model_dir}")
            loaded_tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir) #.to(device)
            loaded_model = selected_class.from_pretrained(model_dir,) #.to(device) #  provider= exec_provider not use when using directly torch
            print("---- torch ------")

        elif engine == 'onnx':
            selected_class = info_models[engine][model_name]["m_class"]
            model_dir = info_models[engine][model_name]["model_dir"]
            tokenizer_dir = info_models['onnx'][model_name]["tokenizer_dir"] # using same than onnx

            print(f"class: {selected_class} - model_dir {model_dir}")
            if device.startswith('cuda'):
                loaded_tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir) #.to(device)
            else:
                loaded_tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir) #.to(device)
            #model = ORTModelForSeq2SeqLM.from_pretrained(model_dir,    return_dict = True, use_cache = True)
            # different configurations: try other if problems            
            if model_name in ['codeparrot-small','pythia-410m','tinyllama','pythia1-4b',
                              'phi2','slm','bloomz-560m', 'stablecode-3b', 'tiny_starcoder',
                              'codegemma-2b', 'starcoderbase-1b','bloomz-1b1','stablecode-3b-completion']: # [ADD]
                loaded_model = selected_class.from_pretrained(model_dir, provider=exec_provider)
            else:
                loaded_model = selected_class.from_pretrained(model_dir,    return_dict = True, use_cache = False, provider=exec_provider, use_io_binding=False)
            
            print("---- onnx ------")
        elif engine == 'ov':
            # model_dir = 'models/ov/codet5-base'
            # loaded_tokenizer = AutoTokenizer.from_pretrained('models/onnx/codet5-base') 
            # loaded_model = OVModelForSeq2SeqLM.from_pretrained(model_dir,    return_dict = True, use_cache = False)
            selected_class = info_models[engine][model_name]["m_class"]
            model_dir = info_models[engine][model_name]["model_dir"]
            tokenizer_dir = info_models['onnx'][model_name]["tokenizer_dir"] # using same than onnx

            print(f"class: {selected_class} - model_dir {model_dir} - model_name {model_name} - device {device}")
            loaded_tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir) 
            
            # different configurations: try other if problems
            if model_name =='codeparrot-small':
                loaded_model = selected_class.from_pretrained(model_dir, ov_config={"CACHE_DIR":""}, provider=exec_provider)
            elif model_name == ['tinyllama','slm']: #
                loaded_model = selected_class.from_pretrained(model_dir, provider=exec_provider)
            elif model_name == ['pythia1-4b','phi2','pythia-410m','tiny_starcoder',
                                'codegemma-2b', 'starcoderbase-1b','bloomz-1b1', 'stablecode-3b-completion']: # [ADD]
                # check if model in (below), this line is wrong
                loaded_model = selected_class.from_pretrained(model_dir, provider=exec_provider,use_cache=True)
            elif model_name in ['tiny_starcoder',]: # [ADD]
                print(" -> use_cache=True")
                loaded_model = selected_class.from_pretrained(model_dir, provider=exec_provider,use_cache=True)
            else:
                loaded_model = selected_class.from_pretrained(model_dir,return_dict = True, use_cache = False, use_io_binding=False, provider=exec_provider)
        elif engine == 'torchscript':
            #selected_class = info_models[engine][model_name]["m_class"]
            model_dir = info_models[engine][model_name]["model_dir"]
            tokenizer_dir = info_models['onnx'][model_name]["tokenizer_dir"] # using same than onnx
            loaded_tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir) 
            # Load the TorchScript model
            loaded_model = torch.jit.load(model_dir)
            #print(loaded_model.code)
            if exec_provider == 'CUDAExecutionProvider': loaded_model = loaded_model.to(device)
            loaded_model.eval()  # Set the model to evaluation mode, turn off gradients computation
            #decorator_to_use = self.track_torchscript
            print("---- torchscript ------")
            
        
        print(f"model_dir: {model_dir}")
        #print(f"tokenizer: {loaded_tokenizer}")
        print(f"EP and R using: {device}:{exec_provider}")
        end_time = datetime.now()
        #print("Loading finished at:", end_time)
        with open('results/load_times.csv', mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([start_time, end_time, engine, model_name])
        
        return {"message": f"Model {model_name} loaded successfully from {info_models[engine][model_name]["model_dir"]}"}
        
    except Exception as e:
        print(f"Error : {e}")
        print(f"----------------------------------------------------------")
        raise HTTPException(status_code=500, detail=str(e))


# not updated
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


# not updated
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


# not updated
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



@app.post("/huggingface_models/slm/{engine}", tags=["Hugging Face Models"])
@construct_response
def _predict_slm(request: Request, payload: PredictSLM, engine: str = None):
    """SLM model."""
    
    input_text = payload.input_text 
    print("Input text")
    print(input_text)
    
    model = SLMModel()
    print(f"Model: {model.name}")

    if input_text:
        prediction = model.predict(input_text, engine, loaded_model, loaded_tokenizer)
        
        response = {
            "message": HTTPStatus.OK.phrase,
            "status-code": HTTPStatus.OK,
            "data": {
                "model-type": model.name,
                "input_text": input_text,
                "prediction": prediction,
            },
        }
    else:
        response = {
            "message": "Model not found",
            "status-code": HTTPStatus.BAD_REQUEST,
        }
    return response


@app.post("/huggingface_models/pythia1-4b/{engine}", tags=["Hugging Face Models"])
@construct_response
def _predict_pythia1_4b(request: Request, payload: PredictSLM, engine: str = None):
    """SLM model."""
    
    input_text = payload.input_text 
    print("Input text")
    print(input_text)
    
    model = SLMModel()
    model.name='pythia1-4b'
    print(f"Model: {model.name}")

    if input_text:
        prediction = model.predict(input_text, engine, loaded_model, loaded_tokenizer)
        
        response = {
            "message": HTTPStatus.OK.phrase,
            "status-code": HTTPStatus.OK,
            "data": {
                "model-type": model.name,
                "input_text": input_text,
                "prediction": prediction,
            },
        }
    else:
        response = {
            "message": "Model not found",
            "status-code": HTTPStatus.BAD_REQUEST,
        }
    return response


@app.post("/huggingface_models/phi2/{engine}", tags=["Hugging Face Models"])
@construct_response
def _predict_phi2(request: Request, payload: PredictSLM, engine: str = None):
    """SLM model."""
    
    input_text = payload.input_text 
    print("Input text")
    print(input_text)
    
    model = SLMModel()
    model.name='phi2'
    print(f"Model: {model.name}")

    if input_text:
        prediction = model.predict(input_text, engine, loaded_model, loaded_tokenizer)
        
        response = {
            "message": HTTPStatus.OK.phrase,
            "status-code": HTTPStatus.OK,
            "data": {
                "model-type": model.name,
                "input_text": input_text,
                "prediction": prediction,
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
def _predict_tinyllama(request: Request, payload: PredictSLM, engine: str = None):
    """SLM model."""
    
    input_text = payload.input_text 
    print("Input text")
    print(input_text)
    
    model = SLMModel()
    model.name='tinyllama'
    print(f"Model: {model.name}")

    if input_text:
        prediction = model.predict(input_text, engine, loaded_model, loaded_tokenizer)
        
        response = {
            "message": HTTPStatus.OK.phrase,
            "status-code": HTTPStatus.OK,
            "data": {
                "model-type": model.name,
                "input_text": input_text,
                "prediction": prediction,
            },
        }
    else:
        response = {
            "message": "Model not found",
            "status-code": HTTPStatus.BAD_REQUEST,
        }
    return response


@app.post("/huggingface_models/bloomz-560m/{engine}", tags=["Hugging Face Models"])
@construct_response
def _predict_bloomz_560m(request: Request, payload: PredictSLM, engine: str = None):
    """SLM model."""
    
    input_text = payload.input_text 
    print("Input text")
    print(input_text)
    
    model = SLMModel()
    model.name='bloomz-560m'
    print(f"Model: {model.name}")

    if input_text:
        prediction = model.predict(input_text, engine, loaded_model, loaded_tokenizer)
        
        response = {
            "message": HTTPStatus.OK.phrase,
            "status-code": HTTPStatus.OK,
            "data": {
                "model-type": model.name,
                "input_text": input_text,
                "prediction": prediction,
            },
        }
    else:
        response = {
            "message": "Model not found",
            "status-code": HTTPStatus.BAD_REQUEST,
        }
    return response


@app.post("/huggingface_models/stablecode-3b/{engine}", tags=["Hugging Face Models"])
@construct_response
def _predict_stablecode_3b(request: Request, payload: PredictSLM, engine: str = None):
    """SLM model."""
    
    input_text = payload.input_text 
    print("Input text")
    print(input_text)
    
    model = SLMModel()
    model.name='stablecode-3b'
    print(f"Model: {model.name}")

    if input_text:
        prediction = model.predict(input_text, engine, loaded_model, loaded_tokenizer)
        
        response = {
            "message": HTTPStatus.OK.phrase,
            "status-code": HTTPStatus.OK,
            "data": {
                "model-type": model.name,
                "input_text": input_text,
                "prediction": prediction,
            },
        }
    else:
        response = {
            "message": "Model not found",
            "status-code": HTTPStatus.BAD_REQUEST,
        }
    return response


@app.post("/huggingface_models/{model_name}/{engine}", tags=["Hugging Face Models"])
@construct_response
def predict_model(request: Request, payload: PredictSLM, model_name: str, engine: str = None):
    """Generic endpoint for Hugging Face Models."""
    
    input_text = payload.input_text
    print("Input text")
    print(input_text)
    
    # Initialize the model
    model = SLMModel()
    model.name = model_name
    print(f"Model: {model.name}")

    # Check if input_text is provided
    if input_text:
        # Generate prediction
        prediction = model.predict(input_text, engine, loaded_model, loaded_tokenizer)
        
        # Construct the response
        response = {
            "message": HTTPStatus.OK.phrase,
            "status-code": HTTPStatus.OK,
            "data": {
                "model-type": model.name,
                "input_text": input_text,
                "prediction": prediction,
            },
        }
    else:
        # Handle missing input_text
        response = {
            "message": "Input text not provided",
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


