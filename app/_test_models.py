""" Tests for models

ToDo:
- Use pytest
- Test Cases
    - Each model must have predict method and return a dictionary
    - Generation models should return longer text
"""
#from models import LMBERTModel, T5Model, CNNModel, CodeGenModel, Pythia_70mModel, Codet5p_220mModel
from models_code import  CodeT5_BaseModel, Codet5p_220mModel, CodeGen_350mModel, GPTNeo_125m, CodeParrot_smallModel, Pythia_410mModel


examples = {
    "BERT" : [
        "I am from [MASK].",
    ],
    "T5" : [
        "translate English to German: Hello, how are you?",
    ],
    "CodeGen" : [
        "def get_random_element(dictionary):",
    ],
    "Pythia_70m" : [
        "def get_random_element(dictionary):",
    ],
    "Codet5p_220m" : [
        "def get_random_element(my_dictionary):<extra_id_0>",
    ],
    "CNN" : [
        "101233",
    ],
    "code_default" : [
      "def hello_world():",
    ],
    
}

#model_classes = [LMBERTModel, T5Model, CNNModel, CodeGenModel,  Pythia_70mModel, Codet5p_220mModel]
model_classes = [CodeT5_BaseModel, Codet5p_220mModel, CodeGen_350mModel, GPTNeo_125m, CodeParrot_smallModel, Pythia_410mModel]

#model_classes = model_classes[-1:]

code_models = True

for class_model in model_classes:
    try:
        instance_model = class_model()
        print(f"Model: {instance_model.name}")
        if examples[instance_model.name]:
            model_response = instance_model.predict(examples[instance_model.name][0])
        elif code_models:
            model_response = instance_model.predict(examples["code_default"][0])
        print(f"model response: {model_response}")
        assert isinstance(model_response,dict)
        assert model_response['prediction'] is not None
        assert isinstance(model_response['prediction'], str)
        print(model_response)
    except Exception as e:
        print(f"Exception: {e}")
    print("====================================================")
    
    
