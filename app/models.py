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

"""
# External
from codecarbon import track_emissions
from enum import Enum

# Required to run CNN model
#import tensorflow as tf
#import tensorflow.keras as keras
import numpy as np
import matplotlib.pyplot as plt
import random


# ---

from torch.nn import functional as F
import torch

# [bert]
from transformers import BertTokenizer, BertForMaskedLM
# [t5, codet5p_220m]
from transformers import T5Tokenizer, T5ForConditionalGeneration
# [codegen]
from transformers import AutoModelForCausalLM, AutoTokenizer
# [pythia-70m]
from transformers import GPTNeoXForCausalLM, AutoTokenizer

from transformers import T5ForConditionalGeneration, AutoTokenizer

# metrics

# Constants
RESULTS_DIR = 'results/'

class ML_task(Enum):
    MLM = 1 # Masked Language Modeling
    TRANSLATION = 2
    CV = 3 # Computer Vision
    CODE = 4
    
class models_names(Enum):
    BERT = 1
    T5 = 2
    CodeGen = 3
    CNN = 4
    Pythia_70m = 5
    Codet5p_220m = 6
    

class Model:
    """
    Creates a default model
    """
    def __init__(self, model_name : models_names = None, ml_task : ML_task = None):
        self.name = model_name.name
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
        

class LMBERTModel(Model):
    """
    Creates a LM Bert model. Inherits from Model()
    """
    def __init__(self):
        super().__init__(models_names.BERT, ML_task.MLM)
        
    def predict(self, user_input: str, n = 5):
        #user_text = input('Enter text with [MASK]: ')
        tokenizer = BertTokenizer.from_pretrained(pretrained_model_name_or_path = 'bert-base-uncased') # ./bert-base-uncased
        # bert model for masked language modelling
        model = BertForMaskedLM.from_pretrained(pretrained_model_name_or_path = 'bert-base-uncased',    return_dict = True) # ./bert-base-uncased
        # return_dict True to use mask token
        
        response = {
            "prediction" : self.infer(user_input, model, tokenizer)
        }
        return response
    
    @track_emissions(project_name = "bert", output_file = RESULTS_DIR + "emissions_bert.csv")
    def infer(self, text: str, model, tokenizer) -> str:
        # tokenize
        input_token = tokenizer.encode_plus(text, return_tensors = "pt")
        mask_index = torch.where(input_token["input_ids"][0] == tokenizer.mask_token_id)
        
        # generate
        output = model(**input_token)
        #tokens = model.generate(**inputs)
        # decode
        logits = output.logits
        #print(logits)
       
        softmax = F.softmax(logits, dim = -1)
        #print(softmax)
        #print(f"softmax shape:{softmax.shape}")
        mask_word = softmax[0, mask_index, :]
        #print(f"mask_word softmax function result {mask_word}")
        # Get first n words
        top_n = torch.topk(mask_word, 5, dim = 1)[1][0]
        print(top_n)
        # problem with decode
        list_results = tokenizer.convert_ids_to_tokens(top_n)
        prediction = str(list_results)
        return prediction

class T5Model(Model):
    """
    Creates a T5 model. Inherits from Model()
    """
    def __init__(self):
        super().__init__(models_names.T5, ML_task.TRANSLATION)
        
    def predict(self, user_input: str):
        #user_text = input('Enter text with [MASK]: ')
        tokenizer = T5Tokenizer.from_pretrained("t5-small")
        model = T5ForConditionalGeneration.from_pretrained("t5-small", low_cpu_mem_usage=True)
        
        response = {
            "prediction" : self.infer(user_input, model, tokenizer)
        }
        return response
    
    @track_emissions(project_name = "t5", output_file = RESULTS_DIR + "emissions_t5.csv")
    def infer(self, text: str, model, tokenizer) -> str:
        # tokenize
        input_ids = tokenizer(text, return_tensors="pt").input_ids
        #inputs = tokenizer(text, return_tensors="pt")
        # generate
        outputs = model.generate(input_ids)
        #tokens = model.generate(**inputs)
        # decode
        prediction = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return prediction

    


class CodeGenModel(Model):
    """_summary_ Creates a CodeGen model. Inherits from Model()

    Args:
        Model (_type_): _description_
    """

    def __init__(self):
        super().__init__(models_names.CodeGen, ML_task.CODE)
        
    def predict(self, user_input: str):
        
        checkpoint = "Salesforce/codegen-350M-mono"

        model = AutoModelForCausalLM.from_pretrained(checkpoint, device_map = 'auto', torch_dtype = 'auto')
        tokenizer = AutoTokenizer.from_pretrained(checkpoint)

        #text = "def get_random_element(dictionary):"

        #completion = model.generate(**tokenizer(text, return_tensors="pt"))
        #completion = model.generate(**tokenizer(text, return_tensors="pt"),max_new_tokens =25)
        response = {
            "prediction" : self.infer(user_input,model,tokenizer),
        }
        
        return response

    @track_emissions(project_name = "codegen", output_file = RESULTS_DIR + "emissions_codegen.csv")
    def infer(self, text: str, model, tokenizer) -> str:
        # tokenize
        inputs = tokenizer(text, return_tensors="pt")
        # generate
        tokens = model.generate(**inputs)
        # decode
        prediction = tokenizer.decode(tokens[0])
        return prediction
    

class Pythia_70mModel(Model):
    """_summary_ Creates a Pythia model. Inherits from Model()

    Args:
        Model (_type_): _description_
    """

    def __init__(self):
        super().__init__(models_names.Pythia_70m, ML_task.CODE)
        
    def predict(self, user_input: str):
        
        model = GPTNeoXForCausalLM.from_pretrained(
        "EleutherAI/pythia-70m",
        revision="step3000",
        cache_dir="./pythia-70m/step3000",
        )

        tokenizer = AutoTokenizer.from_pretrained(
        "EleutherAI/pythia-70m",
        revision="step3000",
        cache_dir="./pythia-70m/step3000",
        )

        #text = "def get_random_element(dictionary):"
        #text = user_input
        #inputs = tokenizer(text, return_tensors="pt")
        #tokens = model.generate(**inputs)
        response = {
            "prediction" : self.infer(user_input,model,tokenizer),
        }
        
        return response

    @track_emissions(project_name = "pythia", output_file = RESULTS_DIR + "emissions_pythia.csv")
    def infer(self, text: str, model, tokenizer) -> str:
        # tokenize
        inputs = tokenizer(text, return_tensors="pt")
        # generate
        tokens = model.generate(**inputs)
        # decode
        prediction = tokenizer.decode(tokens[0])
        return prediction


class Codet5p_220mModel(Model):
    """_summary_ Creates a Codet5p_220m model. Inherits from Model()

    Args:
        Model (_type_): _description_
    """

    def __init__(self):
        super().__init__(models_names.Codet5p_220m, ML_task.CODE)
        
    def predict(self, user_input: str):
        
        checkpoint = "Salesforce/codet5p-220m"
        #device = "cpu" # for GPU usage or "cpu" for CPU usage

        tokenizer = AutoTokenizer.from_pretrained(checkpoint)
        # torch_dtype = 'auto' not implemented
        model = T5ForConditionalGeneration.from_pretrained(checkpoint, device_map = 'auto')

        #text = "def get_random_element(my_dictionary):<extra_id_0>"
        #text = user_input
        # inputs = tokenizer.encode(text, return_tensors="pt").to(device)
        # #outputs = model.generate(inputs, max_length=10,max_new_tokens = 30)
        # outputs = model.generate(inputs, max_new_tokens = 30)
        # prediction = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        response = {
            "prediction" : self.infer(user_input,model,tokenizer),
        }
        return response
    
    @track_emissions(project_name = "codet5p", output_file = RESULTS_DIR + "emissions_codet5p.csv")
    def infer(self, text: str, model, tokenizer) -> str:
        inputs = tokenizer.encode(text, return_tensors="pt").to('cpu')
        #outputs = model.generate(inputs, max_length=10,max_new_tokens = 30)
        outputs = model.generate(inputs, max_new_tokens = 30)
        prediction = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return prediction


class CNNModel(Model):
    """
    Creates a LM Bert model. Inherits from Model()
    """
    def __init__(self):
        super().__init__(models_names.CNN, ML_task.CV)
        
    def predict(self, user_input: str, n = 5):
        dataset = "fashion"
        label_names = {
            0: "T-shirt/top",
            1: "Trouser",
            2: "Pullover",
            3: "Dress",
            4: "Coat",
            5: "Sandal",
            6: "Shirt",
            7: "Sneaker",
            8: "Bag",
            9: "Ankle boot"
        }

        saved_model_dir = f"models/model_{dataset}.h5"

        # Get test set 
        fashion_mnist=keras.datasets.fashion_mnist
        (_, _), (x_test, y_test) = fashion_mnist.load_data()

        # load model
        try:
            print(saved_model_dir)
            model = keras.models.load_model(saved_model_dir)
            print("Model loaded correctly")
        except:
            print("There is a problem with the file path")
            
        def see_x_image(x,y,name=None,caption=True, save_dir="."):
            '''
            See image
            '''
            plt.figure()
            
            plt.imshow((x.reshape((28,28))).astype("uint8"))
            title=str(y)
            if name:
                title += " "+name
                plt.title(title)
            if caption:
                plt.title(title)
            print(save_dir)
            plt.savefig(save_dir+"/"+dataset+"_image"+ ".png")
            plt.axis("off")
        
        if int(user_input) <= len(x_test):
            ran = int(user_input)
            print(" User entered ",user_input)
        else:
        # Get random number between 0 and len(x_test)
            ran = random.randint(0, len(y_test))
            print("Using random")
            print(ran)
        print(f"label of selected input: {y_test[ran]}")
        #print(list(y_test[ran]).index(max(y_test[ran])))
        label_name = label_names[y_test[ran]]
        see_x_image(x_test[ran],y_test[ran],label_name,save_dir="./")

        # Inference
        # predict with that random
        x_test = x_test.reshape(-1, 28, 28, 1)
        print(x_test[ran:ran+1].shape)
        
        #model_predict = model.predict(x_test[ran:ran+1])
        #print(f"model_predict: {model_predict}")
        #cat_pred = np.argmax(model_predict)
        cat_pred = self.infer(x_test[ran:ran+1], model)
        
        print(f"argmax: {cat_pred}")

        label = f"{cat_pred} : {label_names[cat_pred]}"
        is_correct = False
        if y_test[ran] == cat_pred:
            is_correct = True
        
        print("Prediction: ",cat_pred)
        print("Prediction clothes: ", label_names[cat_pred])
        print("Correct label: ",y_test[ran])
        print(f"is_correct: ", is_correct)
            
        response = {
            "prediction": label,
            "is_correct": is_correct,
            #"inference": self.infer(x_test[ran:ran+1], model),
        }
        return response
    
    @track_emissions(project_name = "cnn", output_file = RESULTS_DIR + "emissions_cnn.csv")
    def infer(self, user_input, model):
        model_predict = model.predict(user_input)
        #print(f"model_predict: {model_predict}")
        #cat_pred = np.argmax(model_predict)
        #print(f"argmax: {cat_pred}")
        return np.argmax(model_predict)
        