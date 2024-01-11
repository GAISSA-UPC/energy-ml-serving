from transformers import BertTokenizer, BertForMaskedLM
from transformers import T5Tokenizer, T5ForConditionalGeneration
from torch.nn import functional as F
import torch

class Model:
    """
    Creates a default model
    """
    def __init__(self, name = None, ml_task = None):
        self.name = name
        # Masked Language Modeling - MLM
        self.ml_task = ml_task
    
    def predict(self, user_input):
        # Do prediction
        prediction = "Not defined yet "
        return prediction
    
class LMBERTModel(Model):
    """
    Creates a LM Bert model. Inherits from Model()
    """
    def __init__(self):
        super().__init__('BERT', 'MLM')
        
    def predict(self, user_input: str, n = 5):
        #user_text = input('Enter text with [MASK]: ')
        user_text = user_input
        tokenizer = BertTokenizer.from_pretrained(pretrained_model_name_or_path = './bert-base-uncased')
        # bert model for masked language modelling
        model = BertForMaskedLM.from_pretrained(pretrained_model_name_or_path = './bert-base-uncased',    return_dict = True)
        # return_dict True to use mask token
        text = user_text
        input_token = tokenizer.encode_plus(text, return_tensors = "pt")
        print(f"string encoded: \n{input_token}")
        # Select index of mask
        mask_index = torch.where(input_token["input_ids"][0] == tokenizer.mask_token_id)
        print(f"Mask index: {mask_index}")
        output = model(**input_token)
        # logits are output of BERT model before softmax activation
        logits = output.logits
        #print(logits)
        print(f"logits.shape: {logits.shape}")
        softmax = F.softmax(logits, dim = -1)
        #print(softmax)
        print(f"softmax shape:{softmax.shape}")
        mask_word = softmax[0, mask_index, :]
        print(f"mask_word softmax function result {mask_word}")
        # Get first n words
        n = n
        top_n = torch.topk(mask_word, n, dim = 1)[1][0]
        print(top_n)
        print("Decoding...")
        # problem with decode
        list_results = tokenizer.convert_ids_to_tokens(top_n)
        print("\nResults:")
        for word in list_results:
            print(word)
        
        return list_results
    

class T5Model(Model):
    """
    Creates a LM Bert model. Inherits from Model()
    """
    def __init__(self):
        super().__init__('T5', 'MLM')
        
    def predict(self, user_input: str, n = 5):
        #user_text = input('Enter text with [MASK]: ')
        tokenizer = T5Tokenizer.from_pretrained("t5-small")
        model = T5ForConditionalGeneration.from_pretrained("t5-small", low_cpu_mem_usage=True)

        def infer_t5(text):
            input_ids = tokenizer(text, return_tensors="pt").input_ids
            outputs = model.generate(input_ids)

            return tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        return infer_t5(user_input)