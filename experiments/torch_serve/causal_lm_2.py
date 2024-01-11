import torch
from torch.nn import functional as F
from ts.torch_handler.base_handler import BaseHandler

from transformers import AutoTokenizer
from transformers import AutoModelForMaskedLM, AutoModelForCausalLM, AutoModelForSeq2SeqLM, T5ForConditionalGeneration


import json
import logging
import os

from codecarbon import track_emissions


RESULTS_DIR = '/home/fjdur/cloud-api/results/'


logger = logging.getLogger(__name__)


class MyHandler(BaseHandler):
    def __init__(self):
        super().__init__()
        self.initialized = False
        
    
    def initialize(self, ctx):
        
        self.manifest = ctx.manifest

        properties = ctx.system_properties
        model_dir = properties.get("model_dir")
        self.device = torch.device("cuda:" + str(properties.get("gpu_id")) if torch.cuda.is_available() else "cpu")

        # Read model serialize/pt file
        logger.info("Model_dir is: '%s'", model_dir)
        self.model = AutoModelForCausalLM.from_pretrained(model_dir)
        # change the checkpoint
        self.tokenizer = AutoTokenizer.from_pretrained('codeparrot/codeparrot-small')
        """
        model_checkpoint = {'codet5-base':"Salesforce/codet5-base", 'codet5p-220':'Salesforce/codet5p-220m', 
                    'codegen-350-mono':"Salesforce/codegen-350M-mono", 'gpt-neo-125m':"EleutherAI/gpt-neo-125M",
                    'codeparrot-small':'codeparrot/codeparrot-small', 'pythia-410m':"EleutherAI/pythia-410m"} # model:checkpoint
        """

        self.model.to(self.device)
        self.model.eval()

        logger.debug('Transformer model from path {0} loaded successfully'.format(model_dir))

        # Read the mapping file, index to object name
        mapping_file_path = os.path.join(model_dir, "index_to_name.json")

        if os.path.isfile(mapping_file_path):
            with open(mapping_file_path) as f:
                self.mapping = json.load(f)
        else:
            logger.warning('Missing the index_to_name.json file. Inference output will not include class name.')

        self.initialized = True
    
    
    def preprocess(self, data):
        """
        Preprocess function to convert the request input to a tensor(Torchserve supported format).
        The user needs to override to customize the pre-processing

        Args :
            data (list): List of the data from the request input.

        Returns:
            tensor: Returns the tensor data of the input
        """
        
        text = data[0].get("data")
        if text is None:
            text = data[0].get("body")
        sentences = text.decode('utf-8')
        print(f"Sentences,")
        logger.info("Received text: '%s'", sentences)
        #input_token = self.tokenizer(sentences, return_tensors="pt").input_ids
        
        return sentences
        
        #return torch.as_tensor(data, device=self.device)

    


    def inference(self,  data, *args, **kwargs):
        #def predict(user_input: str, n = 5):
        #user_text = input('Enter text with [MASK]: ')
        #tokenizer = BertTokenizer.from_pretrained(pretrained_model_name_or_path = 'bert-base-uncased') # ./bert-base-uncased
        #model_dir = 'models/onnx/onnx_bert/'
        #self.manifest = ctx.manifest

        #properties = ctx.system_properties
        #model_dir = properties.get("model_dir")
        #tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased') # ./bert-base-uncased
        #self.model = AutoModelForMaskedLM.from_pretrained('bert-base-uncased')
        
        #self.model.load_state_dict('models/torch/torch_bert.pth')
        logger.info("state_dict loaded: " )
        # bert model for masked language modelling
        # model = None
        # if runtime_engine == 'onnx':
        #     model = ORTModelForMaskedLM.from_pretrained(model_dir,    return_dict = True) # ./bert-base-uncased
        # elif runtime_engine == 'ov':
        #     model = OVModelForMaskedLM.from_pretrained(model_dir,    return_dict = True) # ./bert-base-uncased
        # return_dict True to use mask token
        logger.info("Received data: '%s'", data)
        
        response = {
            "prediction" : self.infer(data, self.model, self.tokenizer)
        }
        return response

    #codeparrot/codeparrot-small
    @track_emissions(project_name = "codeparrot-small_ts", output_file = RESULTS_DIR + "emissions_codeparrot-small.csv")
    def infer(self, text: str, model, tokenizer) -> str:
        # tokenize
        #input_ids = text

        input_ids = self.tokenizer(text, return_tensors="pt").input_ids

        # generate
        outputs = None
        
        with torch.no_grad():
            outputs = model.generate(input_ids)
        #tokens = model.generate(**inputs)
        # decode
        prediction = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return prediction
    
        
    def postprocess(self, data):
        """
        The post process function makes use of the output from the inference and converts into a
        Torchserve supported response output.

        Args:
            data (Torch Tensor): The torch tensor received from the prediction output of the model.

        Returns:
            List: The post process function returns a list of the predicted output.
        """
        logger.info("Received data: '%s'", data)
        logger.info("Received data: '%s'", data['prediction'])
        return [data['prediction']] #  model: bert, number of batch response mismatched, expect: 1, got: 5.

_service = MyHandler()


def handle(data, context):
    try:
        if not _service.initialized:
            _service.initialize(context)

        if data is None:
            return None

        data = _service.preprocess(data)
        data = _service.inference(data)
        data = _service.postprocess(data)

        return data
    except Exception as e:
        raise e