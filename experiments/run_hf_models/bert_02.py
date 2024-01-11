from transformers import BertTokenizer, BertForMaskedLM
from torch.nn import functional as F
import torch

#from models import *
from models import LMBERTModel, Model


def decode(tokenizer, token_ids,
           skip_special_tokens: bool = False,
           clean_up_tokenization_spaces: bool = None,
           spaces_between_special_tokens: bool = True,
           **kwargs,) -> str:
   tokenizer._decode_use_source_tokenizer = kwargs.pop("use_source_tokenizer", False)

   filtered_tokens = tokenizer.convert_ids_to_tokens(token_ids, skip_special_tokens=skip_special_tokens)

   # To avoid mixing byte-level and unicode for byte-level BPT
   # we need to build string separately for added tokens and byte-level tokens
   # cf. https://github.com/huggingface/transformers/issues/1133
   sub_texts = []
   current_sub_text = []
   for token in filtered_tokens:
      if skip_special_tokens and token in tokenizer.all_special_ids:
            continue
      if token in tokenizer.added_tokens_encoder:
            if current_sub_text:
               sub_texts.append(tokenizer.convert_tokens_to_string(current_sub_text))
               current_sub_text = []
            sub_texts.append(token)
      else:
            current_sub_text.append(token)
   if current_sub_text:
      sub_texts.append(tokenizer.convert_tokens_to_string(current_sub_text))

   if spaces_between_special_tokens:
      text = " ".join(sub_texts)
   else:
      text = "".join(sub_texts)

   clean_up_tokenization_spaces = (
      clean_up_tokenization_spaces
      if clean_up_tokenization_spaces is not None
      else tokenizer.clean_up_tokenization_spaces
   )
   if clean_up_tokenization_spaces:
      clean_text = tokenizer.clean_up_tokenization(text)
      return clean_text
   else:
      return text

testing_input = False



if testing_input:
    user_text = input('Enter text with [MASK]: ')

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
    n = 5
    top_n = torch.topk(mask_word, n, dim = 1)[1][0]
    print(top_n)
    print("Decoding...")
    # problem with decode
    list_results = tokenizer.convert_ids_to_tokens(top_n)
    print("\nResults:")
    for word in list_results:
        print(word)

testing_model = True

#user_text = input('Enter text with [MASK]: ')
user_text = 'I am working in a [MASK] company writing code. '
if testing_model:
    my_model = LMBERTModel()
    my_general_model = Model()
    
    print(my_general_model.predict(user_text))
    
    print(my_model.predict(user_text))
    