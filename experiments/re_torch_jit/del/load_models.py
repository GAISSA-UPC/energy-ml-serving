from transformers import BertModel, BertTokenizer, BertConfig
import torch
#import torch.neuron


enc = BertTokenizer.from_pretrained("bert-base-uncased")

# Tokenizing input text
text = "[CLS] Who was Jim Henson ? [SEP] Jim Henson was a puppeteer [SEP]"
tokenized_text = enc.tokenize(text)

# Masking one of the input tokens
masked_index = 8
tokenized_text[masked_index] = "[MASK]"
indexed_tokens = enc.convert_tokens_to_ids(tokenized_text)
segments_ids = [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1]

# Creating a dummy input
tokens_tensor = torch.tensor([indexed_tokens])
segments_tensors = torch.tensor([segments_ids])
dummy_input = [tokens_tensor, segments_tensors]



loaded_model = torch.jit.load("traced_bert.pt")
loaded_model.eval()

#all_encoder_layers, pooled_output = loaded_model(*dummy_input)

res = loaded_model(tokens_tensor, segments_tensors)

print(res)
print(type(res))
print(res[0].shape)
print(len(res))

input_token = enc(res[0], return_tensors="pt").input_ids

print(input_token)

#torch.jit.trace(model, [tokens_tensor, segments_tensors])
#torch.neuron.trace(model, [token_tensor, segments_tensors])