from transformers import  BertTokenizer, BertForMaskedLM
import torch

# Load the BERT model and tokenizer
model_name = "bert-base-uncased"
model = BertForMaskedLM.from_pretrained(model_name, low_cpu_mem_usage=True, torch_dtype=torch.float16)

tokenizer = BertTokenizer.from_pretrained(model_name)

# Define the fill-mask function
def fill_mask(text):
    # Tokenize the input text
    tokenized_text = tokenizer.tokenize(text)

    # Find the position of the masked token
    mask_token_index = tokenized_text.index("[MASK]")

    # Convert tokenized text to token IDs
    input_ids = tokenizer.convert_tokens_to_ids(tokenized_text)

    # Create an attention mask
    attention_mask = [1] * len(input_ids)

    # Set the masked token index to the mask token ID
    input_ids[mask_token_index] = tokenizer.mask_token_id

    # Convert input IDs and attention mask to tensors
    input_ids = torch.tensor([input_ids])
    attention_mask = torch.tensor([attention_mask])

    # Perform forward pass through the BERT model
    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask).to('cpu')


    # Get the predicted probabilities for the masked token
    masked_token_logits = outputs.logits[0, mask_token_index]
    masked_token_probs = masked_token_logits.softmax(dim=-1)

    # Get the top-k predictions for the masked token
    top_k = 5
    top_k_indices = masked_token_probs.topk(top_k).indices.tolist()
    top_k_tokens = tokenizer.convert_ids_to_tokens(top_k_indices)

    # Return the top-k predicted tokens
    return top_k_tokens

# Use the pipeline to perform masked language modeling
text = "The capital of Mexico, " + tokenizer.mask_token + ", contains the Mayan piramid."
#text = "I love the " + tokenizer.mask_token + ", it is from Mexico."

result = fill_mask(text)
print(result)