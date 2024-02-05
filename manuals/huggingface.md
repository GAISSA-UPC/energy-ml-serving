# Useful notes to use Hugging Face libraries

## Inference


## Working with torchscript

- https://huggingface.co/transformers/v1.0.0/torchscript.html
- When doing inference using a loaded .pt model, according to the input length it will be the output. 
- padding to specific length: max_length=20,padding='max_length'
        inputs = tokenizer.encode_plus(input_text, return_tensors = 'pt',max_length=20,padding='max_length')

https://discuss.huggingface.co/t/generate-without-using-the-generate-method/11379/6
https://huggingface.co/docs/transformers/en/pad_truncation

- In most cases, padding your batch to the length of the longest sequence and truncating to the maximum length a model can accept works pretty well.
