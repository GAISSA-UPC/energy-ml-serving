# Deploying codellama using runtime engine and web framework

[HuggingFace codellama](https://huggingface.co/codellama/CodeLlama-7b-hf)
https://huggingface.co/blog/codellama#conversational-instructions
## Requirements
RAM Requirements 	VRAM Requirements
6GB (Swap to Load*) 	6GB

https://discuss.huggingface.co/t/llama-7b-gpu-memory-requirement/34323
To run the 7B model in full precision, you need 7 * 4 = 28GB of GPU RAM. You should add torch_dtype=torch.float16 to use half the memory and fit the model on a T4. 

Basicly the idea is that you store the row weights (weigths are store in 16bit parameters format) and you also need to store the gradient of the weights. As 1 bytes = 8 bits, you need 2B for every weights and another 2B for the gradient. And that’s only the case if you use SGD optimization because if you use ADAM as your optimizer, you need more memory per weights.
So you ends up with a raw memory requirement of 4*nb_parameters if you use SGD.

in full precision (float32), every parameter of the model is stored in 32 bits or 4 bytes. Hence 4 bytes / parameter * 7 billion parameters = 28 billion bytes = 28 GB of GPU memory required, for inference only. In half precision, each parameter would be stored in 16 bits, or 2 bytes. Hence you would need 14 GB for inference. There are now also 8 bit and 4 bit algorithms, so with 4 bits (or half a byte) per parameter you would need 3.5 GB of memory for inference.