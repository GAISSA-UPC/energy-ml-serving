**ToDo:**

- [x] Add all problems in this directory
- [x] Organize
- [ ] Update if new problems

-------------------
- [FAQ and issues arised during deployment](#faq-and-issues-arised-during-deployment)
  - [Hugging Face issues](#hugging-face-issues)
    - [low\_cpu\_mem\_usage](#low_cpu_mem_usage)
    - [How to use pretrained Huggingface model](#how-to-use-pretrained-huggingface-model)
  - [API issues](#api-issues)
    - [How can I open a swagger UI that is in a remote server, in which I can connect by ssh?](#how-can-i-open-a-swagger-ui-that-is-in-a-remote-server-in-which-i-can-connect-by-ssh)
  - [Using python modules in old or small CPU](#using-python-modules-in-old-or-small-cpu)
  - [Not enough disk space](#not-enough-disk-space)
  - [How to emulate Linux in Windows?](#how-to-emulate-linux-in-windows)
  - [New problem or question](#new-problem-or-question)

# FAQ and issues arised during deployment

## Hugging Face issues
### low_cpu_mem_usage

- Error message:
  - ```RuntimeError: Tensor on device meta is not on the expected device cpu!```
  - - ```ValueError: BertForMaskedLM does not support `device_map='auto'` yet```
torch1.* has the problem with  `low_cpu_mem_usage=True`
- This error seems to arise when enabling the argument `low_cpu_mem_usage=True` from `HuggingFaceModel.from_pretrained(model_name, low_cpu_mem_usage=True)` and this feature is not implemented for the model.
- If `low_cpu_mem_usage==True` means it will try to use no more than 1x of the maximum memory usage.
- If `HuggingFaceModel.from_pretrained()` has implemented the parameter `device_map='auto'`, then it automatically sets `low_cpu_mem_usage=True` to reduce the memory usage
  - By passing `device_map="auto"`, we tell Accelerate to determine automatically where to put each layer of the model depending on the available resources:
- When using the Bert model, you can see that these options are not implemented. On the other hand, for T5 model, they are implemented. (See how it is used in [models.py](../app/models.py)).
- More info:
  - https://huggingface.co/docs/transformers/main_classes/model
  - no_split_module_classes (`List[str]`): A list of class names for layers we don't want to be split
  - https://github.com/huggingface/transformers/issues/23086
- Example to raise the error 
  - ```model = BertForMaskedLM.from_pretrained(pretrained_model_name_or_path = './bert-base-uncased', return_dict = True, device_map="auto")```
- t5, _no_split_modules
  - https://github.com/huggingface/transformers/blob/main/src/transformers/models/t5/modeling_t5.py#L785
- bert
  - https://github.com/huggingface/transformers/blob/main/src/transformers/models/bert/modeling_bert.py
  - does not have that attribute
- condition that prevents bert model of using device_map
  - https://github.com/huggingface/transformers/blob/main/src/transformers/modeling_utils.py#L2691


### How to use pretrained Huggingface model 

1. Install Hugging Face "transformers" module
2. Load pre-trained model
3. Load pretrained model
```
from transformers import AutoModel

model = AutoModel.from_pretrained('bert-base-uncased')
```
4. Tokenize the input text:
```
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
input_text = "Hello, world!"
tokenized_input = tokenizer(input_text, return_tensors='pt')

```
5. Run the model
```
# Run model
outputs = model(**tokenized_input)
```

--------------------
## API issues
### How can I open a swagger UI that is in a remote server, in which I can connect by ssh?

- run server in remote machine
Create SSH tunnel to access endpoints, forwards traffic from my local port XXXX to the remote server's port XXXX
- It has to be running

```
ssh -L 8080:localhost:8080 myusername@123.45.67.89
ssh -L 8000:localhost:8000  alumne@10.4.41.62
```
--------------------

## Using python modules in old or small CPU
- Cloud providers with free-tier VMs which had this problem: 
  - Virtech
- Errors:
  - Illegal instruction (core dumped)
- Some CPUs are not able to load some modules such as
  - transformers
    - `from transformers import pipeline` 
  - tensorflow
    - `import tensorflow` 
- Not able to load pipeline
  - `from transformers import pipeline` 
- In brief, the error will be thrown if we’re running recent TensorFlow binaries on CPU(s) that do not support Advanced Vector Extensions (AVX), an instruction set that enables faster
computation especially for vector operations. Starting from TensorFlow 1.6, pre-built TensorFlow binaries use AVX instructions. An except from TensorFlow 1.6 release announcement: tf 1.6 - feb 18, transformers - 19
- https://tech.amikelive.com/node-887/how-to-resolve-error-illegal-instruction-core-dumped-when-running-import-tensorflow-in-a-python-program/
- My flags
  - `flags           : fpu de pse tsc msr pae mce cx8 apic sep mtrr pge mca cmov pse36 clflush mmx fxsr sse sse2 syscall nx lm rep_good nopl xtopology cpuid tsc_known_freq pni cx16 x2apic hypervisor lahf_lm cpuid_fault pti`
- How to check your CPU features:
  - Windows
    - check system information, then search {cpu model} CPU features
  - Linux
    - `more /proc/cpuinfo | grep flags`
- See accelerators
  - https://www.intel.com/content/www/us/en/developer/articles/guide/deep-learning-with-avx512-and-dl-boost.html
  - cat /proc/cpuinfo

--------------------
## Not enough disk space

- Useful commands
  - `free -h` Display amount of free and used memory in the system
  - `df -h` Report file system disk space usage


--------------------
## How to install CUDA? problems with CUDA or cuDNN, check:

2024-11-09 14:48:41.016222360 [E:onnxruntime:Default, provider_bridge_ort.cc:1862 TryGetProviderInfo_CUDA] /onnxruntime_src/onnxruntime/core/session/provider_bridge_ort.cc:1539 onnxruntime::Provider& onnxruntime::ProviderLibrary::Get() [ONNXRuntimeError] : 1 : FAIL : Failed to load library libonnxruntime_providers_cuda.so with error: libcudnn.so.9: cannot open shared object file: No such file or directory

2024-11-09 14:48:41.016237619 [W:onnxruntime:Default, onnxruntime_pybind_state.cc:993 CreateExecutionProviderInstance] Failed to create CUDAExecutionProvider. Require cuDNN 9.* and CUDA 12.*. Please install all dependencies as mentioned in the GPU requirements page (https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html#requirements), make sure they're in the PATH, and that your GPU is supported.
[models_code_load.py]  response:<Response [200]>

Install CUDA 12
https://developer.nvidia.com/cuda-downloads
cuDNN 9
https://developer.nvidia.com/cudnn-downloads?target_os=Linux&target_arch=x86_64&Distribution=Ubuntu&target_version=24.04&target_type=deb_network





## How to emulate Linux in Windows?
- Use a command line terminal emulator: git bash
- https://gitforwindows.org/

--------------------
## New problem or question

--------------------


--------------------

