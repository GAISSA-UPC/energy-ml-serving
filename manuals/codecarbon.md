# codecarbon
https://mlco2.github.io/codecarbon/usage.html
https://github.com/dani-kjh/TFG_replication_package/blob/main/src/app.py
https://fastapi.tiangolo.com/advanced/custom-response/
https://mlco2.github.io/codecarbon/parameters.html

## How to use codecarbon
1. Install codecarbon module
```bash
    pip install codecarbon
```

2. Using this code that makes a prediction using the T5 model

```python
from transformers import T5Tokenizer, T5ForConditionalGeneration

tokenizer = T5Tokenizer.from_pretrained("t5-small")
model = T5ForConditionalGeneration.from_pretrained("t5-small", low_cpu_mem_usage=True)

def infer_t5(text):
    input_ids = tokenizer(text, return_tensors="pt").input_ids
    outputs = model.generate(input_ids)

    return tokenizer.decode(outputs[0], skip_special_tokens=True)

text = "Hello men, this is a test to get the energy metrics of a inference using a machine learning model"

infer_t5(text)
```

3. Import the decorator and decorate the function that will be tracked

```python
from transformers import T5Tokenizer, T5ForConditionalGeneration
from codecarbon import track_emissions

tokenizer = T5Tokenizer.from_pretrained("t5-small")
model = T5ForConditionalGeneration.from_pretrained("t5-small", low_cpu_mem_usage=True)

@track_emissions
def infer_t5(text):
    input_ids = tokenizer(text, return_tensors="pt").input_ids
    outputs = model.generate(input_ids)

    return tokenizer.decode(outputs[0], skip_special_tokens=True)

text = "Hello men, this is a test to get the energy metrics of a inference using a machine learning model"

infer_t5(text)
```

4. You should get the measurements in the output:
```bash
[codecarbon INFO @ 12:48:11]   Available RAM : 15.837 GB
[codecarbon INFO @ 12:48:11]   CPU count: 8
[codecarbon INFO @ 12:48:11]   CPU model: Intel(R) Core(TM) i5-10210U CPU @ 1.60GHz
[codecarbon INFO @ 12:48:11]   GPU count: None
[codecarbon INFO @ 12:48:11]   GPU model: None
D:\GAISSA\cloud-api\lib\site-packages\transformers\generation\utils.py:1313: UserWarning: Using `max_length`'s default (20) to control the generation length. This behaviour is deprecated and will be removed from the config in v5 of Transformers -- we recommend using `max_new_tokens` to control the maximum length of the generation.
  warnings.warn(
[codecarbon INFO @ 12:48:15]
Graceful stopping: collecting and writing information.
Please wait a few seconds...
[codecarbon INFO @ 12:48:15] Energy consumed for RAM : 0.000001 kWh. RAM Power : 5.93895149230957 W
[codecarbon INFO @ 12:48:15] Energy consumed for all CPUs : 0.000002 kWh. Total CPU Power : 12.5 W
[codecarbon INFO @ 12:48:15] 0.000003 kWh of electricity used since the beginning.
[codecarbon INFO @ 12:48:15] Done!
```

## Codecarbon output
https://mlco2.github.io/codecarbon/output.html