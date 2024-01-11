#!/bin/bash
python3 testing/main.py -i ov -r 1 | tee -a results/out_ov.log;
python3 testing/main.py -i ov -r 10 | tee -a results/out_ov.log;
python3 testing/main.py -i onnx -r 1 | tee -a results/out_onnx.log;
python3 testing/main.py -i onnx -r 10 | tee -a results/out_onnx.log;
python3 testing/main.py -i torchscript -r 1 | tee -a results/out_torchscript.log;
python3 testing/main.py -i torchscript -r 10 | tee -a results/out_torchscript.log;
python3 testing/main.py -i torch -r 1 | tee -a results/out_torch.log;
python3 testing/main.py -i torch -r 10 | tee -a results/out_torch.log;