#!/bin/bash
REPS=1
python3 testing/main.py -i ov -r $REPS | tee -a results/out_ov.log;
python3 testing/main.py -i onnx -r $REPS | tee -a results/out_onnx.log;
python3 testing/main.py -i torchscript -r $REPS | tee -a results/out_torchscript.log;
python3 testing/main.py -i torch -r $REPS | tee -a results/out_torch.log;