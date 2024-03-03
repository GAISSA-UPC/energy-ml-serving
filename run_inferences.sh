#!/bin/bash
# REPS=1
# python3 testing/main.py -i ov -r $REPS | tee -a results/out_ov.log;
# python3 testing/main.py -i onnx -r $REPS | tee -a results/out_onnx.log;
# python3 testing/main.py -i torchscript -r $REPS | tee -a results/out_torchscript.log;
# python3 testing/main.py -i torch -r $REPS | tee -a results/out_torch.log;

# python3 testing/main.py -i ov -r 10 | tee -a results/out_ov.log;
# python3 testing/main.py -i onnx -r 10 | tee -a results/out_onnx.log;
# python3 testing/main.py -i torchscript -r 10 | tee -a results/out_torchscript.log;
# python3 testing/main.py -i torch -r 10 | tee -a results/out_torch.log;

# python3 testing/main.py -i ov -r 3 | tee -a results/out_ov.log;
# python3 testing/main.py -i onnx -r 3 | tee -a results/out_onnx.log;
# python3 testing/main.py -i torchscript -r 3 | tee -a results/out_torchscript.log;
# python3 testing/main.py -i torch -r 3 | tee -a results/out_torch.log;

#MODELS = [ 'codet5-base', 'codeparrot-small', 'pythia-410m', 'codet5p-220']  #'gpt-neo-125m', 'codet5p-220'

python3 testing/main.py -i ov -r 1 -m 'codet5-base' | tee -a results/out_ov.log;
echo "________________________________________" | tee -a results/out_ov.log;
python3 testing/main.py -i ov -r 1 -m 'codet5p-220' | tee -a results/out_ov.log;
echo "________________________________________" | tee -a results/out_ov.log;
python3 testing/main.py -i ov -r 1 -m 'codeparrot-small' | tee -a results/out_ov.log;
echo "________________________________________" | tee -a results/out_ov.log;
python3 testing/main.py -i ov -r 1 -m 'pythia-410m' | tee -a results/out_ov.log;
echo "________________________________________" | tee -a results/out_ov.log;


python3 testing/main.py -i onnx -r 1 -m 'codet5-base' | tee -a results/out_onnx.log;
python3 testing/main.py -i onnx -r 1 -m 'codet5p-220' | tee -a results/out_onnx.log;
python3 testing/main.py -i onnx -r 1 -m 'codeparrot-small' | tee -a results/out_onnx.log;
python3 testing/main.py -i onnx -r 1 -m 'pythia-410m' | tee -a results/out_onnx.log;


python3 testing/main.py -i torchscript -r 1 -m 'codet5-base' | tee -a results/out_torchscript.log;
python3 testing/main.py -i torchscript -r 1 -m 'codet5p-220' | tee -a results/out_torchscript.log;
python3 testing/main.py -i torchscript -r 1 -m 'codeparrot-small' | tee -a results/out_torchscript.log;
python3 testing/main.py -i torchscript -r 1 -m 'pythia-410m' | tee -a results/out_torchscript.log;


python3 testing/main.py -i torch -r 1 -m 'codet5-base' | tee -a results/out_torch.log;
echo "________________________________________" | tee -a results/out_torch.log;
python3 testing/main.py -i torch -r 1 -m 'codet5p-220' | tee -a results/out_torch.log;
echo "________________________________________" | tee -a results/out_torch.log;
python3 testing/main.py -i torch -r 1 -m 'codeparrot-small' | tee -a results/out_torch.log;
echo "________________________________________" | tee -a results/out_torch.log;
python3 testing/main.py -i torch -r 1 -m 'pythia-410m' | tee -a results/out_torch.log;
echo "________________________________________" | tee -a results/out_torch.log;
