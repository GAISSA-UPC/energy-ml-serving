#!/bin/bash

energibridge="/d/GAISSA/EnergiBridge-main/target/release/energibridge.exe" # windows
energibridge="/home/fjdur/EnergiBridge/target/release/energibridge" # linux

python3=python3

$energibridge -h

echo "_________________________"
$energibridge --max-execution 0 --output energy_bridge.csv --command-output output.log --interval 200 --summary $python3 example.py
echo "_________________________"
