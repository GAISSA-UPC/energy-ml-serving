#!/bin/bash

energibridge="/d/GAISSA/EnergiBridge-main/target/release/energibridge.exe"

$energibridge -h

echo "_________________________"
$energibridge --max-execution 0 --output energy_bridge.csv --command-output output.log --interval 200 --summary python example.py
echo "_________________________"
echo "program finished"
