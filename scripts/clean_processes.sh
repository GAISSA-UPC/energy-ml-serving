#!/bin/bash

pkill -f repeat.sh
pkill -f runall_update.sh
pkill -f python3
pkill -f '/home/fjdur/EnergiBridge/target/release/energibridge'
pkill -f nvidia-smi
pkill -f runall
pkill -f energibridge
pkill -f python



ps aux | grep runall
ps aux | grep energibridge
ps aux | grep python
ps aux | grep nvidia-smi