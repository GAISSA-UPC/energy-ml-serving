#!/bin/bash

pkill -f repeat.sh
pkill -f runall_update.sh
pkill -f python3
pkill -f '/home/fjdur/EnergiBridge/target/release/energibridge'

ps aux | grep runall
ps aux | grep energibridge
ps aux | grep python