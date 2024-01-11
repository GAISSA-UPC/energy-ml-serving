#!/bin/bash
sudo apt update
sudo apt upgrade
sudo apt install python3
sudo apt install uvicorn
sudo apt install pip
curl -O https://bootstrap.pypa.io/get-pip.py
python3 get-pip.py
python3 -m pip install torch