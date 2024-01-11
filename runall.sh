python -m pip install --upgrade pip
python -m pip install -r requirements.txt
uvicorn app.api_code:app  --host 0.0.0.0 --port 8000 &
python testing/main.py -i onnx -r 1