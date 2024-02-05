#!/bin/bash
echo 'Starting server using uvicorn, API for code LMs'
uvicorn app.api_code:app  --host 0.0.0.0 --port 8000  --reload   --reload-dir app 