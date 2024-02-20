import uvicorn

# async def app(scope, receive, send):
#     ...

if __name__ == "__main__":
    # uvicorn app.api_code:app  --host 0.0.0.0 --port 8000  #--reload   --reload-dir app 
    uvicorn.run("app.api_code:app", port=8000, log_level="info")