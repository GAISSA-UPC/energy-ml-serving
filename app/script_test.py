import json
import requests

response = requests.get("http://localhost:8000/")
print (json.loads(response.text))