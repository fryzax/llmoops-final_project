import requests
from requests.auth import HTTPBasicAuth
import os
PUBLIC_KEY = os.getenv("LANGFUSE_PUBLIC_KEY")
SECRET_KEY = os.getenv("LANGFUSE_SECRET_KEY")

url = "https://cloud.langfuse.com/api/public/v2/traces"

response = requests.get(url, auth=HTTPBasicAuth(PUBLIC_KEY, SECRET_KEY))
print("Status:", response.status_code)
print("Body:", response.text)  

data = response.json()

for trace in data.get("data", []):
    print(trace.get("name"), trace.get("attributes"))
