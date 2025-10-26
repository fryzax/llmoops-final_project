import requests
from requests.auth import HTTPBasicAuth
import os


PUBLIC_KEY = os.getenv("LANGFUSE_PUBLIC_KEY")
SECRET_KEY = os.getenv("LANGFUSE_SECRET_KEY")

url = "https://cloud.langfuse.com/api/public/v2/dashboards/widgets"

payload = {
    "view": "traces",
    "metric": "attributes.article_word_count",
    "aggregation": "avg",
    "filters": [
        {
            "field": "trace.name",
            "operator": "equals",
            "value": "vertex_ai_summarization"
        }
    ],
    "visualization": "number",
    "name": "Article Word Count (Vertex AI)"
}   

response = requests.post(
    url,
    json=payload,
    auth=HTTPBasicAuth(PUBLIC_KEY, SECRET_KEY)
)

print(response.status_code)
print(response.json())
