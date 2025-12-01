import requests
import base64
import numpy as np

with open("image1.jpg", "rb") as f:
    img = base64.b64encode(f.read()).decode()

payload = {
    "id": "test123",
    "inputs": [
        {
            "name": "image",
            "datatype": "BYTES",
            "shape": [1],
            "data": [img]
        }
    ]
}

response = requests.post(
    "http://localhost:8080/v2/models/img2vec/infer",
    json=payload
)

vector = np.array(response.json()["outputs"][0]["data"])
print(vector)