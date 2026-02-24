import requests
import base64
import numpy as np

def get_mlserver_version():
    try:
        response = requests.get("http://localhost:9600/v2")
        if response.status_code == 200:
            data = response.json()
            return data.get("version")
        return None
    except Exception as e:
        print(f"Failed to get MLServer version: {e}")
        return None

mlserver_version = get_mlserver_version()
if mlserver_version:
    print(f"Connected to MLServer version: {mlserver_version}")
else:
    print("Could not retrieve MLServer version")

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
    "http://localhost:9600/v2/models/img2vec/infer",
    json=payload
)

vector = np.array(response.json()["outputs"][0]["data"])
version = response.json()["model_version"]
print(vector)
print(version)