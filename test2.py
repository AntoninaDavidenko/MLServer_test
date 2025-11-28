import requests
import base64
from sklearn.metrics.pairwise import cosine_similarity

def get_vector(image_path):
    with open(image_path, "rb") as f:
        img_b64 = base64.b64encode(f.read()).decode()
    payload = {
        "id": "test123",
        "inputs": [
            {
                "name": "image",
                "datatype": "BYTES",
                "shape": [1],
                "data": [img_b64]
            }
        ]
    }
    response = requests.post("http://localhost:8080/v2/models/img2vec/infer", json=payload)
    vector = response.json()["outputs"][0]["data"]
    return vector

vec1 = get_vector("image1.jpg")
vec2 = get_vector("image2.jpg")

similarity = cosine_similarity([vec1], [vec2])[0][0]
print(f"Similarity: {similarity:.4f}")
