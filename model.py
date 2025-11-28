from mlserver import MLModel
from mlserver.types import InferenceResponse, ResponseOutput
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import base64
import io
import torch

class ImageModel(MLModel):
    async def load(self):
        self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

        return True

    async def predict(self, payload):
        image_b64 = payload.inputs[0].data[0]
        image_bytes = base64.b64decode(image_b64.split(',')[-1])
        image = Image.open(io.BytesIO(image_bytes))

        inputs = self.processor(images=image, return_tensors="pt")
        with torch.no_grad():
            vector = self.model.get_image_features(**inputs)[0].tolist()


        output = ResponseOutput(
            name="vector",
            shape=[len(vector)],
            datatype="FP32",
            data=vector
        )

        return InferenceResponse(
            model_name=self.name,
            outputs=[output]
        )