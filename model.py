import mlserver
from mlserver import MLModel
from mlserver.types import InferenceResponse, ResponseOutput
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import base64
import io
import torch
import tensorflow as tf
import numpy as np


class ImageModel(MLModel):
    async def load(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(self.device)
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        self.model.eval()
        self.mlserver_version = mlserver.__version__
        return True

    async def predict(self, payload):
        image_b64 = payload.inputs[0].data[0]
        image_bytes = base64.b64decode(image_b64.split(',')[-1])
        image = Image.open(io.BytesIO(image_bytes))

        inputs = self.processor(images=image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            vector = self.model.get_image_features(**inputs)[0].cpu().tolist()


        output = ResponseOutput(
            name="vector",
            shape=[len(vector)],
            datatype="FP32",
            data=vector
        )

        return InferenceResponse(
            model_name=self.name,
            outputs=[output],
            model_version=mlserver.__version__
        )

class TFLightModel(MLModel):
    async def load(self):
        self.interpreter = tf.lite.Interpreter(
            model_path="mobile_face_net.tflite",
            num_threads=4
        )
        self.interpreter.allocate_tensors()
        # Get input and output tensors.
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        # check the type of the input tensor
        if self.input_details[0]['dtype'] == np.float32:
            self.floating_model = True
        # NxHxWxC, H:1, W:2
        #self.height = self.input_details[0]['shape'][1]
        #self.width = self.input_details[0]['shape'][2]
        return True

    async def predict(self, payload):
        image_b64 = payload.inputs[0].data[0]
        image_bytes = base64.b64decode(image_b64.split(',')[-1])
        image = Image.open(io.BytesIO(image_bytes))

        input_data = np.array(image)
        input_data = input_data[:, :, ::-1]  # RGB → BGR

        if self.floating_model:
            input_data = (np.float32(input_data) - 128) / 128

        input_data = np.expand_dims(input_data, axis=0)
        self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
        self.interpreter.invoke()

        vector = self.interpreter.get_tensor(self.output_details[0]['index'])
        vector = vector[0].tolist()

        output = ResponseOutput(
            name="vector",
            shape=[len(vector)],
            datatype="FP32",
            data=vector
        )

        return InferenceResponse(
            model_name=self.name,
            outputs=[output],
            model_version=mlserver.__version__
        )