import onnxruntime as ort
from urllib.request import urlopen
from io import BytesIO
from PIL import Image
import numpy as np

def prepare_image(url):
    data = urlopen(url).read()
    img = Image.open(BytesIO(data))
    img = img.resize((200, 200), Image.NEAREST)
    x = np.array(img).astype("float32")
    x = (x - 127.5) / 127.5
    x = np.expand_dims(x, 0)
    return x

# load model
session = ort.InferenceSession("hair_classifier_empty.onnx")

input_name = session.get_inputs()[0].name

def predict(url):
    x = prepare_image(url)

    # transpose if needed
    if session.get_inputs()[0].shape[1] == 3:
        x = np.transpose(x, (0, 3, 1, 2))

    y = session.run(None, {input_name: x})[0]

    return float(y.squeeze())

def lambda_handler(event, context):
    url = event["url"]
    return predict(url)
