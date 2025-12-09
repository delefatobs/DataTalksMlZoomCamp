from urllib.request import urlopen
from io import BytesIO
from PIL import Image
import numpy as np
import onnxruntime as ort

# download image
url = "https://habrastorage.org/webt/yf/_d/ok/yf_dokzqy3vcritme8ggnzqlvwa.jpeg"
data = urlopen(url).read()
img = Image.open(BytesIO(data))

# resize (model expects 200x200!)
img = img.resize((200, 200), Image.NEAREST)

# convert to numpy
x = np.array(img).astype("float32")

# same preprocessing as training
x = (x - 127.5) / 127.5

# add batch dim
x = np.expand_dims(x, 0)

# load ONNX model
sess = ort.InferenceSession("hair_classifier_v1.onnx")

# get input name
input_name = sess.get_inputs()[0].name

# rotate if model uses NCHW
if sess.get_inputs()[0].shape[1] == 3:
    x = np.transpose(x, (0, 3, 1, 2))

# run prediction
y = sess.run(None, {input_name: x})[0]

print("Model output:", y)
