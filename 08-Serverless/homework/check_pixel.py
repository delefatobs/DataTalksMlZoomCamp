from urllib.request import urlopen
from io import BytesIO
from PIL import Image
import numpy as np

url = "https://habrastorage.org/webt/yf/_d/ok/yf_dokzqy3vcritme8ggnzqlvwa.jpeg"

# download
data = urlopen(url).read()
img = Image.open(BytesIO(data))

# resize: (same as training!)
img = img.resize((128, 128), Image.NEAREST)

# convert to numpy
x = np.array(img).astype("float32")

# SAME preprocessing as training
# in HW8 we did: x / 255
x = (x - 127.5) / 127.5

print("First pixel R:", x[0, 0, 0])

