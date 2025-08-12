import numpy as np
import cv2
from PIL import Image

def PilToOpenCV(pil_image):
    data = pil_image.convert("RGB")
    data = np.array(data)
    opencv_image = cv2.cvtColor(data, cv2.COLOR_RGB2BGR)
    return opencv_image

def OpenCVToPil(opencv_image):
    data = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(data)
    return pil_image