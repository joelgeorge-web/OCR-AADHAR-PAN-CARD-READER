import easyocr
import cv2
import numpy as np
import os
import re
from PIL import Image
from skimage import io
from skimage.transform import rotate
from skimage.color import rgb2gray
from deskew import determine_skew
from typing import Union
import math

# Initialize the OCR reader
reader = easyocr.Reader(['en'], gpu=False)

# Grab path to current working directory
CWD_PATH = os.getcwd()

# Initialize webcam feed
video = cv2.VideoCapture(0)
ret = video.set(3, 1280)
ret = video.set(4, 720)

# Define the path to save the image
save_path = os.path.join(CWD_PATH, 'test_images', 'image.jpg')

while True:
    # Acquire frame from the video feed
    ret, frame = video.read()

    # Display the frame
    cv2.imshow('IMAGE', frame)

    # Press 's' to capture and save the image
    if cv2.waitKey(1) == ord('q'):
        cv2.imwrite(save_path, frame)
        break

# Clean up
video.release()
cv2.destroyAllWindows()

# Load the captured image
image = cv2.imread(save_path)

result = reader.readtext('test_images/image.jpg')

# Print the OCR results
for detection in result:
    text = detection[1]
    print(text.upper())


os.remove('test_images/image.jpg')