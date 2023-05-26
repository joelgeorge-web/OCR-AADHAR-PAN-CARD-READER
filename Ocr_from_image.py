# Import packages
import os
import cv2
import numpy as np
import tensorflow as tf
import sys
import easyocr
import re
from PIL import Image
from skimage import io
from skimage.transform import rotate
from skimage.color import rgb2gray
from deskew import determine_skew
from typing import Union
import math

aadhar_regex = r'^\d{4}\s\d{4}\s\d{4}$'
male = r'(?i)^Male\s*$'
dob = r'\d{2}/\d{2}/\d{4}'
name_regex = r'^[A-Za-z]+(?:\s[A-Za-z]+){1,2}$'
pan_regex = r'^[A-Z\d]{10}$'
name_found = False  # Flag to track if a name has been printed
name_found1 = False  # Flag to track if a name has been printed
name_found2 = False  # Flag to track if a name has been printed
id_match_found = False # Flag to track if an ID match has been found
global n1

# This is needed since the notebook is stored in the object_detection folder.
sys.path.append("..")

# Import utilites
from utils import label_map_util
from utils import visualization_utils as vis_util

# Grab path to current working directory
CWD_PATH = os.getcwd()

# Define the path to save the image
save_path = os.path.join(CWD_PATH, 'test_images', 'image.jpg')

# Initialize webcam feed
video = cv2.VideoCapture(0)
ret = video.set(3, 1280)
ret = video.set(4, 720)

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



# Initialize the OCR reader
reader = easyocr.Reader(['en'], gpu=False)  # This needs to run only once to load the model into memory


def rotate(
        image: np.ndarray, angle: float, background: Union[int, tuple[int, int, int]]
) -> np.ndarray:
    old_width, old_height = image.shape[:2]
    angle_radian = math.radians(angle)
    width = abs(np.sin(angle_radian) * old_height) + abs(np.cos(angle_radian) * old_width)
    height = abs(np.sin(angle_radian) * old_width) + abs(np.cos(angle_radian) * old_height)

    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    rot_mat[1, 2] += (width - old_width) / 2
    rot_mat[0, 2] += (height - old_height) / 2
    return cv2.warpAffine(image, rot_mat, (int(round(height)), int(round(width))), borderValue=background)

# Load the captured image
image = cv2.imread(save_path)
grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
angle = determine_skew(grayscale)
rotated = rotate(image, angle, (0, 0, 0))
cv2.imwrite('image2.jpg', rotated)

# Perform OCR on the image
result = reader.readtext('image2.jpg')


for detection in result:
    text = detection[1]
    aadhar_match = re.search(aadhar_regex, text)
    pan_match = re.search(pan_regex, text)
    if aadhar_match:
        print("\n")
        print("AADHAR CARD")
        aadhar_no = aadhar_match.group()
        print("Aadhar No:", aadhar_no)
        for detection in result:
            text = detection[1]
            name_match = re.search(name_regex, text)
            if name_match and name_match.group() != "Government of India" and name_match.group() != "GOVERNMENT OF INDIA" and name_match.group() != "GOVERNMENT OF INDIA" and name_match.group() != "GOvernment OFINDIA" and name_match.group() != "GOVERNMENT Of INDIA":
                name = name_match.group()
                print("Name:", name)
                name_found = True
                break  # Stop iterating after finding a valid name
        for detection in result:
            text = detection[1]
            dob_match = re.search(dob, text)
            if dob_match:
                dob = dob_match.group()
                print("Date of Birth:", dob)
        for detection in result:
            text = detection[1]
            male_match = re.search(male, text)
            if male_match:
                print("Gender: Male")
        # Handle Aadhar card fields
        id_match_found = True
        break  # Stop iterating after finding a valid ID

    elif pan_match:
        global n1
        print("\n")
        print("PAN CARD")
        pan_no = pan_match.group()
        corrected_pan = pan_no[:5].upper() + pan_no[5:9].upper() + pan_no[9:].upper()
        pan_no = corrected_pan[:5] + corrected_pan[5:9].replace('I', '1').replace('i', '1').replace('o', '0').replace('O', '0').replace('z', '2').replace('Z', '2') + corrected_pan[9:]
        print("Pan Account No:", pan_no)
        for detection in result:
            text = detection[1]
            text.upper()
            name_match = re.search(name_regex, text)
            if name_match and name_match.group() != "GOVT OF INDIA" and name_match.group() != "Dale of Birtn" and name_match.group() != "INCOME TAX DEPARTMENT" and name_match.group() != "GOVERNMENT OF INDIA" and name_match.group() != "GOvernment OFINDIA" and name_match.group() != "GOVERNMENT Of INDIA":
                name = name_match.group()
                print("Name:", name.upper())
                name_found = True
                n1 = name.upper()
                break  # Stop iterating after finding a valid name
        for detection in result:
            text = detection[1]
            text.upper()
            name_match = re.search(name_regex, text)
            if name_match and name_match.group() != "GOVT OF INDIA" and name_match.group() != "Dale of Birtn" and name_match.group() != "INCOME TAX DEPARTMENT" and name_match.group() != "GOVERNMENT OF INDIA" and name_match.group() != "GOvernment OFINDIA" and name_match.group() != "GOVERNMENT Of INDIA" and name_match.group().upper() != n1:
                name1 = name_match.group()
                print("Father's Name:", name1.upper()) 
                name_found = True
                break  # Stop iterating after finding a valid name       
        for detection in result:
            text = detection[1]
            dob_match = re.search(dob, text)
            if dob_match:
                dob = dob_match.group()
                print("Date of Birth:", dob)

        # Handle Aadhar card fields
        id_match_found = True
        break  # Stop iterating after finding a valid ID

print("\n")

if not id_match_found:
    print("No valid ID card found")
    print("The following text was detected:")
    for detection in result:
        text = detection[1]
        print(text.upper())



print("\n")

# Delete the file
os.remove(save_path)
os.remove('image2.jpg')

