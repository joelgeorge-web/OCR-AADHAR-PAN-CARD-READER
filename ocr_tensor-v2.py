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


# pan_regex = r'^[A-Z\d]{10}$'
# aadhar_regex = r'^\d{4}\s\d{4}\s\d{4}$'
aadhar_regex = r'^\d{4}\s\d{4}\s\d{4}$'
male = r'(?i)^Male\s*$'
dob = r'\d{2}/\d{2}/\d{4}'
name_regex = r'^[A-Za-z]+(?:\s[A-Za-z]+){1,2}$'
pan_regex = r'^[A-Z\d]{10}$'
name_found = False  # Flag to track if a name has been printed
name_found1 = False  # Flag to track if a name has been printed
global n1


# This is needed since the notebook is stored in the object_detection folder.
sys.path.append("..")

# Import utilites
from utils import label_map_util
from utils import visualization_utils as vis_util

# Name of the directory containing the object detection module we're using
MODEL_NAME = 'model'

# Grab path to current working directory
CWD_PATH = os.getcwd()

# Path to frozen detection graph .pb file, which contains the model that is used
# for object detection.
PATH_TO_CKPT = os.path.join(CWD_PATH,MODEL_NAME,'frozen_inference_graph.pb')

# Path to label map file
PATH_TO_LABELS = os.path.join(CWD_PATH,'data','labelmap.pbtxt')

# Number of classes the object detector can identify
NUM_CLASSES = 1

## Load the label map.
# Label maps map indices to category names, so that when our convolution
# network predicts `5`, we know that this corresponds to `king`.
# Here we use internal utility functions, but anything that returns a
# dictionary mapping integers to appropriate string labels would be fine
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

# Load the Tensorflow model into memory.
detection_graph = tf.Graph()
with detection_graph.as_default():
    loaded_graph_def = tf.compat.v1.GraphDef()
    with tf.compat.v2.io.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        loaded_graph_def.ParseFromString(fid.read())
        tf.import_graph_def(loaded_graph_def, name='')

    sess = tf.compat.v1.Session(graph=detection_graph)

# Input tensor is the image
image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')

# Output tensors are the detection boxes, scores, and classes
# Each box represents a part of the image where a particular object was detected
detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')

# Each score represents level of confidence for each of the objects.
# The score is shown on the result image, together with the class label.
detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')

# Number of objects detected
num_detections = detection_graph.get_tensor_by_name('num_detections:0')

# Initialize webcam feed
video = cv2.VideoCapture(0)
ret = video.set(3,1280)
ret = video.set(4,720)

# Variables to track ID card detection and accuracy
id_card_detected = False
accuracy_threshold = 0.8

# Define the path to save the image
save_path = os.path.join(CWD_PATH, 'test_images', 'image.jpg')

while(True):

    # Acquire frame and expand frame dimensions to have shape: [1, None, None, 3]
    # i.e. a single-column array, where each item in the column has the pixel RGB value
    ret, frame = video.read()
    frame_expanded = np.expand_dims(frame, axis=0)

    # Perform the actual detection by running the model with the image as input
    (boxes, scores, classes, num) = sess.run(
        [detection_boxes, detection_scores, detection_classes, num_detections],
        feed_dict={image_tensor: frame_expanded})

    # Convert the classes array to integers
    classes = classes.astype(np.int32)
    
    # Draw the results of the detection (aka 'visulaize the results')
    image, array_coord = vis_util.visualize_boxes_and_labels_on_image_array(
        frame,
        np.squeeze(boxes),
        np.squeeze(classes),
        np.squeeze(scores),
        category_index,
        use_normalized_coordinates=True,
        line_thickness=3,
        min_score_thresh=0.80)


     # Get the bounding box coordinates
    ymin, xmin, ymax, xmax = boxes[0][0]

    # Convert the normalized coordinates to pixel coordinates
    im_width, im_height = frame.shape[1], frame.shape[0]
    left, right, top, bottom = int(xmin * im_width), int(xmax * im_width), int(ymin * im_height), int(ymax * im_height)

    # Save the image inside the bounding box
    image_inside_box = frame[top:bottom, left:right]
    cv2.imwrite(save_path, image_inside_box)

    # Display the frame
    cv2.imshow('ID CARD DETECTOR', frame)

    # Press 'q' to quit
    if cv2.waitKey(1) == ord('q'):
        break

# Clean up
video.release()
cv2.destroyAllWindows()



IMAGE_NAME = 'test_images/image.jpg'


# Path to frozen detection graph .pb file, which contains the model that is used
# for object detection.
PATH_TO_CKPT = os.path.join(CWD_PATH,MODEL_NAME,'frozen_inference_graph.pb')

# Path to label map file
PATH_TO_LABELS = os.path.join(CWD_PATH,'data','labelmap.pbtxt')

# Path to image
PATH_TO_IMAGE = os.path.join(CWD_PATH,IMAGE_NAME)

# Number of classes the object detector can identify
NUM_CLASSES = 1

# Load the label map.
# Label maps map indices to category names, so that when our convolution
# network predicts `5`, we know that this corresponds to `king`.
# Here we use internal utility functions, but anything that returns a
# dictionary mapping integers to appropriate string labels would be fine
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

# Load the Tensorflow model into memory.
detection_graph = tf.Graph()
with detection_graph.as_default():
    loaded_graph_def = tf.compat.v1.GraphDef()
    with tf.compat.v2.io.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        loaded_graph_def.ParseFromString(fid.read())
        tf.import_graph_def(loaded_graph_def, name='')

    sess = tf.compat.v1.Session(graph=detection_graph)



# Define input and output tensors (i.e. data) for the object detection classifier

# Input tensor is the image
image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')

# Output tensors are the detection boxes, scores, and classes
# Each box represents a part of the image where a particular object was detected
detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')

# Each score represents level of confidence for each of the objects.
# The score is shown on the result image, together with the class label.
detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')

# Number of objects detected
num_detections = detection_graph.get_tensor_by_name('num_detections:0')

# Load image using OpenCV and
# expand image dimensions to have shape: [1, None, None, 3]
# i.e. a single-column array, where each item in the column has the pixel RGB value
image = cv2.imread(PATH_TO_IMAGE)
image_expanded = np.expand_dims(image, axis=0)

# Perform the actual detection by running the model with the image as input
(boxes, scores, classes, num) = sess.run(
    [detection_boxes, detection_scores, detection_classes, num_detections],
    feed_dict={image_tensor: image_expanded})

# Draw the results of the detection (aka 'visulaize the results')
image, array_coord = vis_util.visualize_boxes_and_labels_on_image_array(
    image,
    np.squeeze(boxes),
    np.squeeze(classes).astype(np.int32),
    np.squeeze(scores),
    category_index,
    use_normalized_coordinates=True,
    line_thickness=3,
    min_score_thresh=0.60)

ymin, xmin, ymax, xmax = array_coord

shape = np.shape(image)
im_width, im_height = shape[1], shape[0]
(left, right, top, bottom) = (xmin * im_width, xmax * im_width, ymin * im_height, ymax * im_height)

# Using Image to crop and save the extracted copied image
image_path = PATH_TO_IMAGE
output_path = 'test_images/image1_cropped.png'

im = Image.open(image_path)
im.crop((left, top, right, bottom)).save(output_path, quality=95)

image_cropped = cv2.imread(output_path)

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

image = cv2.imread(output_path)
grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
angle = determine_skew(grayscale)
rotated = rotate(image, angle, (0, 0, 0))
cv2.imwrite('image2.jpg', rotated)

# Perform OCR on the image
result = reader.readtext('image2.jpg')

print("\n")

# Print the text extracted from the image

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

print("\n")

cv2.waitKey(0)

# Clean up
cv2.destroyAllWindows()

# Delete the file
os.remove(save_path)
os.remove(output_path)
os.remove('image2.jpg')