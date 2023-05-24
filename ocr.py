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
from utils import label_map_util
from utils import visualization_utils as vis_util

pan_regex = r'^[A-Z\d]{10}$'
aadhar_regex = r'^\d{4}\s\d{4}\s\d{4}$'

sys.path.append("..")

MODEL_NAME = 'model'
CWD_PATH = os.getcwd()
PATH_TO_CKPT = os.path.join(CWD_PATH, MODEL_NAME, 'frozen_inference_graph.pb')
PATH_TO_LABELS = os.path.join(CWD_PATH, 'data', 'labelmap.pbtxt')
NUM_CLASSES = 1

label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

detection_graph = tf.Graph()

with detection_graph.as_default():
    loaded_graph_def = tf.compat.v1.GraphDef()
    with tf.compat.v2.io.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        loaded_graph_def.ParseFromString(fid.read())
        tf.import_graph_def(loaded_graph_def, name='')
    sess = tf.compat.v1.Session(graph=detection_graph)

image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
num_detections = detection_graph.get_tensor_by_name('num_detections:0')

video = cv2.VideoCapture(0)
ret = video.set(3, 1280)
ret = video.set(4, 720)

id_card_detected = False
accuracy_threshold = 0.8
save_path = os.path.join(CWD_PATH, 'test_images', 'image.jpg')

while True:
    ret, frame = video.read()
    frame_expanded = np.expand_dims(frame, axis=0)
    (boxes, scores, classes, num) = sess.run(
        [detection_boxes, detection_scores, detection_classes, num_detections],
        feed_dict={image_tensor: frame_expanded}
    )
    classes = classes.astype(np.int32)
    image, array_coord = vis_util.visualize_boxes_and_labels_on_image_array(
        frame,
        np.squeeze(boxes),
        np.squeeze(classes),
        np.squeeze(scores),
        category_index,
        use_normalized_coordinates=True,
        line_thickness=3,
        min_score_thresh=0.80
    )
    ymin, xmin, ymax, xmax = boxes[0][0]
    im_width, im_height = frame.shape[1], frame.shape[0]
    left, right, top, bottom = int(xmin * im_width), int(xmax * im_width), int(ymin * im_height), int(ymax * im_height)
    image_inside_box = frame[top:bottom, left:right]
    cv2.imwrite(save_path, image_inside_box)
    cv2.imshow('ID CARD DETECTOR', frame)

    if cv2.waitKey(1) == ord('q'):
        break

video.release()
cv2.destroyAllWindows()

IMAGE_NAME = 'test_images/image.jpg'
PATH_TO_IMAGE = os.path.join(CWD_PATH, IMAGE_NAME)

label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
num_detections = detection_graph.get_tensor_by_name('num_detections:0')

image = cv2.imread(PATH_TO_IMAGE)
image_expanded = np.expand_dims(image, axis=0)
(boxes, scores, classes, num) = sess.run(
    [detection_boxes, detection_scores, detection_classes, num_detections],
    feed_dict={image_tensor: image_expanded}
)

image, array_coord = vis_util.visualize_boxes_and_labels_on_image_array(
    image,
    np.squeeze(boxes),
    np.squeeze(classes).astype(np.int32),
    np.squeeze(scores),
    category_index,
    use_normalized_coordinates=True,
    line_thickness=3,
    min_score_thresh=0.60
)

ymin, xmin, ymax, xmax = array_coord
shape = np.shape(image)
im_width, im_height = shape[1], shape[0]
(left, right, top, bottom) = (xmin * im_width, xmax * im_width, ymin * im_height, ymax * im_height)

image_path = PATH_TO_IMAGE
output_path = 'test_images/image1_cropped.png'

im = Image.open(image_path)
im.crop((left, top, right, bottom)).save(output_path, quality=95)

image_cropped = cv2.imread(output_path)

reader = easyocr.Reader(['en'], gpu=False)

def rotate_image(image: np.ndarray, angle: float) -> np.ndarray:
    old_width, old_height = image.shape[:2]
    angle_radian = math.radians(angle)
    width = abs(np.sin(angle_radian) * old_height) + abs(np.cos(angle_radian) * old_width)
    height = abs(np.sin(angle_radian) * old_width) + abs(np.cos(angle_radian) * old_height)
    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    rot_mat[1, 2] += (width - old_width) / 2
    rot_mat[0, 2] += (height - old_height) / 2
    return cv2.warpAffine(image, rot_mat, (int(round(height)), int(round(width))), borderValue=(0, 0, 0))

image = cv2.imread('test_images/image.jpg')
grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
angle = determine_skew(grayscale)
rotated = rotate_image(image, angle)
cv2.imwrite('image2.jpg', rotated)

result = reader.readtext('image2.jpg')

print("\n")

for detection in result:
    text = detection[1]
    aadhar_match = re.search(aadhar_regex, text)
    pan_match = re.search(pan_regex, text, re.I)
    
    if aadhar_match:
        print("AADHAR CARD\n")
        print(text)
    elif pan_match:
        print("PAN CARD\n")
        corrected_pan = text[:5].upper() + text[5:9].upper() + text[9:].upper()
        corrected_pan_number = corrected_pan[:5] + corrected_pan[5:9].replace('I', '1').replace('i', '1').replace('o', '0').replace('O', '0').replace('z', '2').replace('Z', '2') + corrected_pan[9:]
        print(corrected_pan_number)

cv2.waitKey(0)
cv2.destroyAllWindows()

os.remove(save_path)
os.remove(output_path)
os.remove('image2.jpg')