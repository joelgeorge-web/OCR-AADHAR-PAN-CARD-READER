# OCR-AADHAR-PAN-CARD-READER and Text Extraction


This script allows you to detect an ID card in a video stream and extract the text from the detected card. It uses object detection techniques to locate the ID card in the frame and performs Optical Character Recognition (OCR) to extract the text.

## Setup

1. Clone the repository and navigate to the project directory.
2. Install the required dependencies by running the command: `pip install -r requirements.txt`.

## Usage

1. Run the script by executing the command: `python id_card_detection.py`.
2. The webcam feed will open, and the script will start detecting ID cards in real-time.
3. Once an ID card is detected, it will be highlighted, and the extracted text will be printed in the console.
4. Press 'q' to quit the script.

## Configuration

- The accuracy threshold for ID card detection can be adjusted by modifying the `accuracy_threshold` variable in the script.
- The script supports the detection of multiple types of ID cards. Currently, it is configured to detect cards defined in the `labelmap.pbtxt` file. You can modify the label map file to add or remove card types as per your requirements.

## Dependencies

- OpenCV (cv2)
- TensorFlow (tf)
- EasyOCR
- NumPy
- Pillow (PIL)

Note: The script assumes that you have a trained object detection model and the corresponding frozen inference graph (.pb file). You need to provide the path to the model files in the script.

