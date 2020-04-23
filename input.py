# Import OpenCV2 for image processing
import cv2
import os
import argparse
from pathlib import Path
from PIL import Image
from mtcnn import MTCNN
from datetime import datetime

from PIL import Image
import numpy as np
from mtcnn_pytorch.src.align_trans import get_reference_facial_points, warp_and_crop_face

def assure_path_exists(path):
    dir = os.path.dirname(path)
    if not os.path.exists(dir):
        os.makedirs(dir)
face_id=input('enter your id')
# Start capturing video 
#vid_cam = cv2.VideoCapture(0)
cap = cv2.VideoCapture(0)
cap.set(3,1280)
cap.set(4,720)
mtcnn = MTCNN()
# Detect object in video stream using Haarcascade Frontal Face
face_detector = cv2.CascadeClassifier('/Users/vijender/opt/anaconda3/lib/python3.7/site-packages/cv2/data/haarcascade_frontalface_default.xml')

# Initialize sample face image
count = 0
mtcnn = MTCNN()
assure_path_exists("dataset/")

# Start looping
while(True):

    # Capture video frame
    _, image_frame = cap.read()
    
    p =  Image.fromarray(image_frame[...,::-1])          
    warped_face = np.array(mtcnn.align(p))[...,::-1]
    count += 1

        # Save the captured image into the datasets folder
    cv2.imwrite("data/facebank/" + str(face_id)+'/'+str(face_id) + '_' + str(count) + ".jpg", warped_face)

        # Display the video frame, with bounded rectangle on the person's face
    cv2.imshow('frame', image_frame)

    # To stop taking video, press 'q' for at least 100ms
    if cv2.waitKey(100) & 0xFF == ord('q'):
        break

    # If image taken reach 100, stop taking video
    elif count>=30:
        print("Successfully Captured")
        break

# Stop video
cap.release()

# Close all started windows
cv2.destroyAllWindows()