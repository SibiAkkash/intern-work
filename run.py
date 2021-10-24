#!/usr/bin/env python3

import torch
import cv2
import mediapipe as mp
import numpy as np
import torch
from PIL import Image

model = torch.load('weights/yolov5s.pt')
print('model loaded ?')
print(model)

base_dir = 'data/images'
img1 = Image.open(f'{base_dir}/zidane.jpg')  # PIL image
img2 = cv2.imread(f'{base_dir}/bus.jpg')[:, :, ::-1]  # OpenCV image (BGR to RGB)
imgs = [img1, img2]  # batch of images


# Inference
results = model(imgs, size=640)  # includes NMS

# Results
results.print()  
results.show()  # or .save()

# results.xyxy[0]  # img1 predictions (tensor)
# results.pandas().xyxy[0]  # img1 predictions (pandas)

#* webcam input
# cap = cv2.VideoCapture(1)
# print('opened webcam stream')

# while cap.isOpened():
#     success, image = cap.read()
#     if not success:
#         print("Ignoring empty camera frame.")
#         # If loading a video, use 'break' instead of 'continue'
#         continue

#     # cv2.waitKey(delay) waits for atleast delay ms, then returns keycode of pressed key if any
#     # key was pressed, else returns -1
#     if cv2.waitKey(1) == ord('q'):
#         break

# cap.release()
# cv2.destroyAllWindows()
        
