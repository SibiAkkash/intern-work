#!/usr/bin/env python3

import cv2
import numpy as np
from PIL import Image

cap = cv2.VideoCapture(0)
print("opened webcam stream")

count = 0
# Get each frame from webcam stream
while cap.isOpened():
    success, img = cap.read()
    count += 1
    if not success:
        print("Ignoring empty camera frame.")
        # If loading a video, use 'break' instead of 'continue'
        continue

    cv2.imshow("vid", img)

    if cv2.waitKey(5) == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
        
