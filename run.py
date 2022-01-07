#!/usr/bin/env python3

import cv2
import numpy as np
from PIL import Image

cap = cv2.VideoCapture(2)
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
        
def list_ports():
    """
    Test the ports and returns a tuple with the available ports and the ones that are working.
    """
    non_working_ports = []
    dev_port = 0
    working_ports = []
    available_ports = []
    while len(non_working_ports) < 6: # if there are more than 5 non working ports stop the testing. 
        camera = cv2.VideoCapture(dev_port)
        if not camera.isOpened():
            non_working_ports.append(dev_port)
            print("Port %s is not working." %dev_port)
        else:
            is_reading, img = camera.read()
            w = camera.get(3)
            h = camera.get(4)
            if is_reading:
                print("Port %s is working and reads images (%s x %s)" %(dev_port,h,w))
                working_ports.append(dev_port)
            else:
                print("Port %s for camera ( %s x %s) is present but does not reads." %(dev_port,h,w))
                available_ports.append(dev_port)
        dev_port +=1
    return available_ports,working_ports,non_working_ports

# list_ports()