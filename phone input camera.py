import cv2
import numpy as np
from PIL import Image
import os

cam=cv2.VideoCapture('http://10.42.0.209:8080/video')
cv2.namedWindow('Mobile_Input', cv2.WINDOW_NORMAL)
cv2.resizeWindow("Mobile_Input", 600,600)
while True:
    _,img=cam.read()
    cv2.imshow("Mobile_Input",img)
    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()

















    
