# -*- coding: utf-8 -*-
"""
Created on Fri May 10 16:27:06 2024

@author: kristina

https://github.com/iCounterBOX/Paddle-OCR-on-Webcam/blob/main/README.md



"""

import paddle
import cv2
import os
import pkg_resources
import sys
import traceback
import logging

from paddleocr import PaddleOCR
from class_ocr import c_ocr 

cocr = c_ocr()


#some add-info
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
gpu_available  = paddle.device.is_compiled_with_cuda()
print("GPU available:", gpu_available)

print("cv2.__file__: " + cv2.__file__) 
print ( "cv2 Version: " +  cv2. __version__ )
print ( "paddleOCR  Version: " +  pkg_resources.get_distribution("paddleocr").version )
print("Cuda 11.2  und cuDNN8")


#erkennt nur die hälfte vom purLabel - schlecht bei curved text!
# https://learnopencv.com/optical-character-recognition-using-paddleocr/

ocrModel = PaddleOCR( use_angle_cls =True, lang = 'en', use_gpu=True, det_limit_side_len=3456)    

#-------------- webcam -----------------------------------------------

#https://forum.opencv.org/t/how-to-use-waitkey-with-videocapture/10718
cap = cv2.VideoCapture(1)
if not cap.isOpened():
    print("Cannot open camera")
    cap.release()
    cv2.destroyAllWindows()
    exit()
    
n = 0
i = 0
frames_to_count= cap.get(cv2.CAP_PROP_FPS)


while True:  
    ret, capFrame = cap.read()               

    if (2*n) % frames_to_count == 0:
        # update the frame image        
        imgPaddleOCR = cocr.pp_ocrOnSingleImageCV2(ocrModel, capFrame)
        cocr.showInMovedWindow(  "paddle OCR ( P for PAUSE ... C continue..  Q FINISH) ", imgPaddleOCR, 100, 200)
        
        i+=1
    n+=1
    if ret==False:
        break
    # MAGIC exit from UI
    
    key = cv2.waitKey(1)
    if key == ord('q'):
        break
    if key == ord('p'):
        print('press c to continue')
        cv2.waitKey(-1) #wait until any key is pressed
        if key == ord('c'):
            print('c pressed')
           
 


# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()


  











print ("ende")

sys.exit()



