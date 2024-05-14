# -*- coding: utf-8 -*-
"""
Created on Fri Mar 29 08:21:01 2024
Class to hold all kind of tools

@author: kristina
"""

import logging
import datetime
import numpy as np
import os
import cv2
import traceback
from paddleocr import  draw_ocr

import pytesseract
pytesseract.pytesseract.tesseract_cmd = 'C:\Program Files\Tesseract-OCR/tesseract.exe'  # your path may be different
from pytesseract import Output


class c_ocr:        
    def __init__(self):
        
       pass
        
      
    def dt(self):
        return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S") + ' - '



    def showInMovedWindow(self,  winname, img, x, y):
        cv2.namedWindow(winname)        # Create a named window
        cv2.moveWindow(winname, x, y)   # Move it to (x,y)...THis way the image ma appear on TOP of other screens!
        cv2.imshow(winname,img)

        
    
    
    
    def pp_ocrOnSingleImageCV2(self, ocrModel, cvImg):
        
        try:
            result = ocrModel.ocr(cvImg, cls=True)            
        except AttributeError:
            logging.error(traceback.format_exc())
            return cvImg
            
        try:       
            
            print(type(result))
            if result is None:
                logging.error(self.dt() + 'pp_ocrOnSingleImageCV2() - result is none!' )                
                return cvImg
            
            for idx in range(len(result)):
                res = result[idx]
                if res is None:
                    return cvImg
                for line in res:
                    print(line)  
                    
            result = result[0]       
            boxes = [line[0] for line in result]
            txts = [line[1][0] for line in result]
            scores = [line[1][1] for line in result]
                
            # Specifying font path for draw_ocr method
            font = os.path.join('PaddleOCR', 'doc', 'fonts', 'latin.ttf')          
        
            # draw annotations on image
            im_2show= draw_ocr(cvImg, boxes, txts, scores, font_path=font) 
        except Exception as e:
            logging.error(traceback.format_exc())
            print(e)     
            return cvImg
        return im_2show
   
    
#pyTesseract

    def ocrTesseractSingleImage(self, imgpath, minimalConfidence):
        img = cv2.imread(imgpath)  # original pumpe mit schild
        # convert both the image and template to grayscale
        capFrameGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)        

        #OCR
        try: 
            results = pytesseract.image_to_data(capFrameGray, output_type=Output.DICT)                
               # loop over each of the individual text localizations
            for i in range(0, len(results["text"])):
                # extract the bounding box coordinates of the text region from
                # the current result
                x = results["left"][i]
                y = results["top"][i]
                w = results["width"][i]
                h = results["height"][i]
                # extract the OCR text itself along with the confidence of the
                # text localization
                text = results["text"][i]
                conf = int(results["conf"][i])

            # filter out weak confidence text localizations            
                                
                if conf > minimalConfidence:
                    # display the confidence and text to our terminal
                    #print("Confidence: {}".format(conf))               
                    print("")
                    # strip out non-ASCII text so we can draw the text on the image
                    # using OpenCV, then draw a bounding box around the text along
                    # with the text itself
                    text = "".join([c if ord(c) < 128 else "" for c in text]).strip()
                    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 1)
                    cv2.putText(img, text, (x+150, y ), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 1)                ##print("Text: {}".format(text))
                    print(text)
                    
            # show the output image
            frame = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        except Exception as e:
            logging.error(traceback.format_exc())
            print(e)     
            cv2.destroyAllWindows()
        
        return frame