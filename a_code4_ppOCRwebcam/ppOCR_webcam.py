# -*- coding: utf-8 -*-
"""
Created on Fri May 14 16:27:06 2024

@author: kristina

https://github.com/iCounterBOX/Paddle-OCR-on-Webcam/blob/main/README.md

Issues When Using auto-py-to-exe:
https://nitratine.net/blog/post/issues-when-using-auto-py-to-exe/#google_vignette


"""


import cv2  # install macht probleme ...siehe doc
import os
import pkg_resources # für die version anzeige
import sys
import logging
import traceback
from sys import exit
from configparser import ConfigParser


import paddle
from paddleocr import PaddleOCR
from class_ocr import c_ocr 

cocr = c_ocr()
config = ConfigParser()

#.. some kind of global ..........

_camNr = 0

'''
https://stackoverflow.com/questions/7484454/removing-handlers-from-pythons-logging-loggers
Zäh! einmal angelegt wird der name dem Handler übergeben..das auch NUR in einem unserer Module!
Im notfall wenn sich mal der namen des log ändern sollte, dann den Handler rücksetzen:
        
    logging.getLogger().removeHandler(logging.getLogger().handlers[0])
    
    logFileName = os.getcwd() + "\\" +"companyTasks.log"
    logger = logging.getLogger(logFileName)
    logging.basicConfig(filename='companyTasks.log', encoding='utf-8', level=logging.INFO)
    logger.debug('This message should go to the log file')
    logger.info('So should this')
    logger.warning('And this, too')
    logger.error('And non-ASCII stuff, too, like Øresund and Malmö')
'''

logFileName = os.getcwd() + "\\" +"ppOcrCam.log"
logger = logging.getLogger(logFileName)
logging.basicConfig(filename = logFileName,  level=logging.INFO)

logging.info(cocr.dt() + '******** This is Module  ' + os.path.basename(__file__) + '  GO ***************************')



#some add-info
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
gpu_available  = paddle.device.is_compiled_with_cuda()
print("GPU available:", gpu_available)
print("cv2.__file__: " + cv2.__file__) 
print ( "cv2 Version: " +  cv2. __version__ )
print ( "paddleOCR  Version: " +  pkg_resources.get_distribution("paddleocr").version )
print("Cuda 11.2  und cuDNN8")
print ( "scipy   Version: " +  pkg_resources.get_distribution("scipy").version )


'''
Config file:
https://stackoverflow.com/questions/19078170/python-how-would-you-save-a-simple-settings-config-file
standard is webcam 0.. on NB the inbuild cam..1 is normally the external USB cam
[main]
webcamnr = 1
key2 = 4 laterUse
key3 = 4 futureUse
'''
myIniFile = os.getcwd() + "\config.ini"
logging.info(cocr.dt() + 'inifile : ' + myIniFile)
try:
    if config.read(myIniFile):
        _camNr = config.getint('main', 'webCamNr')
        print("webCamNr : " + str(_camNr)) # -> "value1"
    else:     
            with open(myIniFile,'a'): pass  # if not exist..create
            config.read(myIniFile)
            config.add_section('main')
            config.set('main', 'webCamNr', '0')
            config.set('main', 'key2', '4 laterUse')
            config.set('main', 'key3', '4 futureUse')        
            with open('config.ini', 'w') as f:
                config.write(f)            
except Exception as e:
     print(e)        
     _camNr = 0
     logging.error(traceback.format_exc())     
     pass
logging.info(cocr.dt() + 'Camera Nr __ ' + str(_camNr) + ' __ is selected')


try: 
    logging.info(cocr.dt() + 'try.. PaddleOCR( .. ' )
    ocrModel = PaddleOCR( use_angle_cls =True, lang = 'en', use_gpu=True, det_limit_side_len=3456)    
except Exception as e:
    logging.error(traceback.format_exc())
    print(e)     
    

#-------------- webcam -----------------------------------------------

#https://forum.opencv.org/t/how-to-use-waitkey-with-videocapture/10718

try:
    cap = cv2.VideoCapture(_camNr)
    if not cap.isOpened():
        print("Cannot open camera")
        logging.error(cocr.dt() + 'NO CAM AVAILABLE!? - check config-file - check WebCam if available in Windows!' )
        cap.release()
        cv2.destroyAllWindows()
        exit()
except Exception as e:
    logging.error(traceback.format_exc())
    print(e)    
    exit()       
    
n = 0
i = 0
frames_to_count= cap.get(cv2.CAP_PROP_FPS)


while True:  
    ret, capFrame = cap.read()               

    if (2*n) % frames_to_count == 0:
        # update the frame image     
        try:
            imgPaddleOCR = cocr.pp_ocrOnSingleImageCV2(ocrModel, capFrame)
            
            ################imgPaddleOCR = capFrame # nur testen wegen crash in exe
            
            cocr.showInMovedWindow(  "paddle OCR ( P for PAUSE ... C continue..  Q FINISH) ", imgPaddleOCR, 100, 200)
        except Exception as e:
            print(e)    
            logging.error(traceback.format_exc()) 
            break
        i+=1
    n+=1
    if ret==False:
        break
    # MAGIC exit from UI
    
    key = cv2.waitKey(1)
    if key == ord('q'):
        logging.info(cocr.dt() + 'waitkey() - user pressed q - quit Application' )
        break
    if key == ord('p'):
        print('press c to continue')
        logging.info(cocr.dt() + 'waitkey() - user pressed p - PAUSE' )
        cv2.waitKey(-1) #wait until any key is pressed
        if key == ord('c'):
            logging.info(cocr.dt() + 'waitkey() - user pressed c - continue' )
            print('c pressed')
           
 


# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()


print ("ende")
logging.info(cocr.dt() + '******** This is Module  ' + os.path.basename(__file__) + '  END **************************')

sys.exit()



