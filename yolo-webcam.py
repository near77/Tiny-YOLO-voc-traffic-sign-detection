# -*- coding: utf-8 -*-
"""
Created on Fri Dec 29 14:11:07 2017

@author: NEAR6
"""

from darkflow.net.build import TFNet
import cv2
import matplotlib.pyplot as plt
from PIL import ImageGrab
import numpy as np
import time
#%config InlineBackend.figure_format = 'svg'

options = {
        "model":'cfg/yolo.cfg',
        'load':'bin/yolo.weights',
        'threshold':0.5,
        'gpu':0.5
        }
tfnet=TFNet(options)
colors = [tuple(255*np.random.rand(3))for i in range(5)]



cap=cv2.VideoCapture(0)
while True:
    stime=time.time()
    ret,img=cap.read()
    results=tfnet.return_predict(img)
    
    if ret:
        for color,result in zip(colors,results):
            tl=(result['topleft']['x'],result['topleft']['y'])
            br=(result['bottomright']['x'],result['bottomright']['y'])
            label = result['label']
            confidence=result['confidence']
            text='{}:{:.0f}%'.format(label,confidence*100)
            img = cv2.rectangle(img,tl,br,color,4)
            img = cv2.putText(img,label,tl,cv2.FONT_HERSHEY_COMPLEX,1,(0,0,0),2)
    print('{:.1f}'.format(1/(time.time()-stime)))
    
    cv2.imshow('object detection',img)
    if cv2.waitKey(25)&0xFF==ord('q'):
        cv2.destroyAllWindows()
        break
cap.release()












