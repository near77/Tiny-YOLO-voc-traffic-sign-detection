# -*- coding: utf-8 -*-
"""
Created on Fri Dec 29 14:11:07 2017

@author: NEAR6
"""

from darkflow.net.build import TFNet
import cv2
import matplotlib.pyplot as plt
from picamera.array import PiRGBArray
from picamera import PiCamera
import time
import numpy as np
import pygame
#%config InlineBackend.figure_format = 'svg'

options = {
        "model":'cfg/tiny-yolo-voc-25c.cfg',
        'load':45000,
        'threshold':0.1,
        'gpu':0
        }
tfnet=TFNet(options)
colors = [tuple(255*np.random.rand(3))for i in range(5)]

camera = PiCamera()
camera.resolution = (640, 480)
camera.framerate = 32
rawCapture = PiRGBArray(camera, size=(640, 480))

for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
    stime=time.time()
    img = frame.array
    results = tfnet.return_predict(img)
	
    for color,result in zip(colors,results):
        tl=(result['topleft']['x'],result['topleft']['y'])
        br=(result['bottomright']['x'],result['bottomright']['y'])
        label = result['label']
        font = cv2.FONT_HERSHEY_PLAIN
        helpText="Press 'q' to Quit"
        confidence=("%.2f"%(result['confidence']*100))
        cv2.putText(frame, helpText, (11,20), font, 1.0, (32,32,32), 4, cv2.LINE_AA)
        cv2.putText(frame, helpText, (10,20), font, 1.0, (240,240,240), 1, cv2.LINE_AA)
        img = cv2.rectangle(img,tl,br,color,4)
        img = cv2.putText(img, label+" "+str(confidence)+"%", tl, font, 1.0, (32,32,32), 4, cv2.LINE_AA)
        img = cv2.putText(img, label+" "+str(confidence)+"%", tl, font, 1.0, (240,240,240), 1, cv2.LINE_AA)
        cv2.putText(img,'FPS {:.1f}'.format(1 / (time.time() - stime)), (620,20), font, 1.0, (32,32,32), 4, cv2.LINE_AA)
        cv2.putText(img,'FPS {:.1f}'.format(1 / (time.time() - stime)), (619,20), font, 1.0, (240,240,240), 1, cv2.LINE_AA)
        if label[:11]=="Speed_limit":
            file='mp3/speed_limit.mp3' 
            pygame.mixer.init() 
            track = pygame.mixer.music.load(file)
            pygame.mixer.music.play()
            time.sleep(1.35)
            pygame.mixer.music.stop()
        elif label[:13]=="Traffic_light":
            file='mp3/traffic_light.mp3' 
            pygame.mixer.init() 
            track = pygame.mixer.music.load(file)
            pygame.mixer.music.play()
            time.sleep(1.25)
            pygame.mixer.music.stop()
        elif label=="Bend_to_left":
            file='mp3/bend_to_left.mp3' 
            pygame.mixer.init() 
            track = pygame.mixer.music.load(file)
            pygame.mixer.music.play()
            time.sleep(1.4)
            pygame.mixer.music.stop()
        elif label=="Bend_to_right":
            file='mp3/bend_to_right.mp3' 
            pygame.mixer.init() 
            track = pygame.mixer.music.load(file)
            pygame.mixer.music.play()
            time.sleep(2)
            pygame.mixer.music.stop()
        elif label[:11]=="Double_bend":
            file='mp3/double_bend.mp3' 
            pygame.mixer.init() 
            track = pygame.mixer.music.load(file)
            pygame.mixer.music.play()
            time.sleep(1.4)
            pygame.mixer.music.stop()
        elif label=="Fork_road":
            file='mp3/fork_road.mp3' 
            pygame.mixer.init() 
            track = pygame.mixer.music.load(file)
            pygame.mixer.music.play()
            time.sleep(1.4)
            pygame.mixer.music.stop()
        elif label=="Narrow_road":
            file='mp3/narrow_road.mp3' 
            pygame.mixer.init() 
            track = pygame.mixer.music.load(file)
            pygame.mixer.music.play()
            time.sleep(1.4)
            pygame.mixer.music.stop()
        elif label=="No_entry":
            file='mp3/no_entry.mp3' 
            pygame.mixer.init() 
            track = pygame.mixer.music.load(file)
            pygame.mixer.music.play()
            time.sleep(1.4)
            pygame.mixer.music.stop()
        elif label=="No_left_turn":
            file='mp3/no_left_turn.mp3' 
            pygame.mixer.init() 
            track = pygame.mixer.music.load(file)
            pygame.mixer.music.play()
            time.sleep(1.5)
            pygame.mixer.music.stop()
        elif label=="No_right_turn":
            file='mp3/no_right_turn.mp3' 
            pygame.mixer.init() 
            track = pygame.mixer.music.load(file)
            pygame.mixer.music.play()
            time.sleep(1.5)
            pygame.mixer.music.stop()
        elif label=="No_u_turn":
            file='mp3/no_u_turn.mp3' 
            pygame.mixer.init() 
            track = pygame.mixer.music.load(file)
            pygame.mixer.music.play()
            time.sleep(1.5)
            pygame.mixer.music.stop()
    
    print('{:.1f}'.format(1/(time.time()-stime)))
    cv2.imshow('object detection',img)
    key = cv2.waitKey(1) & 0xFF 
    rawCapture.truncate(0)
 
    # if the `q` key was pressed, break from the loop
    if key == ord("q"):
        cv2.destroyAllWindows()
        break


camera.close()










