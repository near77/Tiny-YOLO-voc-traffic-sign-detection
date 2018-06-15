import sys
import argparse
import cv2
import numpy as np
from darkflow.net.build import TFNet
import time
import threading
import pygame

flag = True
label_name = None
windowName = None

options = {
        "model":'cfg/tiny-yolo-voc-25c.cfg',
        'load':76500,
        'threshold':0.3,
        'gpu':0.3
        }
tfnet=TFNet(options)

colors = [tuple(255*np.random.rand(3))for i in range(5)]

def parse_cli_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_device", dest="video_device",
                        help="Video device # of USB webcam (/dev/video?) [0]",
                        default=0, type=int)
    arguments = parser.parse_args()
    return arguments

# On versions of L4T previous to L4T 28.1, flip-method=2
# Use the Jetson onboard camera
def open_onboard_camera():
    return cv2.VideoCapture("nvcamerasrc ! video/x-raw(memory:NVMM), width=(int)640, height=(int)480, format=(string)I420, framerate=(fraction)30/1 ! nvvidconv ! video/x-raw, format=(string)BGRx ! videoconvert ! video/x-raw, format=(string)BGR ! appsink")

# Open an external usb camera /dev/videoX
def open_camera_device(device_number):
    return cv2.VideoCapture(device_number)
   

def read_cam_precondition(video_capture):
    if video_capture.isOpened():
        global windowName
        windowName = "YOLOv2 on TX2"
        cv2.namedWindow(windowName, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(windowName,1280,720)
        cv2.moveWindow(windowName,0,0)
        cv2.setWindowTitle(windowName,"YOLOv2 on TX2")
    
    else:
        print ("camera open failed")
        
def read_cam(video_capture):
    stime = time.time()
    ret_val, frame = video_capture.read()
    results = tfnet.return_predict(frame)
    font = cv2.FONT_HERSHEY_PLAIN
    helpText="Press 'q' to Quit"
    if showHelp == True:
        cv2.putText(frame, helpText, (11,20), font, 1.0, (32,32,32), 4, cv2.LINE_AA)
        cv2.putText(frame, helpText, (10,20), font, 1.0, (240,240,240), 1, cv2.LINE_AA)
    for color,result in zip(colors,results):
        tl=(result['topleft']['x'],result['topleft']['y'])
        br=(result['bottomright']['x'],result['bottomright']['y'])
        label = result['label']
        confidence=result['confidence']
        global label_name
        label_name = label
        text='{}:{:.0f}%'.format(label,confidence*100)
        frame = cv2.rectangle(frame,tl,br,color,4)
        frame = cv2.putText(frame, label+" "+str(confidence)+"%", tl, font, 1.0, (32,32,32), 4, cv2.LINE_AA)
        frame = cv2.putText(frame, label+" "+str(confidence)+"%", tl, font, 1.0, (240,240,240), 1, cv2.LINE_AA)
        cv2.putText(frame,'FPS {:.1f}'.format(1 / (time.time() - stime)), (720,20), font, 1.0, (32,32,32), 4, cv2.LINE_AA)
        cv2.putText(frame,'FPS {:.1f}'.format(1 / (time.time() - stime)), (719,20), font, 1.0, (240,240,240), 1, cv2.LINE_AA)

    cv2.imshow(windowName,frame)
    print('FPS {:.1f}'.format(1 / (time.time() - stime)))

def play_music(label_name):
    if label_name != None:
        if label_name[:11]=="Speed_limit":
            file='mp3/speed_limit.mp3' 
            pygame.mixer.init() 
            track = pygame.mixer.music.load(file)
            pygame.mixer.music.play()
            time.sleep(1.35)
            pygame.mixer.music.stop()
        elif label_name[:13]=="Traffic_light":
            print("gg")
            file='mp3/traffic_light.mp3' 
            pygame.mixer.init() 
            track = pygame.mixer.music.load(file)
            pygame.mixer.music.play()
            time.sleep(1.25)
            pygame.mixer.music.stop()
        elif label_name=="Bend_to_left":
            file='mp3/bend_to_left.mp3' 
            pygame.mixer.init() 
            track = pygame.mixer.music.load(file)
            pygame.mixer.music.play()
            time.sleep(1.45)
            pygame.mixer.music.stop()
        elif label_name=="Bend_to_right":
            file='mp3/bend_to_right.mp3' 
            pygame.mixer.init() 
            track = pygame.mixer.music.load(file)
            pygame.mixer.music.play()
            time.sleep(2)
            pygame.mixer.music.stop()
        elif label_name[:11]=="Double_bend":
            file='mp3/double_bend.mp3' 
            pygame.mixer.init() 
            track = pygame.mixer.music.load(file)
            pygame.mixer.music.play()
            time.sleep(1.4)
            pygame.mixer.music.stop()
        elif label_name=="Fork_road":
            file='mp3/fork_road.mp3' 
            pygame.mixer.init() 
            track = pygame.mixer.music.load(file)
            pygame.mixer.music.play()
            time.sleep(1.4)
            pygame.mixer.music.stop()
        elif label_name=="Narrow_road":
            file='mp3/narrow_road.mp3' 
            pygame.mixer.init() 
            track = pygame.mixer.music.load(file)
            pygame.mixer.music.play()
            time.sleep(1.45)
            pygame.mixer.music.stop()
        elif label_name=="No_entry":
            file='mp3/no_entry.mp3' 
            pygame.mixer.init() 
            track = pygame.mixer.music.load(file)
            pygame.mixer.music.play()
            time.sleep(1.4)
            pygame.mixer.music.stop()
        elif label_name=="No_left_turn":
            file='mp3/no_left_turn.mp3' 
            pygame.mixer.init() 
            track = pygame.mixer.music.load(file)
            pygame.mixer.music.play()
            time.sleep(1.55)
            pygame.mixer.music.stop()
        elif label_name=="No_right_turn":
            file='mp3/no_right_turn.mp3' 
            pygame.mixer.init() 
            track = pygame.mixer.music.load(file)
            pygame.mixer.music.play()
            time.sleep(1.5)
            pygame.mixer.music.stop()
        elif label_name=="No_u_turn":
            file='mp3/no_u_turn.mp3' 
            pygame.mixer.init() 
            track = pygame.mixer.music.load(file)
            pygame.mixer.music.play()
            time.sleep(1.5)
            pygame.mixer.music.stop()

    else:
        print(label_name)


class myThread (threading.Thread):
    def __init__(self, threadID, name, method, label_name):
        threading.Thread.__init__(self)
        self.threadID = threadID
        self.name = name
        self.method = str(method)
        self.label_name = label_name
    def run(self):
        print ("Start thread ：" + self.name)
        if self.method == 'play_music':
            play_music(self.label_name)
        else :
            print("else")
        print ("Exit thread ：" + self.name)
        global flag
        flag = True


if __name__ == '__main__':
    arguments = parse_cli_args()
    print("Called with args:")
    print(arguments)
    print("OpenCV version: {}".format(cv2.__version__))
    print("Device Number:",arguments.video_device)
    if arguments.video_device==0:
      video_capture=open_onboard_camera()
    else:
      video_capture=open_camera_device(arguments.video_device)
    
    read_cam_precondition(video_capture)
    font = cv2.FONT_HERSHEY_PLAIN
    showFullScreen = False
    showWindow = 0
    showHelp = True
    

    while True: 
    
        read_cam(video_capture)
    
        if flag:
            flag = False
            thread1 = myThread(1, "Thread-1", 'play_music', label_name)
            thread1.start()
        
        if cv2.waitKey(25) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            print('Detection ends.')
            break

    video_capture.release()
    cv2.destroyAllWindows()
