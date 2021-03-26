import sys
import argparse
import cv2
import numpy as np
from darkflow.net.build import TFNet
import time
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
   

def read_cam(video_capture):
    if video_capture.isOpened():
        windowName = "TX2 Demo"
        cv2.namedWindow(windowName, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(windowName,1280,720)
        cv2.moveWindow(windowName,0,0)
        cv2.setWindowTitle(windowName,"TX2 Demo")
        font = cv2.FONT_HERSHEY_PLAIN
        showFullScreen = False
        showWindow = 0
        showHelp = True
        helpText="Press 'q' to Quit"
        while True:
            stime = time.time()
            if cv2.getWindowProperty(windowName, 0) < 0: 
                break
            ret_val, frame = video_capture.read()
            results = tfnet.return_predict(frame)
            if showHelp == True:
                cv2.putText(frame, helpText, (11,20), font, 1.0, (32,32,32), 4, cv2.LINE_AA)
                cv2.putText(frame, helpText, (10,20), font, 1.0, (240,240,240), 1, cv2.LINE_AA)
                cv2.putText(frame,'FPS {:.1f}'.format(1 / (time.time() - stime)), (560,20), font, 1.0, (32,32,32), 4, cv2.LINE_AA)
                cv2.putText(frame,'FPS {:.1f}'.format(1 / (time.time() - stime)), (559,20), font, 1.0, (240,240,240), 1, cv2.LINE_AA)
            for color,result in zip(colors,results):
                tl=(result['topleft']['x'],result['topleft']['y'])
                br=(result['bottomright']['x'],result['bottomright']['y'])
                label = result['label']
                confidence=("%.2f"%(result['confidence']*100))
                frame = cv2.rectangle(frame, tl, br, color, 7)
                frame = cv2.putText(frame, label+" "+str(confidence)+"%", tl, font, 1.0, (32,32,32), 4, cv2.LINE_AA)
                frame = cv2.putText(frame, label+" "+str(confidence)+"%", tl, font, 1.0, (240,240,240), 1, cv2.LINE_AA)
            cv2.imshow(windowName,frame)
            if cv2.waitKey(25) & 0xFF == ord("q"):
                cv2.destroyAllWindows()
                break
    else:
     print ("camera open failed")

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
    read_cam(video_capture)
    video_capture.release()
    cv2.destroyAllWindows()
