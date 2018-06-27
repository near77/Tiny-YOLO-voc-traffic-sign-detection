import cv2
from darkflow.net.build import TFNet
import numpy as np
from PIL import ImageGrab
import time

options = {
    'model': 'cfg/tiny-yolo-voc-25c.cfg',
    'load': 76500,
    'threshold': 0.15,
    'gpu': 0.6
}

tfnet = TFNet(options)

colors = [tuple(255 * np.random.rand(3)) for i in range(5)]


def detection():
    stime = time.time()
    screen = np.array(ImageGrab.grab(bbox=(0,140,800,740)))
    frame = cv2.cvtColor(screen, cv2.COLOR_BGR2RGB)
    font = cv2.FONT_HERSHEY_PLAIN
    results = tfnet.return_predict(frame)
    helpText="Press 'q' to Quit"
    cv2.putText(frame, helpText, (11,20), font, 1.0, (32,32,32), 4, cv2.LINE_AA)
    cv2.putText(frame, helpText, (10,20), font, 1.0, (240,240,240), 1, cv2.LINE_AA)
    cv2.putText(frame,'FPS {:.1f}'.format(1 / (time.time() - stime)), (720,20), font, 1.0, (32,32,32), 4, cv2.LINE_AA)
    cv2.putText(frame,'FPS {:.1f}'.format(1 / (time.time() - stime)), (719,20), font, 1.0, (240,240,240), 1, cv2.LINE_AA)
    for color, result in zip(colors, results):
        tl = (result['topleft']['x'], result['topleft']['y'])
        br = (result['bottomright']['x'], result['bottomright']['y'])
        label = result['label']
        confidence=("%.2f"%(result['confidence']*100))
        frame = cv2.rectangle(frame, tl, br, color, 7)
        frame = cv2.putText(frame, label+" "+str(confidence)+"%", tl, font, 1.0, (32,32,32), 4, cv2.LINE_AA)
        frame = cv2.putText(frame, label+" "+str(confidence)+"%", tl, font, 1.0, (240,240,240), 1, cv2.LINE_AA)
    cv2.imshow('PCDemo',frame)
    print('FPS {:.1f}'.format(1 / (time.time() - stime)))



while True:
    detection()  
    if cv2.waitKey(25) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
        print('Detection ends.')
        break
    