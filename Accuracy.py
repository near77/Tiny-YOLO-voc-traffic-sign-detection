import cv2
from darkflow.net.build import TFNet
import numpy as np
import os
from xml.dom.minidom import parse
import xml.dom.minidom
options = {
    'model': 'cfg/tiny-yolo-voc-25c.cfg',
    'load': 76500,
    'threshold': 0.15,
    'gpu': 0.6
}

tfnet = TFNet(options)

colors = [tuple(255 * np.random.rand(3)) for i in range(5)]

label_name = None

count = 0
correct = 0
for img , labeldir in zip(os.scandir('dataset/image'),os.scandir('dataset/label')):
    DOMTree = xml.dom.minidom.parse(labeldir.path)
    collection = DOMTree.documentElement
    labels = collection.getElementsByTagName("object")
    img = cv2.imread(img.path)
    results = tfnet.return_predict(img)
    for result, label in zip(results, labels):
        type = label.getElementsByTagName('name')[0]
        real = type.childNodes[0].data
        
        if result['label'] == real:
            correct+=1
        else:
            pass
        count +=1

print(correct/count)


    