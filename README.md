Tiny-YOLO-voc-traffic-sign-detection
===
## Description
### Functionality
User can use our code to detect 25 types of traffic signs in real time on following platforms:
- PC
- NVIDIA TX2
- Raspberry Pi
### Dataset
About 2600 images and labels containing 22 types of traffic sign, car, motorcycle and people.
The data is manually collected through Google Street View and Google Image Search.

## Requirement
python3
Tensorflow 1.15
Tensorflow-gpu 1.15
Cython
DarkFlow
OpenCV

## Training
```shell
python flow --model .\cfg\tiny-yolo-voc-25c.cfg --train --annotation .\dataset\label --dataset .\dataset\image
```

## Demo
PC Demo

![](https://i.imgur.com/ue5stlx.png)

![](https://i.imgur.com/03vqx3E.png)

Raspberry Pi Demo

![](https://i.imgur.com/1X9gukf.png)
