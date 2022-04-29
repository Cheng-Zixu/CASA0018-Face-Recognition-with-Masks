# CASA0018-Face-Recognition-with-Masks

CASA0018 Project - Real-time Face Recognition with masks on Raspberry Pi

Find more details on [my report](./media/Face Recognition with masks report.md).

## Application Overview

In this project, my face recognition with masks system consists of 3 steps:

1. Face detection: a face detector is used to localise faces in images or videos;

2. Face alignments: with the facial landmark detector, the faces are aligned to normalised canonical coordinates;

3. Deep Face Recognition: face recognition module consists of face processing, deep feature extraction and face matching.

![](./media/imgs/Face Recognition with masks.png)

<center>Fig 2. Workflow of my Face Recognition with mask system</center>

I used **YOLOv5** to train an object detection model to detect faces with masks or without masks. Then, the face landmarks predictor in the **Dlib** package was adopted to get 68 key points on faces and get the faces after alignment. Consequently, the face recognition model in Dlib would generate **Face ID vectors (or Face Embeddings)** for the faces. Finally, the face vectors would be compared to the Face IDs in different databases (mask/nomask) to recognise your identity according to the detection result. I applied **L2 Distance (or Euclidean Distance)** to calculate the distance between two face IDs. In addition, you need to upload your face with a mask and without a mask to generate Face IDs first in the Face database for recognition.

### Experiments Environments

OS: Windows 10 / Raspberry Pi OS(64-bit)

Platform: Python 3.8.8 Pytorch 1.9.1

GPU: RTX 3080 Laptop 16G VRAM (PC only)

## Install requirements

1. install python package

```
pip install -r yolov5/requirements.txt
pip install dlib
```

2. download Dlib model from browser, unzip and save in `yolov5` folder：
```
http://dlib.net/files/dlib_face_recognition_resnet_model_v1.dat.bz2
http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
```


## Usage

1. Get your face crops, you need one `mask` and one `nomask` images in a folder, and then run the  following command in your terminal:

```
python yolov5/detect.py -- source /dir/to/your/face/folder --save_crop --project Face_crops --name [input your name]
```

Then you will see a new folder named `Face_crops` and in this folder you can see a folder named with your name. You can see your face in the `crop` folder.

![](./media/facecrops.gif)

2. Register your face:

Create a new folder named `Face` in the `yolov5` folder, and create two new folder in `Face` named `mask` and `nomask` , respectively and you need to move your face crop images to these two corresponding folders.

Now, your folder would look like:

```
CASA0018-Face-Recognition-with-Masks
---yolov5
------Face
---------mask
------------your mask face crop image
---------nomask
------------your nomask face crop image
```

Then generate your face embedding:

```
python face_embedding.py
```

3. Now you are ready to run the project using your webcam:

```
python yolov5/detect.py --source 0 --nosave
```

The recognition result would be shown like:

![](./media/imgs/combination.jpg)

## Train your own model

We only train the yolov5 detection model here, since the recognition model is well-trained by Dlib.

1. Download Dataset from https://app.roboflow.com/zixu-cheng/face-recognition-with-mask/  and unzip the file in the project. Find more information in [YOLOv5 train tutorial](https://github.com/ultralytics/yolov5/wiki/Train-Custom-Data) or [YOLOv5 Githhub repository](https://github.com/ultralytics/yolov5).
2.  run the train command in the terminal:

```
python train.py --data /dir/to/dataset/data.yaml --weights yolov5n/s/m/l/x.pt
```

## Download my pretrained model

You can download my pretrained model [here](). (To be updated later...)

### My Detection training Results

For the YOLOv5s I trained for PC, Cos-lr gets the best results of precise, Cos-lr with label smoothing get the best results of recall, and multi-scale training gets the best results of mAP.

<center>Table 1. The experiment results of YOLOv5s model</center>

| exp  |  Model   |      input size      | parameter | GFLOPs | Precise     val/test | Recall     val/test | mAP     val/test | model size | cos-lr | label smoothing | multi-scale | train time(h) |
| :--: | :------: | :------------------: | :-------: | :----: | :------------------: | :-----------------: | :--------------: | :--------: | :----: | :-------------: | :---------: | :-----------: |
|  1   | YOLOv5-s |       640x640        |   7.0M    |  15.8  |     0.966/0.885      |     0.949/0.862     |   0.570/0.441    |   14.5MB   |        |                 |             |     3.373     |
|  2   | YOLOv5-s |       640x640        |   7.0M    |  15.8  |   0.960/**0.889**    |     0.969/0.843     |   0.578/0.442    |   14.5MB   |   ✔    |                 |             |     3.35      |
|  3   | YOLOv5-s |       640x640        |   7.0M    |  15.8  |     0.968/0.871      |     0.969/0.844     |   0.569/0.436    |   14.5MB   |        |        ✔        |             |     3.204     |
|  4   | YOLOv5-s | 320x320~     960x960 |   7.0M    |  15.8  |     0.958/0.883      |     0.963/0.848     | 0.564/**0.444**  |   14.5MB   |        |                 |      ✔      |     3.367     |
|  5   | YOLOv5-s |       640x640        |   7.0M    |  15.8  |     0.969/0.877      |   0.968/**0.865**   |   0.572/0.442    |   14.5MB   |   ✔    |        ✔        |             |     2.916     |
|  6   | YOLOv5-s | 320x320~     960x960 |   7.0M    |  15.8  |     0.956/0.865      |     0.953/0.846     |   0.551/0.431    |   14.5MB   |   ✔    |                 |      ✔      |     3.365     |
|  7   | YOLOv5-s | 320x320~     960x960 |   7.0M    |  15.8  |     0.954/0.871      |     0.948/0.832     |   0.562/0.421    |   14.5MB   |        |        ✔        |      ✔      |     3.355     |
|  8   | YOLOv5-s | 320x320~     960x960 |   7.0M    |  15.8  |      0.961/0.86      |     0.966/0.838     |    0.56/0.419    |   14.5MB   |   ✔    |        ✔        |      ✔      |     3.092     |

For the YOLOv5n I trained for Raspberry, multi-scale training gets the best results of precise and Cos-lr gets the best results of recall and mAP. 

<center>Table 2. The experiment results of YOLOv5n model</center>

| exp  |  Model   |      input size      | parameter | GFLOPs | Precise     val/test | Recall     val/test | mAP     val/test | model size | cos-lr | label smoothing | multi-scale | train time(h) |
| :--: | :------: | :------------------: | :-------: | :----: | :------------------: | :-----------------: | :--------------: | :--------: | :----: | :-------------: | :---------: | :-----------: |
|  1   | YOLOv5-n |       640x640        |   1.8M    |  4.2   |     0.963/0.861      |     0.948/0.822     |   0.518/0.391    |   3.9MB    |        |                 |             |     3.154     |
|  2   | YOLOv5-n |       640x640        |   1.8M    |  4.2   |     0.954/0.856      |   0.953/**0.837**   | 0.516/**0.397**  |   3.9MB    |   ✔    |                 |             |     3.226     |
|  3   | YOLOv5-n |       640x640        |   1.8M    |  4.2   |     0.962/0.884      |     0.932/0.815     |   0.512/0.383    |   3.9MB    |        |        ✔        |             |     3.788     |
|  4   | YOLOv5-n | 320x320~     960x960 |   1.8M    |  4.2   |   0.958/**0.885**    |     0.944/0.799     |   0.496/0.387    |   3.9MB    |        |                 |      ✔      |     2.785     |
|  5   | YOLOv5-n |       640x640        |   1.8M    |  4.2   |     0.965/0.854      |     0.951/0.817     |   0.509/0.383    |   3.9MB    |   ✔    |        ✔        |             |     3.275     |
|  6   | YOLOv5-n | 320x320~     960x960 |   1.8M    |  4.2   |     0.952/0.880      |     0.94/0.811      |   0.485/0.383    |   3.9MB    |   ✔    |                 |      ✔      |     2.942     |
|  7   | YOLOv5-n | 320x320~     960x960 |   1.8M    |  4.2   |     0.948/0.851      |     0.933/0.814     |   0.504/0.386    |   3.9MB    |        |        ✔        |      ✔      |     3.542     |
|  8   | YOLOv5-n | 320x320~     960x960 |   1.8M    |  4.2   |     0.954/0.875      |     0.922/0.808     |   0.507/0.392    |   3.9MB    |   ✔    |        ✔        |      ✔      |     3.317     |

