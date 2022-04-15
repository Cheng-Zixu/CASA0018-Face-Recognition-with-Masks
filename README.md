# CASA0018-Face-Recognition-with-Masks
CASA0018 Project - Real-time Face Recognition with masks on Raspberry Pi

## Install requirements

```
pip install -r yolov5/requirements.txt
pip install dlib
```

## Usage

1. Register new face:

You need to upload your face images with mask and without mask in the folder `yolov5/Face` first.

```
python face_embedding.py
```

2. download Dlib model from browser, and save in `yolov5` folder：

```
http://dlib.net/files/dlib_face_recognition_resnet_model_v1.dat
http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
```

3. Inference:

```
python yolov5/detect.py --source 0 --nosave
```

## Train your own model

1. Download Dataset https://universe.roboflow.com/zixu-cheng/face-recognition-with-mask/
2. 
