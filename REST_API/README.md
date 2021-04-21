# Face Detection REST API with Docker

## 1. clone repo and move directory
```shell
$ git clone https://github.com/J-TKim/Face_Detection.git
$ cd Face_Detection/REST_API
```
## 2. Unzip model
```shell
$ cd models
$ unzip faster_rcnn_r101_fpn-20210420T043440Z-001 (1).zip
$ cd ..
```

## 3. Build Docker image
```shell
$ docker build -t face_detection_image ./
```

## 4. Run Container
```shell
$ docker run --name face_detection_container -d -p 5000:5000 face_detection_image
```