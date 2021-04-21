# Face Detection REST API with Docker

## 1. clone repo and move directory
```shell
$ git clone https://github.com/J-TKim/Face_Detection.git
$ cd Face_Detection/REST_API
```

## 2. Build Docker image
```shell
$ docker build -t face_detection_image ./
```

## 3. Run Container
```shell
$ docker run --name face_detection_container -d -p 5000:5000 face_detection_image
```