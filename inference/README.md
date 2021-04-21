# Face Detection Inference

## 1. clone and unzip model
```shell
$ git clone https://github.com/J-TKim/Face_Detection.git
$ cd Face_Detection/REST_API/models
$ unzip unzip faster_rcnn_r101_fpn-20210420T043440Z-001 (1).zip
```


## 1. install requirements
```shell
$ pip install torch==1.8.1+cpu torchvision==0.9.1+cpu torchaudio==0.8.1 -f https://download.pytorch.org/whl/torch_stable.html
$ python -m pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cpu/torch1.8/index.html
$ pip install opencv-python
```
If you want to use this code in gpu then install gpu version detectron, and torch from 
[detectron2 Installation](https://detectron2.readthedocs.io/en/latest/tutorials/install.html?highlight=cuda)

## 2. unzip model
```shell

```
## How to use code
```python
from inference_cpu.py import inference_face_detection


path = <image_path>
output = inference_face_detection(path)
```