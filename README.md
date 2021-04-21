# Face Detection with Detectron2

## 1. Install detectron2
[detectron2 Installation](https://detectron2.readthedocs.io/en/latest/tutorials/install.html?highlight=cuda)

## 2. Import modules
```python
# Some basic setup:
# Setup detectron2 logger
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common libraries
import numpy as np
import os, json, cv2, random
from google.colab.patches import cv2_imshow

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
```

## 3. Load dataset (json)
data example
```python
{'annotations': [{'bbox': [53, 43, 70, 98],
   'bbox_mode': <BoxMode.XYWH_ABS: 1>,
   'category_id': 0}],
 'file_name': '/content/drive/MyDrive/Colab_Notebooks/dataset/od_custom/train/101.jpg',
 'image_id': '101.jpg50451'}
```

### visualization 101.jpg
<img src="https://github.com/J-TKim/Face_Detection/blob/main/images/train/train101.png?raw=true">

## 4. Visualizing dataset (train)
<img src="https://github.com/J-TKim/Face_Detection/blob/main/images/train/train1.png?raw=true">

<br/>

<img src="https://github.com/J-TKim/Face_Detection/blob/main/images/train/train2.png?raw=true">

## 6. Set Model, Hyperparameter & Train model

```python
from detectron2.engine import DefaultTrainer

cfg = get_cfg()

cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml"))
cfg.DATASETS.TRAIN = ("face_train",)
cfg.DATASETS.TEST = ()
cfg.DATALOADER.NUM_WORKERS = 6
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml")  # Let training initialize from model zoo
cfg.SOLVER.IMS_PER_BATCH = 6
cfg.SOLVER.BASE_LR = 0.01  # pick a good LR
cfg.SOLVER.MAX_ITER = 500
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128   # faster, and good enough for this toy dataset (default: 512)
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # only has one class (face)

# cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.8

model_path = PATH + "models/faster_rcnn_r101_fpn"
cfg.OUTPUT_DIR = model_path
os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

trainer = DefaultTrainer(cfg)
trainer.resume_or_load(resume=False)
trainer.train()
```

You can choose another model from 
[detectron2 model zoo](https://github.com/yunjey/pytorch-tutorial/tree/master/tutorials/03-advanced/image_captioning)

## 7.Load model to test

```python
# cfg already contains everything we've set previously. Now we changed it a little bit for inference:
cfg = get_cfg()

cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml"))
cfg.DATALOADER.NUM_WORKERS = 20
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # only has one class (face)

# cfg.MODEL.DEVICE='cpu'
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.8

model_path = PATH + 'models/faster_rcnn_r101_fpn'
cfg.OUTPUT_DIR = model_path
cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")  # path to the model we just trained
predictor = DefaultPredictor(cfg)
```

## 8. Test with test another dataset

<img src="https://github.com/J-TKim/Face_Detection/blob/main/images/test/test1.png?raw=true">
<img src="https://github.com/J-TKim/Face_Detection/blob/main/images/test/test2.png?raw=true">
<img src="https://github.com/J-TKim/Face_Detection/blob/main/images/test/test3.png?raw=true">
<img src="https://github.com/J-TKim/Face_Detection/blob/main/images/test/test4.png?raw=true">
<img src="https://github.com/J-TKim/Face_Detection/blob/main/images/test/test5.png?raw=true">