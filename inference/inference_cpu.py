from detectron2.utils.logger import setup_logger
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
import detectron2
import torchvision
import numpy as np
import cv2

import torch
assert torch.__version__.startswith("1.8")


# Some basic setup:
# Setup detectron2 logger
setup_logger()

# import some common libraries
# from google.colab.patches import cv2_imshow

# import some common detectron2 utilities

# cfg already contains everything we've set previously. Now we changed it a little bit for inference:
cfg = get_cfg()

cfg.merge_from_file(model_zoo.get_config_file(
    "COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml"))
cfg.DATALOADER.NUM_WORKERS = 20
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # only has one class (face)

cfg.MODEL.DEVICE = 'cpu'  # if you want to inference with gpu then delete this line
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.8

cfg.OUTPUT_DIR = '../REST_API/models/faster_rcnn_r101_fpn/'  # set output data path
# path to the model we just trained
cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
predictor = DefaultPredictor(cfg)


def inference_face_detection(image_path):
    '''Receives an image and returns the position of the face'''
    img = cv2.imread(image_path)
    # format is documented at https://detectron2.readthedocs.io/tutorials/models.html#model-output-format
    outputs = predictor(img)
    v = Visualizer(img[:, :, ::-1],
                   metadata=MetadataCatalog.get("face_train"), scale=1)
    out = v.draw_instance_predictions(outputs["instances"].to("cpu"))

    return outputs
