
import cv2
import random
from flask import Flask, request, render_template, jsonify
import json
import os
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.utils.visualizer import Visualizer
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2 import model_zoo
import numpy as np
from detectron2.utils.logger import setup_logger
import detectron2
import torch
import torchvision
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

cfg.MODEL.DEVICE = 'cpu'  # set cpu
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.8

cfg.OUTPUT_DIR = './models/faster_rcnn_r101_fpn/'  # set output data path
# path to the model we just trained
cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
predictor = DefaultPredictor(cfg)


def inference_face_detection(img):
    '''Receives an image and returns the position of the face'''
    # img = cv2.imread(image_path)
    # format is documented at https://detectron2.readthedocs.io/tutorials/models.html#model-output-format
    outputs = predictor(img)
    v = Visualizer(img[:, :, ::-1],
                   metadata=MetadataCatalog.get("face_train"), scale=1)
    out = v.draw_instance_predictions(outputs["instances"].to("cpu"))

    return outputs


app = Flask(__name__)

APP_ROOT = os.path.dirname(os.path.abspath(__file__))


@app.route('/')
def hello():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def upload_file():
    if request.method == 'POST':

        f = request.files['image']
        img_bytes = f.read()
        img = cv2.imdecode(np.fromstring(
            img_bytes, np.uint8), cv2.IMREAD_UNCHANGED)

        try:
            outputs = inference_face_detection(img)
            box_lst = [box.tolist()
                       for box in outputs["instances"]._fields['pred_boxes']]
            box_lst = str(box_lst)
        except:

            return jsonify({'1': 'Error'})

        return jsonify({'bbox': box_lst})


if __name__ == '__main__':
    app.run(port=5000, host='0.0.0.0')
