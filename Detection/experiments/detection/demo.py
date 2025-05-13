import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"]="1"

# Some basic setup:
# Setup detectron2 logger
import detectron2
from detectron2.utils.logger import setup_logger
from detectron2.data import MetadataCatalog
setup_logger()

# import some common libraries
import numpy as np
import torch
import matplotlib.pyplot as plt
import os, json, cv2, random, sys, traceback
import skimage.io as io

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog, DatasetCatalog
from visualizer import Visualizer
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import pycocotools.mask as maskUtils
import pickle as pkl

from UQHeads import UQHeads

import pdb
from tqdm import tqdm



def remove_whitespace(img):
    tocut = (((img > 250.).astype(float).sum(axis=2)) == 3).astype(float) 
    rows_tocut = tocut.sum(axis=1) == img.shape[1]
    cols_tocut = tocut.sum(axis=0) == img.shape[0]
    img = img[~rows_tocut]
    img = img[:,~cols_tocut]
    return img


if __name__ == "__main__":
    with torch.no_grad():
        print(detectron2.__file__)
        # Evaluations
        annType = ['segm', 'bbox', 'keypoints']
        annType = annType[0]  # specify type here
        dataType = 'val2017'
        prefix = 'person_keypoints' if annType == 'keypoints' else 'instances'
        dataDir = '.'
        annFile = '%s/annotations/%s_%s.json' % (dataDir, prefix, dataType)
        cocoGt = COCO(annFile)

        # Get the label mapping from COCO to detectron2 standard
        label_map = MetadataCatalog['coco_2017_val'].thing_dataset_id_to_contiguous_id

        # Load the model
        cfg = get_cfg()
        cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
        cfg.MODEL.ROI_HEADS.NAME = "UQHeads"
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
        cfg.MODEL.ROI_HEADS.APS_THRESH = 0.99817866  # set threshold for this model
        cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
        cfg.MODEL.DEVICE = "cpu"  # Force model to run on CPU

        predictor = DefaultPredictor(cfg)

        # Process each image
        for img_id in [1425, 70254, 292102, 470618, 309615, 309391, 178749, 196503, 68203]:
            try:
                img_metadata = cocoGt.loadImgs(img_id)[0]
                img = io.imread('%s/%s/%s' % (dataDir, dataType, img_metadata['file_name']))
            except:
                img = io.imread(f"./test_data/{img_id}.jpg")

            if len(img.shape) < 3:
                img = img[:, :, None]

            ann_ids = cocoGt.getAnnIds(imgIds=[img_id, ])
            anns = cocoGt.loadAnns(ann_ids)

            try:
                outputs = predictor(img)
            except:
                extype, value, tb = sys.exc_info()
                traceback.print_exc()
                pdb.post_mortem(tb)
                print(f"Image {img_id} didn't work.")

                # Ensure everything is on cpu
            if len(outputs['instances']) == 0:
                continue
            outputs['instances'] = outputs['instances'].to('cpu')
            tokeep = outputs["instances"].softmax_outputs.max(dim=1)[0] > 0.49
            outputs['instances'].roi_masks.tensor = outputs['instances'].roi_masks.tensor[tokeep]
            outputs['instances'].pred_boxes.tensor = outputs['instances'].pred_boxes.tensor[tokeep]
            outputs['instances'].pred_sets = outputs['instances'].pred_sets[tokeep]
            outputs['instances'].pred_masks = outputs['instances'].roi_masks.to_bitmasks(
            outputs['instances'].pred_boxes, img.shape[0], img.shape[1], 0.31189948).tensor
            outputs['instances'].softmax_outputs = outputs['instances'].softmax_outputs[tokeep]
            outputs['instances'].scores = outputs['instances'].scores[tokeep]
            outputs['instances'].class_ordering = outputs['instances'].class_ordering[tokeep]


            # Visualization and saving
            v = Visualizer(img[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
            out = v.draw_instance_predictions(outputs["instances"])

            os.makedirs('./outputs/', exist_ok=True)
            dpi = 400
            outImg = out.get_image()[:, :, ::-1]
            outImg = cv2.resize(outImg, dsize=(img.shape[1], img.shape[0]), interpolation=cv2.INTER_CUBIC)
            diffR = outImg.shape[0] - img.shape[0]
            diffC = outImg.shape[1] - img.shape[1]
            startR = diffR // 2
            startC = diffC // 2
            endR = startR + img.shape[0]
            endC = startC + img.shape[1]
            outImg = np.concatenate((img, outImg[startR:endR, startC:endC]), axis=1)
            fig = plt.figure(figsize=(outImg.shape[1] / dpi, outImg.shape[0] / dpi))
            plt.imshow(outImg, interpolation='nearest')
            ax = plt.gca()
            ax.axis("off")
            plt.margins(0, 0)
            plt.savefig(f'outputs/{img_id}.jpg', dpi=1000, pad_inches=0, bbox_inches=0)
            plt.close(fig)
