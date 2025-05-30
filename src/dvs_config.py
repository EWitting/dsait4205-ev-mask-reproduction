import os
import sys
import random
import math
import re
import time
import numpy as np
import cv2
import matplotlib
import matplotlib.pyplot as plt

# Root directory of the project
ROOT_DIR = os.path.abspath("../")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn.config import Config
from mrcnn import utils
import mrcnn.model as modellib
from mrcnn import visualize
from mrcnn.model import log

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "model")

# Local path to trained weights file
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")

# Download COCO trained weights from Releases if needed
if not os.path.exists(COCO_MODEL_PATH):
    utils.download_trained_weights(COCO_MODEL_PATH)


class DvsConfig(Config):
    """
    Configuration for training on the N-MNIST dataset.
    Derived from the base Config class.
    """
    # Give the configuration a recognizable name
    NAME = "N-MNIST-DVS"
    MODE = 'RGBD'

    # Train on 1 GPU and 8 images per GPU. Batch size is 8 (GPUs * images/GPU).
    GPU_COUNT = 1
    IMAGES_PER_GPU = 12

    # Number of classes (including background)
    NUM_CLASSES = 1 + 10  # background + 10 digits

    # N-MNIST images are of shape (34, 34) -> Rescaling to 64x64 such that it's a power of 2
    IMAGE_MIN_DIM = 64
    IMAGE_MAX_DIM = 64

    # Use smaller anchors because our image and objects are small
    RPN_ANCHOR_SCALES = (4, 8, 16, 32, 64)  # anchor side in pixels

    # Reduce training ROIs per image because the images are small and have
    # few objects. Aim to allow ROI sampling to pick 33% positive ROIs.
    TRAIN_ROIS_PER_IMAGE = 16

    # Use a small epoch since the data is simple
    STEPS_PER_EPOCH = 100

    # Use small validation steps since the epoch is small
    VALIDATION_STEPS = 5

    # Try with resnet50 first for faster training
    BACKBONE = "resnet50"

    # Shady stuff, leave it as false for easy display
    USE_MINI_MASK = False

    # Number of color channels per image. RGB = 3, grayscale = 1, RGB-D = 4
    # Changing this requires other changes in the code. See the WIKI for more
    # details: https://github.com/matterport/Mask_RCNN/wiki
    IMAGE_CHANNEL_COUNT = 4

    # From: https://github.com/orestis-z/mask-rcnn-rgbd/blob/d590e0f5085f8cbe895a6698e284426fd0116aa4/instance_segmentation/sceneNet/train.py#L66-L85
    MEAN_PIXEL = np.array([123.7, 116.8, 103.9, 127.5]) #

    # Loss weights for more precise optimization.
    # Can be used for R-CNN training setup.
    LOSS_WEIGHTS = {
        "rpn_class_loss": 1.,
        "rpn_bbox_loss": 1.,
        "mrcnn_class_loss": 1.,
        "mrcnn_bbox_loss": 1.,
        "mrcnn_mask_loss": 5.
    }

class InferenceConfig(DvsConfig):
        GPU_COUNT = 1
        IMAGES_PER_GPU = 1

config = DvsConfig()
config.display()







