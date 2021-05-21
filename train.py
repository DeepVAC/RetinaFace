import os
import math
import time
import numpy as np
import cv2

import torch
from torch import nn
from torch import optim

from deepvac import LOG, DeepvacTrain
from deepvac.utils.face_utils import py_cpu_nms, decode, decode_landm, PriorBox

from modules.model_retina import RetinaFaceMobileNet, RetinaFaceResNet
from modules.utils_face_test import FaceTest
from config import config, rec_config

class RetinaTrain(DeepvacTrain):
    def __init__(self, deepvac_config):
        super(RetinaTrain, self).__init__(deepvac_config)
        self.loc_weight = 2
        self.lmk_weight = 3
        self.priorbox_cfgs = {
            'min_sizes': [[16, 32], [64, 128], [256, 512]],
            'steps': [8, 16, 32],
            'clip': False,                                               
        }
        priorbox = PriorBox(self.priorbox_cfgs, self.config.image_size)
        with torch.no_grad():
            self.priors = priorbox.forward()
            self.priors = self.priors.to(self.config.device)
        self.step_index = 0

    def doLoss(self):
        if not self.config.is_train:
            return
        self.loss_l, self.loss_c, self.loss_landm = self.config.criterion(self.config.output, self.priors, self.config.target)
        self.config.loss = self.loc_weight * self.loss_l + self.loss_c + self.lmk_weight * self.loss_landm

    def doFeedData2Device(self):
        self.config.target = [anno.to(self.config.device) for anno in self.config.target]
        self.config.sample = self.config.sample.to(self.config.device)

    def processAccept(self):
        face_test = FaceTest(self.deepvac_config, rec_config)
        face_test()

if __name__ == "__main__":
    train = RetinaTrain(config)
    train()
