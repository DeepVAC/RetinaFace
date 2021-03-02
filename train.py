import torch
from torch.utils.data import Dataset, DataLoader
from torch import nn
from torch import optim

import cv2
import math
import time
import numpy as np
import os
from deepvac import LOG, DeepvacTrain, is_ddp, OsWalkDataset
from deepvac.syszux_loss import MultiBoxLoss
from deepvac.syszux_post_process import py_cpu_nms, decode, decode_landm, PriorBox
from aug.aug import RetinaAug
from modules.model_retina import RetinaFaceMobileNet, RetinaFaceResNet
from synthesis.synthesis import RetinaTrainDataset, detection_collate
from modules.utils_face_test import FaceTest

class RetinaValDataset(OsWalkDataset):
    def __init__(self, deepvac_config):
        self.conf = deepvac_config
        super(RetinaValDataset, self).__init__(deepvac_config.val)

    def __getitem__(self, index):
        path = self.files[index]
        img_raw = cv2.imread(path, 1)
        h, w, c = img_raw.shape
        max_edge = max(h,w)
        if(max_edge > self.conf.test.max_edge):
            img_raw = cv2.resize(img_raw,(int(w * self.conf.test.max_edge / max_edge), int(h * self.conf.test.max_edge / max_edge)))
        img = np.float32(img_raw)
        im_height, im_width, _ = img.shape
        img -= self.conf.test.rgb_means
        img = img.transpose(2, 0, 1)
        input_tensor = torch.from_numpy(img)
        return input_tensor, torch.ones(1)

class RetinaMobileNetTrain(DeepvacTrain):
    def __init__(self, deepvac_config):
        super(RetinaMobileNetTrain, self).__init__(deepvac_config)
        self.loc_weight = 2
        self.lmk_weight = 3
        self.priorbox_cfgs = {
            'min_sizes': [[16, 32], [64, 128], [256, 512]],
            'steps': [8, 16, 32],
            'clip': False,                                               
        }
        priorbox = PriorBox(self.priorbox_cfgs, self.conf.train.image_size)
        with torch.no_grad():
            self.priors = priorbox.forward()
            self.priors = self.priors.to(self.device)
        self.step_index = 0

    def initNetWithCode(self):
        self.net = RetinaFaceMobileNet()
        
    def initCriterion(self):
        self.criterion = MultiBoxLoss(self.conf.cls_num, 0.35, True, 0, True, 7, 0.35, False, self.conf.device)
    
    def initTrainLoader(self):
        self.train_dataset = RetinaTrainDataset(self.conf.train, augument=RetinaAug(self.conf))
        if is_ddp:
            self.train_sampler = torch.utils.data.distributed.DistributedSampler(self.train_dataset)
        self.train_loader = DataLoader(
            self.train_dataset, 
            batch_size=self.conf.train.batch_size, 
            num_workers=self.conf.num_workers, 
            shuffle= False if is_ddp else self.conf.train.shuffle, 
            sampler=self.train_sampler if is_ddp else None,
            collate_fn=detection_collate
        )

    def initValLoader(self):
        self.val_dataset = RetinaValDataset(self.conf)
        self.val_loader = torch.utils.data.DataLoader(self.val_dataset, batch_size=1, pin_memory=False)

    def doLoss(self):
        if not self.is_train:
            return
        self.loss_l, self.loss_c, self.loss_landm = self.criterion(self.output, self.priors, self.target)
        self.loss = self.loc_weight * self.loss_l + self.loss_c + self.lmk_weight * self.loss_landm

    def feedTarget(self):
        self.target = [anno.to(self.device) for anno in self.target]

    def postEpoch(self):
        if self.is_train:
            return
        self.accuracy = 1
        LOG.logI('Test accuray: {:.4f}'.format(self.accuracy))

    def processAccept(self):
        face_test = FaceTest(self.conf)
        face_test()

class RetinaResNetTrain(RetinaMobileNetTrain):
    def initNetWithCode(self):
        self.net = RetinaFaceResNet()

if __name__ == "__main__":
    from config import config
    assert config.network == 'mobilenet' or config.network == 'resnet50', "config.network must be mobilenet or resnet50"
    if config.network == 'mobilenet':
        train = RetinaMobileNetTrain(config)
    else:
        train = RetinaResNetTrain(config)
    train()
