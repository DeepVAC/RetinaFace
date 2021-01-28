import torch
from torch.utils.data import Dataset, DataLoader
from torch import nn
from torch import optim
import torch.nn.functional as F

import cv2
import math
import time
import numpy as np
import os

from deepvac.syszux_log import LOG
from deepvac.syszux_deepvac import DeepvacTrain
from deepvac.syszux_loss import MultiBoxLoss
from deepvac.syszux_post_process import py_cpu_nms, decode, decode_landm, PriorBox

from modules.model import RetinaFace

from aug.aug import RetinaAug
from synthesis.synthesis import RetinaTrainDataset, detection_collate

class DeepvacRetina(DeepvacTrain):
    def __init__(self, retina_config):
        super(DeepvacRetina, self).__init__(retina_config)
        priorbox = PriorBox(self.conf.cfg, image_size=self.conf.cfg['image_size'])
        with torch.no_grad():
            self.priors = priorbox.forward()
            self.priors = self.priors.to(self.device)
        self.step_index = 0
    
    def initNetWithCode(self):
        self.net = RetinaFace(self.conf.cfg)
        self.net.to(self.conf.device)
        if self.conf.cfg['ngpu'] > 1 and self.conf.cfg['gpu_train']:
            self.net = torch.nn.DataParallel(self.net)
        
    def initScheduler(self):
        pass
 
    def initCriterion(self):
        self.criterion = MultiBoxLoss(self.conf.cls_num, 0.35, True, 0, True, 7, 0.35, False, self.conf.device)
    
    def initTrainLoader(self):
        self.train_dataset = RetinaTrainDataset(self.conf.train.label_path, RetinaAug(self.conf))
        self.train_loader = DataLoader(self.train_dataset, batch_size=self.conf.train.batch_size, num_workers=self.conf.num_workers, shuffle=self.conf.train.shuffle, collate_fn=detection_collate)

    def initValLoader(self):
        self.val_dataset = None
        self.val_loader = None

    def initOptimizer(self):
        self.optimizer = optim.SGD(
                self.net.parameters(),
                lr=self.conf.lr,
                momentum=self.conf.momentum,
                weight_decay=self.conf.weight_decay,
        )

    def doLoss(self):
        if not self.is_train:
            return
        self.loss_l, self.loss_c, self.loss_landm = self.criterion(self.output, self.priors, self.target)
        self.loss = self.conf.cfg['loc_weight'] * self.loss_l + self.loss_c + 3 * self.loss_landm


    def doForward(self):
        self.output = self.net(self.sample)

    def preEpoch(self):
        if self.is_train:
            if self.epoch in [self.conf.cfg['decay1'], self.conf.cfg['decay2']]:
                self.step_index += 1
            self.adjust_learning_rate()
        else:
            self.face_count = 0
            self.pr_curve = 0

    def earlyIter(self):        
        start = time.time()
        self.sample = self.sample.to(self.device)
        self.target = [anno.to(self.device) for anno in self.target]
        if not self.is_train:
            return    
        self.data_cpu2gpu_time.update(time.time() - start)
        try:
            self.addGraph(self.sample)
        except:
            LOG.logW("Tensorboard addGraph failed. You network foward may have more than one parameters?")
            LOG.logW("Seems you need reimplement preIter function.")

    def preIter(self):
        pass

    def postIter(self):
        pass

    def postEpoch(self):
        if self.is_train:
            return
        self.accuracy = 1
        LOG.logI('Test accuray: {:.4f}'.format(self.accuracy))

    def processAccept(self):
        pass

    def processVal(self, smoke=False):
        pass

    def adjust_learning_rate(self):
        warmup_epoch = -1
        if self.epoch <= warmup_epoch:
            self.conf.lr = 1e-6 + (self.conf.lr-1e-6) * self.iter / (math.ceil(len(self.train_dataset)/self.conf.train.batch_size) * warmup_epoch)
        else:
            self.conf.lr = self.conf.lr * (self.conf.gamma ** (self.step_index))
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.conf.lr

if __name__ == "__main__":
    from config import config
    dr = DeepvacRetina(config)
    dr()