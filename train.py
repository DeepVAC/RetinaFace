import torch
from torch.utils.data import Dataset, DataLoader
from torch import nn
from torch import optim

import cv2
import math
import time
import numpy as np
import os
from deepvac import LOG, DeepvacTrain, is_ddp
from deepvac.syszux_loss import MultiBoxLoss
from deepvac.syszux_post_process import py_cpu_nms, decode, decode_landm, PriorBox
from aug.aug import RetinaAug
from modules.model import RetinaFaceMobileNet, RetinaFaceResNet
from synthesis.synthesis import RetinaTrainDataset, detection_collate

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
        self.auditConfig()
        priorbox = PriorBox(self.priorbox_cfgs, self.image_size)
        with torch.no_grad():
            self.priors = priorbox.forward()
            self.priors = self.priors.to(self.device)
        self.step_index = 0
    
    def auditConfig(self):
        self.lr_decay = [190, 220]
        self.image_size = (640, 640)

    def initNetWithCode(self):
        self.net = RetinaFaceMobileNet()
        self.net.to(self.conf.device)
        
    def initScheduler(self):
        pass
 
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
        self.loss = self.loc_weight * self.loss_l + self.loss_c + self.lmk_weight * self.loss_landm


    def doForward(self):
        self.output = self.net(self.sample)

    def preEpoch(self):
        if self.is_train:
            if self.epoch in self.lr_decay:
                self.step_index += 1
            self.adjust_learning_rate()

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
        self.setValContext()
        LOG.logI('Phase {} started...'.format(self.phase))
        #prepare the static quant
        self.exportStaticQuant(prepare=True)
        with torch.no_grad():
            self.preEpoch()
            self.postEpoch()
        self.saveState(self.getTime())

    def adjust_learning_rate(self):
        warmup_epoch = -1
        if self.epoch <= warmup_epoch:
            self.conf.lr = 1e-6 + (self.conf.lr-1e-6) * self.iter / (math.ceil(len(self.train_dataset)/self.conf.train.batch_size) * warmup_epoch)
        else:
            self.conf.lr = self.conf.lr * (self.conf.gamma ** (self.step_index))
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.conf.lr

class RetinaResNetTrain(RetinaMobileNetTrain):
    def auditConfig(self):
        self.lr_decay = [70, 90]
        self.image_size = (840, 840)

    def initNetWithCode(self):
        self.net = RetinaFaceResNet()
        self.net.to(self.conf.device)

if __name__ == "__main__":
    from config import config
    assert config.network == 'mobilenet' or config.network == 'resnet50', "config.network must be mobilenet or resnet50"
    if config.network == 'mobilenet':
        train = RetinaMobileNetTrain(config)
    else:
        train = RetinaResNetTrain(config)
    train()
