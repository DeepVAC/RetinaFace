import time
import os
import numpy as np
from PIL import Image
import cv2
import torch
from torchvision import transforms
from deepvac import Deepvac, LOG

class FaceRecTest(Deepvac):
    def __init__(self, deepvac_config):
        super(FaceRecTest,self).__init__(deepvac_config)
        self.db = None
        self.transformer = self.config.transform
        self.tensor_list = []
        self.idx_name_map = {}
        self.config.net.to(self.config.device)


    def makeDB(self, imgs, names, paths):
        db_emb = torch.Tensor().to(self.config.device)
        LOG.log(LOG.S.I, "start make db")
        for idx, ori in enumerate(imgs):
            emb = self._inference(ori)
            db_emb = torch.cat((db_emb, emb))
            if idx % 10000 == 0 and idx != 0:
                LOG.log(LOG.S.I, "gen db features: {}".format(idx))

        LOG.log(LOG.S.I, "gen db sucessfully")
        self.db_emb = db_emb.to(self.config.device)
        self.db_names = names
        self.db_paths = paths

    def setInputImg(self, img):
        self.input_img = img

    def _inference(self, ori):
        with torch.no_grad():
            img = self.transformer(Image.fromarray(ori))
            img = torch.unsqueeze(img,0).to(self.config.device)
            emb = self.config.net(img)
        return emb

    def _compare_cos(self, emb):
        theta = torch.sum(torch.mul(self.db_emb, emb), axis=1)
        max_index = torch.argmax(theta).item()
        return theta[max_index], max_index

    def testFly(self):
        emb = self._inference(self.input_img).to(self.config.device)
        max_distance, max_index = self._compare_cos(emb)
        name = self.db_names[max_index] if  max_distance > self.config.threshold else None

        return (name, self.db_paths[max_index])
