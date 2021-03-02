import cv2
import os
import numpy as np
import time
import torch
from PIL import Image
from torchvision import transforms
from deepvac import Deepvac, LOG
from modules.model_is import MobileFaceNet, Resnet50IR

class ResnetFaceTest(Deepvac):
    def __init__(self, deepvac_config):
        super(ResnetFaceTest,self).__init__(deepvac_config)
        self.db = None
        self.transformer = self.conf.transform
        self.tensor_list = []
        self.idx_name_map = {}

    def initNetWithCode(self):
        self.net = Resnet50IR(self.conf.embedding_size)

    def makeDB(self, imgs, names, paths):
        db_emb = torch.Tensor().to(self.device)
        LOG.log(LOG.S.I, "start make db")
        for idx, ori in enumerate(imgs):
            emb = self._inference(ori)
            db_emb = torch.cat((db_emb, emb))
            if idx % 10000 == 0 and idx != 0:
                LOG.log(LOG.S.I, "gen db features: {}".format(idx))

        LOG.log(LOG.S.I, "gen db sucessfully")
        self.db_emb = db_emb.to(self.conf.device)
        self.db_names = names
        self.db_paths = paths

    def process(self):            
        if not isinstance(self.input_output['input'], list):
            LOG.log(LOG.S.E, "illegal input of ISFace: {}".format(type(self.input_output['input'])))
            return None

        for ori in self.input_output['input']:
            emb = self._inference(ori).to(self.conf.device)
            max_distance, max_index = self._compare_cos(emb)
            name = self.db_names[max_index] if  max_distance > self.getConf().threshold else "a_stranger"

            self.addOutput([name,self.db_paths[max_index]])

    def _inference(self, ori):
        with torch.no_grad():
            img = self.transformer(Image.fromarray(ori))
            img = torch.unsqueeze(img,0).to(self.device)
            emb = self.net(img)
        return emb

    def _compare_cos(self, emb):
        theta = torch.sum(torch.mul(self.db_emb, emb), axis=1)
        max_index = torch.argmax(theta).item()
        return theta[max_index], max_index

class MobileFaceTest(ResnetFaceTest):
    def __init__(self, deepvac_config):
        super(MobileFaceTest,self).__init__(deepvac_config)
    
    def initNetWithCode(self):
        self.net = MobileFaceNet(self.conf.embedding_size)

def isYou(face, img_dir_list):
    for img_dir in img_dir_list:
        imgs = getImages(img_dir)
        report = FaceReport(img_dir.split('/')[-2], len(imgs))
        for img in imgs:
            ori = cv2.imread(img)
            output = face(ori)
            report.add(img.split('/')[-2], output[0])
        report()


if __name__ == "__main__":
    from config import config as deepvac_config
    from deepvac.syszux_report import FaceReport

    models = {
        "Mobile":  MobileFaceTest,
        "Resnet":  ResnetFaceTest
    }
    
    face = models[deepvac_config.model_mode](deepvac_config)
    face.makeDB(deepvac_config.ori_db_path)
    isYou(face, deepvac_config.test_img_path)

