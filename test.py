import torch
import torch.nn.functional as F

import cv2
import numpy as np
import time

import deepvac
from deepvac import LOG, Deepvac
from deepvac.utils.face_utils import py_cpu_nms, decode, decode_landm, PriorBox

class RetinaTest(Deepvac):
    def __init__(self, deepvac_config):
        super(RetinaTest, self).__init__(deepvac_config)
        self.auditConfig()
        self.priorbox_cfgs = {
            'min_sizes': [[16, 32], [64, 128], [256, 512]],
            'steps': [8, 16, 32],
            'clip': False
        }
        self.variance = [0.1, 0.2]
    
    def auditConfig(self):
        pass

    def _post_process(self, preds):
        loc, cls, landms = preds
        conf = F.softmax(cls, dim=-1)
        
        priorbox = PriorBox(self.priorbox_cfgs, image_size=(self.img_raw.shape[0], self.img_raw.shape[1]))
        priors = priorbox.forward()
        priors = priors.to(self.config.device)
        prior_data = priors.data
        resize = 1
        scale = torch.Tensor([self.config.sample.shape[3], self.config.sample.shape[2], self.config.sample.shape[3], self.config.sample.shape[2]])
        scale = scale.to(self.config.device)
        boxes = decode(loc.data.squeeze(0), prior_data, self.variance)
        boxes = boxes * scale / resize
        boxes = boxes.cpu().numpy()
        scores = conf.squeeze(0).data.cpu().numpy()[:, 1]
        landms = decode_landm(landms.data.squeeze(0), prior_data, self.variance)
        scale1 = torch.Tensor([self.config.sample.shape[3], self.config.sample.shape[2], self.config.sample.shape[3], self.config.sample.shape[2],
                        self.config.sample.shape[3], self.config.sample.shape[2], self.config.sample.shape[3], self.config.sample.shape[2],
                        self.config.sample.shape[3], self.config.sample.shape[2]])
        scale1 = scale1.to(self.config.device)
        landms = landms * scale1 / resize
        landms = landms.cpu().numpy()

        # ignore low scores
        inds = np.where(scores > self.config.confidence_threshold)[0]
        boxes = boxes[inds]
        landms = landms[inds]
        scores = scores[inds]
        
        # keep top-K before NMS
        order = scores.argsort()[::-1][:self.config.top_k]
        boxes = boxes[order]
        landms = landms[order]
        scores = scores[order]
        
        # do NMS
        dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
        keep = py_cpu_nms(dets, self.config.nms_threshold)
        dets = dets[keep, :]
        landms = landms[keep]

        # keep top-K faster NMS
        dets = dets[:self.config.keep_top_k, :]
        landms = landms[:self.config.keep_top_k, :]
        
        assert len(dets)==len(landms), "the number of det and landm in the image must be equal."
        
        return dets, landms

    def preIter(self):
        self.img_raw = cv2.imread(self.config.target[0])

    def postIter(self):
        dets, landms = self._post_process(self.config.output)
        name = self.config.target[0].split('/')[-1]
        print('path: ', self.config.target[0])
        print('dets: ', dets)
        print('landms: ', landms)
    
    def doFeedData2Device(self):
        self.config.sample = self.config.sample.to(self.config.device)

if __name__ == "__main__":
    from config import config
   
    retina_test = RetinaTest(config.train)
    retina_test()
    
