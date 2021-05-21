import math
import time
import os
import sys
import numpy as np
import cv2

import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

from deepvac import LOG, Deepvac, FaceReport
from deepvac.utils.face_utils import py_cpu_nms, decode, decode_landm, PriorBox

from modules.utils_align import AlignFace
from modules.utils_face_rec import FaceRecTest

class RetinaTest(Deepvac):
    def __init__(self, retina_config):
        super(RetinaTest, self).__init__(retina_config)
        self.align_face = AlignFace()
        self.auditConfig()
        self.priorbox_cfgs = {
            'min_sizes': [[16, 32], [64, 128], [256, 512]],
            'steps': [8, 16, 32],    
            'clip': False
        }
        self.variance = [0.1, 0.2]
    
    def auditConfig(self):
        pass

    def _pre_process(self, img_raw):
        h, w, c = img_raw.shape
        max_edge = max(h,w)
        if(max_edge > 2000):
            img_raw = cv2.resize(img_raw,(int(w * 2000.0 / max_edge), int(h * 2000.0 / max_edge)))

        img = np.float32(img_raw)

        im_height, im_width, _ = img.shape
        img -= (104, 117, 123)
        img = img.transpose(2, 0, 1)
        self.input_tensor = torch.from_numpy(img).unsqueeze(0)
        self.input_tensor = self.input_tensor.to(self.config.device)
        self.img_raw = img_raw

    def _processWithNoAlign(self, dets):
        detected_imgs = []
        for det in dets:
            det = [0 if d < 0 else d for d in det]
            detected_img = self.img_raw[int(det[1]):int(det[3]), int(det[0]):int(det[2])]
            detected_img = cv2.resize(detected_img, (112, 112))
            detected_imgs.append(detected_img)
        return detected_imgs

    def _processWithAlign(self, landms, dets, align_type):
        detected_imgs = []
        for landmark, det in zip(landms, dets):
            detected_img = self.align_face(self.img_raw, landmark, det, align_type)
            detected_imgs.append(detected_img)
        return detected_imgs

    def _post_process(self, preds, align_type):
        loc, cls, landms = preds
        conf = F.softmax(cls, dim=-1)
        
        priorbox = PriorBox(self.priorbox_cfgs, image_size=(self.img_raw.shape[0], self.img_raw.shape[1]))
        priors = priorbox.forward()
        priors = priors.to(self.config.device)
        prior_data = priors.data
        resize = 1
        scale = torch.Tensor([self.input_tensor.shape[3], self.input_tensor.shape[2], self.input_tensor.shape[3], self.input_tensor.shape[2]])
        scale = scale.to(self.config.device)
        boxes = decode(loc.data.squeeze(0), prior_data, self.variance)
        boxes = boxes * scale / resize
        boxes = boxes.cpu().numpy()
        scores = conf.squeeze(0).data.cpu().numpy()[:, 1]
        landms = decode_landm(landms.data.squeeze(0), prior_data, self.variance)
        scale1 = torch.Tensor([self.input_tensor.shape[3], self.input_tensor.shape[2], self.input_tensor.shape[3], self.input_tensor.shape[2],
                        self.input_tensor.shape[3], self.input_tensor.shape[2], self.input_tensor.shape[3], self.input_tensor.shape[2],
                        self.input_tensor.shape[3], self.input_tensor.shape[2]])
        scale1 = scale1.to(self.config.device)
        landms = landms * scale1 / resize
        landms = landms.cpu().numpy()

        # ignore low scores
        inds = np.where(scores > self.config.post_process.confidence_threshold)[0]
        boxes = boxes[inds]
        landms = landms[inds]
        scores = scores[inds]
        
        # keep top-K before NMS
        order = scores.argsort()[::-1][:self.config.post_process.top_k]
        # order = scores.argsort()[::-1][:args.top_k]
        boxes = boxes[order]
        landms = landms[order]
        scores = scores[order]
        
        # do NMS
        dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
        keep = py_cpu_nms(dets, self.config.post_process.nms_threshold)
        dets = dets[keep, :]
        landms = landms[keep]

        # keep top-K faster NMS
        dets = dets[:self.config.post_process.keep_top_k, :]
        landms = landms[:self.config.post_process.keep_top_k, :]
        if len(dets)==0:
            return []
        if align_type == 'no_align':
            return self._processWithNoAlign(dets)

        return self._processWithAlign(landms, dets, align_type)

    def __call__(self, image, align_type, det_net=None):
        assert align_type in ['align', 'no_align', 'warp_crop'], "align_type must in ['align', 'no_align', 'warp_crop']"
        self._pre_process(image)
        if det_net is not None:
            self.config.net = det_net
        preds = self.config.net(self.input_tensor)

        return self._post_process(preds, align_type)

class FaceTest(object):
    def __init__(self, deepvac_config, rec_config):
        self.config = deepvac_config
        self.face_rec = FaceRecTest(rec_config)
        self.face_det = RetinaTest(deepvac_config)
        
        self.reports = []
        self.names = []
        self.imgs = []
        self.paths = []

    def _detectPics(self, path, pics, align_type, det_net=None):
        for pic in pics:
            pic_path = os.path.join(path, pic)
            img_raw = cv2.imread(pic_path)
            if img_raw is None or img_raw.shape is None:
                LOG.logE('img:{} is readed error! You should get rid of it!'.format(pic_path), exit=True)
            det_res = self.face_det(img_raw, align_type, det_net)
            if len(det_res) == 0:
                continue
            self.imgs.extend(det_res)
            self.names.append(pic_path.split('/')[-2])
            self.paths.append(pic_path)

    def faceDet(self, det_dir, align_type, det_net=None):
        persons = os.listdir(det_dir)

        self.names = []
        self.imgs = []
        self.paths = []
        for person in persons:
            det_person_path = os.path.join(det_dir, person)
            if not os.path.isdir(det_person_path):
                continue
            pics = os.listdir(det_person_path)
            self._detectPics(det_person_path, pics, align_type, det_net)

    def faceRec(self, prefix):
        report = FaceReport(prefix, len(self.imgs))
        for idx, img in enumerate(self.imgs):
            LOG.logI('path: {}'.format(self.paths[idx]))
            infos = self.paths[idx].split('/')
            person, img_name = infos[-2], infos[-1]
            self.face_rec.setInputImg(img)
            pred, _ = self.face_rec()
            label = self.names[idx]
            LOG.logI('label: {}'.format(label))
            LOG.logI('pred: {}'.format(pred))
            report.add(label, pred)
        self.reports.append(report)

    def makeDB(self, align_type, det_net=None):
        imgs = []
        names = []
        paths = []
        for i in range(len(self.config.core.post_process.db_dirs)):
            self.faceDet(self.config.core.post_process.db_dirs[i], align_type, det_net)
            imgs.extend(self.imgs)
            names.extend(self.names)
            paths.extend(self.paths)
        self.face_rec.makeDB(imgs, names, paths)

    def printReports(self):
        for report in self.reports:
            report()

    def __call__(self, det_net=None):
        assert len(self.config.core.post_process.test_dirs) == len(self.config.core.post_process.test_prefix), "test_dirs and test_prefix must have same len."
        for align_type in self.config.core.post_process.align_type:
            LOG.logI('make DB begin.')
            self.makeDB(align_type)
            LOG.logI('make DB finish..')
            self.reports = []
            for i in range(len(self.config.core.post_process.test_dirs)):
                self.faceDet(self.config.core.post_process.test_dirs[i], align_type, det_net)
                self.faceRec(self.config.core.post_process.test_prefix[i])
            self.printReports()

if __name__ == "__main__":
    from config import config as deepvac_config
    face_test = FaceTest(deepvac_config)
    face_test()
