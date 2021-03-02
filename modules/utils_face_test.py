import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import cv2
import math
import time
import numpy as np
import os

from deepvac import LOG, Deepvac, FaceReport
from deepvac.syszux_post_process import py_cpu_nms, decode, decode_landm, PriorBox

from modules.model_retina import RetinaFaceMobileNet, RetinaFaceResNet
from modules.utils_align import AlignFace
from modules.utils_face_rec import MobileFaceTest
import sys

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

    def initNetWithCode(self):
        #torch.set_grad_enabled(False)
        self.net = RetinaFaceMobileNet()

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
        self.input_tensor = self.input_tensor.to(self.device)
        self.img_raw = img_raw

    def _post_process(self, preds, align_type):
        loc, cls, landms = preds
        conf = F.softmax(cls, dim=-1)
        
        priorbox = PriorBox(self.priorbox_cfgs, image_size=(self.img_raw.shape[0], self.img_raw.shape[1]))
        priors = priorbox.forward()
        priors = priors.to(self.device)
        prior_data = priors.data
        resize = 1
        scale = torch.Tensor([self.input_tensor.shape[3], self.input_tensor.shape[2], self.input_tensor.shape[3], self.input_tensor.shape[2]])
        scale = scale.to(self.device)
        boxes = decode(loc.data.squeeze(0), prior_data, self.variance)
        boxes = boxes * scale / resize
        boxes = boxes.cpu().numpy()
        scores = conf.squeeze(0).data.cpu().numpy()[:, 1]
        landms = decode_landm(landms.data.squeeze(0), prior_data, self.variance)
        scale1 = torch.Tensor([self.input_tensor.shape[3], self.input_tensor.shape[2], self.input_tensor.shape[3], self.input_tensor.shape[2],
                        self.input_tensor.shape[3], self.input_tensor.shape[2], self.input_tensor.shape[3], self.input_tensor.shape[2],
                        self.input_tensor.shape[3], self.input_tensor.shape[2]])
        scale1 = scale1.to(self.device)
        landms = landms * scale1 / resize
        landms = landms.cpu().numpy()

        # ignore low scores
        inds = np.where(scores > self.conf.confidence_threshold)[0]
        boxes = boxes[inds]
        landms = landms[inds]
        scores = scores[inds]
        
        # keep top-K before NMS
        order = scores.argsort()[::-1][:self.conf.top_k]
        # order = scores.argsort()[::-1][:args.top_k]
        boxes = boxes[order]
        landms = landms[order]
        scores = scores[order]
        
        # do NMS
        dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
        keep = py_cpu_nms(dets, self.conf.nms_threshold)
        dets = dets[keep, :]
        landms = landms[keep]

        # keep top-K faster NMS
        dets = dets[:self.conf.keep_top_k, :]
        landms = landms[:self.conf.keep_top_k, :]
        if len(dets)==0:
            return []
        detected_imgs = []
        if align_type == 'no_align':
            for det in dets:
                for det_i in range(len(det)):
                    if det[det_i] < 0:
                        det[det_i] = 0
                detected_img = self.img_raw[int(det[1]):int(det[3]), int(det[0]):int(det[2])]
                detected_img = cv2.resize(detected_img, (112, 112))
                detected_imgs.append(detected_img)

        else:
            for landmark, det in zip(landms, dets):
                detected_img = self.align_face(self.img_raw, landmark, det, align_type)
                detected_imgs.append(detected_img)

        return detected_imgs

    def __call__(self, image, align_type):
        assert align_type in ['align', 'no_align', 'warp_crop'], "align_type must in ['align', 'no_align', 'warp_crop']"
        self._pre_process(image)

        tic = time.time()
        preds = self.net(self.input_tensor)
        end = time.time() - tic
        print('net forward time: {:.4f}'.format(time.time() - tic))

        return self._post_process(preds, align_type)

class FaceTest(object):
    def __init__(self, deepvac_config):
        self.conf = deepvac_config
        self.face_rec = MobileFaceTest(deepvac_config.face)
        self.face_det = RetinaTest(deepvac_config.test)
        
        self.reports = []
        self.names = []
        self.imgs = []
        self.paths = []

    def faceDet(self, det_dir, align_type):
        persons = os.listdir(det_dir)

        self.names = []
        self.imgs = []
        self.paths = []
        for person in persons:
            det_person_path = os.path.join(det_dir, person)
            pics = os.listdir(det_person_path)
            for pic in pics:
                pic_path = os.path.join(det_person_path, pic)
                print(pic_path)
                img_raw = cv2.imread(pic_path)

                if img_raw is None or img_raw.shape is None:
                    print('img:[' + pic_path + "] is readed error! You should get rid of it!")
                    continue
                self.imgs.extend(self.face_det(img_raw, align_type))
                self.names.append(pic_path.split('/')[-2])
                self.paths.append(pic_path)

    def faceRec(self, prefix):
        report = FaceReport(prefix, len(self.imgs))
        for idx, img in enumerate(self.imgs):
            print(self.paths[idx])
            infos = self.paths[idx].split('/')
            person, img_name = infos[-2], infos[-1]
            output = self.face_rec(img)
            if len(output) == 0:
                pred = None
            else:
                pred = output[0]
            label = self.names[idx]
            print('label:', label)
            print('pred:', pred)
            report.add(label, pred)
        self.reports.append(report)

    def makeDB(self, align_type):
        imgs = []
        names = []
        paths = []
        for i in range(len(self.conf.test.db_dirs)):
            self.faceDet(self.conf.test.db_dirs[i], align_type)
            imgs.extend(self.imgs)
            names.extend(self.names)
            paths.extend(self.paths)
        self.face_rec.makeDB(imgs, names, paths)

    def printReports(self):
        for report in self.reports:
            report()

    def __call__(self):
        assert len(self.conf.test.test_dirs) == len(self.conf.test.test_prefix), "test_dirs and test_prefix must have same len."
        for align_type in self.conf.test.align_type:
            print('make DB begin.')
            self.makeDB(align_type)
            print('make DB finish..')
            self.reports = []
            for i in range(len(self.conf.test.test_dirs)):
                self.faceDet(self.conf.test.test_dirs[i], align_type)
                self.faceRec(self.conf.test.test_prefix[i])
            self.printReports()



if __name__ == "__main__":
    from config import config as deepvac_config

    deepvac_config.test.test_dirs = ['/opt/private/deepvac_face_1.0_test/ipc7/ds']
    deepvac_config.test.test_prefix = ['ipc']
    
    deepvac_config.test.db_dirs = ['/opt/private/deepvac_face_1.0_test/ipc7/db']

    face_test = FaceTest(deepvac_config)
    face_test()
