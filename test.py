import torch
import torch.nn.functional as F

import cv2
import numpy as np

from deepvac.syszux_log import LOG
from deepvac.syszux_deepvac import Deepvac
from deepvac.syszux_post_process import py_cpu_nms, decode, decode_landm, PriorBox

from modules.model import RetinaFaceMobileNet

class RetinaTestMobileNet(Deepvac):
    def __init__(self, deepvac_config):
        super(RetinaTestMobileNet, self).__init__(deepvac_config)
        self.auditConfig()
    
    def auditConfig(self):
        self.priorbox_cfgs = {
                'min_sizes': [[16, 32], [64, 128], [256, 512]],
                'steps': [8, 16, 32],
                'clip': False,
                }
        self.variance = [0.1, 0.2]

    def initNetWithCode(self):
        torch.set_grad_enabled(False)
        self.net = RetinaFaceMobileNet()

    def _pre_process(self, img_raw):
        h, w, c = img_raw.shape
        max_edge = max(h,w)
        if(max_edge > self.conf.max_edge):
            img_raw = cv2.resize(img_raw,(int(w * self.conf.max_edge / max_edge), int(h * self.conf.max_edge / max_edge)))

        img = np.float32(img_raw)

        im_height, im_width, _ = img.shape
        img -= self.conf.rgb_means
        img = img.transpose(2, 0, 1)
        self.input_tensor = torch.from_numpy(img).unsqueeze(0)
        self.input_tensor = self.input_tensor.to(self.device)
        self.img_raw = img_raw

    def _post_process(self, preds):
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
        
        assert len(dets)==len(landms), "the number of det and landm in the image must be equal."
        
        return dets, landms

    def __call__(self, image):
        self._pre_process(image)
        preds = self.net(self.input_tensor)

        return self._post_process(preds)

if __name__ == "__main__":
    from config import config as deepvac_config

    img = cv2.imread('./sample.jpg')
    retina_test = RetinaTestMobileNet(deepvac_config.test)
    dets, landms = retina_test(img)
    print('dets: ', dets)
    print('landms: ', landms)
