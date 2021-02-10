import torch
import torch.nn.functional as F

import cv2
import numpy as np

from deepvac import LOG, Deepvac, OsWalkDataset
from deepvac.syszux_post_process import py_cpu_nms, decode, decode_landm, PriorBox
from modules.model import RetinaFaceMobileNet, RetinaFaceResNet

class RetinaTestDataset(OsWalkDataset):
    def __init__(self, deepvac_config):
        self.conf = deepvac_config
        super(RetinaTestDataset, self).__init__(deepvac_config.test)

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

        return input_tensor, path

class RetinaMobileNetTest(Deepvac):
    def __init__(self, deepvac_config):
        super(RetinaMobileNetTest, self).__init__(deepvac_config)
        self.auditConfig()
        self.priorbox_cfgs = {
            'min_sizes': [[16, 32], [64, 128], [256, 512]],
            'steps': [8, 16, 32],
            'clip': False
        }
        self.variance = [0.1, 0.2]
        self.initTestLoader()
    
    def auditConfig(self):
        pass

    def initNetWithCode(self):
        torch.set_grad_enabled(False)
        self.net = RetinaFaceMobileNet()

    def initTestLoader(self):
        self.test_dataset = RetinaTestDataset(self.conf)
        self.test_loader = torch.utils.data.DataLoader(self.test_dataset, batch_size=1, pin_memory=False)

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
        inds = np.where(scores > self.conf.test.confidence_threshold)[0]
        boxes = boxes[inds]
        landms = landms[inds]
        scores = scores[inds]
        
        # keep top-K before NMS
        order = scores.argsort()[::-1][:self.conf.test.top_k]
        boxes = boxes[order]
        landms = landms[order]
        scores = scores[order]
        
        # do NMS
        dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
        keep = py_cpu_nms(dets, self.conf.test.nms_threshold)
        dets = dets[keep, :]
        landms = landms[keep]

        # keep top-K faster NMS
        dets = dets[:self.conf.test.keep_top_k, :]
        landms = landms[:self.conf.test.keep_top_k, :]
        
        assert len(dets)==len(landms), "the number of det and landm in the image must be equal."
        
        return dets, landms

    def process(self):
        for input_tensor, path in self.test_loader:
            self.input_tensor = input_tensor.to(self.device)
            self.img_raw = input_tensor.cpu().numpy().squeeze(0).transpose((1, 2, 0))
            preds = self.net(self.input_tensor.to(self.device))

            dets, landms = self._post_process(preds)
            print('path: ', path)
            print('dets: ', dets)
            print('landms: ', landms)

class RetinaResNetTest(RetinaMobileNetTest):
    def auditConfig(self):
        pass

    def initNetWithCode(self):
        torch.set_grad_enabled(False)
        self.net = RetinaFaceResNet()

if __name__ == "__main__":
    from config import config
    assert config.network == 'mobilenet' or config.network == 'resnet50', "config.network must be mobilenet or resnet50"
    
    if config.network == 'mobilenet':
        retina_test = RetinaMobileNetTest(config)
    else:
        retina_test = RetinaResNetTest(config)
    input_tensor = torch.rand(1,3,640,640)
    retina_test(input_tensor)
    
