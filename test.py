import torch
import torch.backends.cudnn as cudnn

import cv2
import numpy as np
import sys
sys.path.insert(0, '/gemfield/hostpv/wangyuhang/github/deepvac')

from deepvac.syszux_log import LOG
from deepvac.syszux_deepvac import Deepvac
from deepvac.syszux_post_process import py_cpu_nms, decode, decode_landm, PriorBox

from modules.model import RetinaFace

class RetinaTest(Deepvac):
    def __init__(self, retina_config):
        super(RetinaTest, self).__init__(retina_config)
    
    def initNetWithCode(self):
        torch.set_grad_enabled(False)
        self.net = RetinaFace(self.conf.cfg, phase='test')
        self.net = self._load_model(self.net, self.conf.trained_model, self.device)
        self.net.eval()
        print('Finished loading model!')
        cudnn.benchmark = True
        self.net.to(self.device)
    
    def _remove_prefix(self, state_dict, prefix):
        ''' Old style model is stored with all names of parameters sharing common prefix 'module.' '''
        print('remove prefix \'{}\''.format(prefix))
        f = lambda x: x.split(prefix, 1)[-1] if x.startswith(prefix) else x
        return {f(key): value for key, value in state_dict.items()}
    
    def _load_model(self, model, pretrained_path, device):
        print('Loading pretrained model from {}'.format(pretrained_path))
        if device == 'cpu':
            pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage)
        else:
            device = torch.cuda.current_device()
            pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage.cuda(device))
        if "state_dict" in pretrained_dict.keys():
            pretrained_dict = self._remove_prefix(pretrained_dict['state_dict'], 'module.')
        else:
            pretrained_dict = self._remove_prefix(pretrained_dict, 'module.')
        #check_keys(model, pretrained_dict)
        model.load_state_dict(pretrained_dict, strict=False)
        return model

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
        loc, conf, landms = preds
        
        priorbox = PriorBox(self.conf.cfg, image_size=(self.img_raw.shape[0], self.img_raw.shape[1]))
        priors = priorbox.forward()
        priors = priors.to(self.device)
        prior_data = priors.data
        resize = 1
        scale = torch.Tensor([self.input_tensor.shape[3], self.input_tensor.shape[2], self.input_tensor.shape[3], self.input_tensor.shape[2]])
        scale = scale.to(self.device)
        boxes = decode(loc.data.squeeze(0), prior_data, self.conf.cfg['variance'])
        boxes = boxes * scale / resize
        boxes = boxes.cpu().numpy()
        scores = conf.squeeze(0).data.cpu().numpy()[:, 1]
        landms = decode_landm(landms.data.squeeze(0), prior_data, self.conf.cfg['variance'])
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
        
        assert len(dets)==len(landms), "the number of det and landm in the image must be equal."
        
        return dets, landms

    def __call__(self, image):
        self._pre_process(image)
        preds = self.net(self.input_tensor)

        return self._post_process(preds)

if __name__ == "__main__":
    from config import config as deepvac_config

    img = cv2.imread('./sample.jpg')
    retina_test = RetinaTest(deepvac_config.test)
    dets, landms = retina_test(img)
    print('dets: ', dets)
    print('landms: ', landms)
