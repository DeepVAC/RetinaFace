import random
import numpy as np
#from deepvac.syszux_aug import AugBase, RetinaCropAug, RetinaDistortAug, RetinaMirrorAug, RetinaPad2SquareAug, RetinaResizeSubtractMeanAug
from deepvac.syszux_aug import AugBase
from deepvac.syszux_executor import RetinaAugExecutor
class RetinaAug(AugBase):

    def __init__(self, deepvac_config):
        self.conf = deepvac_config.aug
        assert self.conf.img_dim, "please set config.aug.img_dim in config"
        assert self.conf.rgb_means, "please set config.aug.rgb_means in config"
        super(RetinaAug, self).__init__(self.conf)
        self.aug = RetinaAugExecutor(self.conf)

    def auditConfig(self):
        pass

    def __call__(self, image, targets):
        assert targets.shape[0] > 0, "this image does not have gt"

        boxes = targets[:, :4].copy()
        labels = targets[:, -1].copy()
        landms = targets[:, 4:-1].copy()

        #image_t, label = self._crop([image, [boxes, landm, labels]])
        #boxes_t, landm_t, labels_t = label
        
        #image_t = self._distort(image_t)
        image_t, label  = self.aug([image, [boxes, landms, labels]])
        boxes_t, landms_t, labels_t = label

        #image_t, label = self._pad_to_square([image_t, [boxes_t, landm_t, labels_t]])
        #boxes_t, landm_t, _ = label
        #if random.randrange(2):
        #    image_t, label = self._mirror([image_t, [boxes_t, landm_t, labels_t]])
        #    boxes_t, landm_t, _ = label
        #height, width, _ = image_t.shape
        #image_t, label = self._resize_subtract_mean([image_t, [boxes_t, landm_t, labels_t]])
        #boxes_t, landm_t, _ = label
        #boxes_t[:, 0::2] /= width
        #boxes_t[:, 1::2] /= height

        #landm_t[:, 0::2] /= width
        #landm_t[:, 1::2] /= height

        labels_t = np.expand_dims(labels_t, 1)
        targets_t = np.hstack((boxes_t, landms_t, labels_t))

        return image_t, targets_t
