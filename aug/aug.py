import random
import numpy as np
from deepvac.syszux_executor import RetinaAugExecutor
class RetinaAug(object):

    def __init__(self, deepvac_config):
        self.conf = deepvac_config.aug
        assert self.conf.img_dim, "please set config.aug.img_dim in config"
        assert self.conf.rgb_means, "please set config.aug.rgb_means in config"
        self.aug = RetinaAugExecutor(self.conf)

    def __call__(self, image, targets):
        assert targets.shape[0] > 0, "this image does not have gt"

        boxes = targets[:, :4].copy()
        labels = targets[:, -1].copy()
        landms = targets[:, 4:-1].copy()

        image_t, label  = self.aug([image, [boxes, landms, labels]])
        boxes_t, landms_t, labels_t = label

        labels_t = np.expand_dims(labels_t, 1)
        targets_t = np.hstack((boxes_t, landms_t, labels_t))

        return image_t, targets_t
