import random
import numpy as np
from deepvac.aug import RetinaAugComposer
class RetinaAug(object):

    def __init__(self, deepvac_aug_config):
        self.config = deepvac_aug_config
        assert self.config.facial_img_dim, "please set config.aug.facial_img_dim in config"
        assert self.config.rgb_means, "please set config.aug.rgb_means in config"
        self.aug = RetinaAugComposer(self.config)

    def auditConfig(self):
        pass

    def __call__(self, image, targets):
        assert targets.shape[0] > 0, "this image does not have gt"

        boxes = targets[:, :4]
        labels = targets[:, -1]
        landms = targets[:, 4:-1]

        image_t, label  = self.aug([image, [boxes, landms, labels]])
        boxes_t, landms_t, labels_t = label

        labels_t = np.expand_dims(labels_t, 1)
        targets_t = np.hstack((boxes_t, landms_t, labels_t))

        return image_t, targets_t
