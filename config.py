import torch 
from deepvac.syszux_config import *
from torchvision import transforms

# global
config.disable_git = True
config.cls_num = 2
config.epoch_num = 250
config.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
config.output_dir = 'output'
config.save_num = 3
config.log_every = 100

config.num_workers = 4
config.momentum = 0.9
config.weight_decay = 5e-4
config.lr = 1e-3
config.gamma = 0.1
config.confidence_threshold = 0.02
config.nms_threshold = 0.4
config.top_k = 5000
config.keep_top_k = 750

config.cfg_mobilenet = {
    'name': 'mobilenet',
    'min_sizes': [[16, 32], [64, 128], [256, 512]],
    'steps': [8, 16, 32],
    'variance': [0.1, 0.2],
    'clip': False,
    'loc_weight': 2.0,
    'gpu_train': True,
    'batch_size': 12,
    'ngpu': 1,
    'epoch': 250,
    'decay1': 190,
    'decay2': 220,
    'image_size': (640, 640),
    'pretrain': False,
    'in_channels_list': [40, 80, 160],
    'return_layers': {'5': '1', '10': '2', '15': '3'},
    'in_channel': 32,
    'out_channel': 64
}

config.cfg_resnet = {
    'name': 'Resnet50',
    'min_sizes': [[16, 32], [64, 128], [256, 512]],
    'steps': [8, 16, 32],
    'variance': [0.1, 0.2],
    'clip': False,
    'loc_weight': 2.0,
    'gpu_train': True,
    'batch_size': 4,
    'ngpu': 1,
    'epoch': 100,
    'decay1': 70,
    'decay2': 90,
    'image_size': (840, 840),
    'pretrain': False,
    'return_layers': {'layer2': 1, 'layer3': 2, 'layer4': 3},
    'in_channel': 256,
    'out_channel': 256    
}

config.network = 'mobilenet'  #'resnet50' or 'mobilenet'
config.cfg = config.cfg_mobilenet if config.network=='mobilenet' else config.cfg_resnet

# train
config.train.shuffle = True
config.train.batch_size = 12 if config.network=='mobilenet' else 6
config.train.label_path = '/ your train list /'

# aug
config.aug = AttrDict()
config.aug.img_dim = 640 if config.network=='mobilenet' else 840
config.aug.rgb_means = (104, 117, 123)

# val


# test
config.test.disable_git = True
config.test.trained_model = '/ pretrained model path /'
config.test.confidence_threshold = 0.02
config.test.nms_threshold = 0.4
config.test.top_k = 5000
config.test.keep_top_k = 1
config.test.max_edge = 2000
config.test.rgb_means = (104, 117, 123)
config.test.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
config.test.network = 'mobilenet'  #'resnet50' or 'mobilenet'
config.test.cfg = config.cfg_mobilenet if config.test.network=='mobilenet' else config.cfg_resnet
