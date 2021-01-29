import torch 
from deepvac.syszux_config import *
from torchvision import transforms

# general
config.network = 'mobilenet' #'resnet50' or 'mobilenet'

config.disable_git = True
config.cls_num = 2
config.epoch_num = 250 if config.network=='mobilenet' else 100
config.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
config.output_dir = 'output'
config.save_num = 3
config.log_every = 100
config.num_workers = 4

config.momentum = 0.9
config.weight_decay = 5e-4
config.lr = 1e-3
config.gamma = 0.1

# train
config.train.shuffle = True
config.train.batch_size = 12 if config.network=='mobilenet' else 6
config.train.fileline_data_path_prefix = '/ your train image dir /'
config.train.fileline_path = '/ your train.txt path /'

# DDP
config.dist_url = 'tcp://localhost:27030'
config.world_size = 2

# aug
config.aug = AttrDict()
config.aug.img_dim = 640 if config.network=='mobilenet' else 840
config.aug.rgb_means = (104, 117, 123)

# val


# test
config.test.disable_git = True
config.test.model_path = '/ model path /'
config.test.confidence_threshold = 0.02
config.test.nms_threshold = 0.4
config.test.top_k = 5000
config.test.keep_top_k = 1
config.test.max_edge = 2000
config.test.rgb_means = (104, 117, 123)
config.test.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
