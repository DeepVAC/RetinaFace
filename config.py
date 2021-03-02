import torch 
from deepvac import config, AttrDict
from torchvision import transforms
import math
from torchvision import transforms as trans
# # # config # # #
config.disable_git = True

# model
config.model_path = "<trained-model-path>"
config.cls_num = 2
config.network = 'resnet50' #'resnet50' or 'mobilenet'
config.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# load script and quantize model path
# config.jit_model_path = "<pretrained-model-path>"

# # # optimizer # # #
config.momentum = 0.9
config.weight_decay = 5e-4
config.gamma = 0.1
config.lr = 1e-3
config.lr_step = [30, 60, 90]
config.lr_factor = 0.1

# # # training # # #
config.epoch_num = 250 if config.network=='mobilenet' else 100
config.save_num = 3
config.log_every = 100
config.num_workers = 4

# # # output # # #
config.output_dir = 'output'
config.trace_model_dir = "./output/trace.pt"
#config.static_quantize_dir = "./output/script.sq"

# train
config.train.shuffle = True
config.train.batch_size = 12 if config.network=='mobilenet' else 6
config.train.fileline_data_path_prefix = "<train-image-dir>"
config.train.fileline_path = "<train-list-path>"
config.train.image_size = (640, 640) if config.network=='mobilenet' else (840, 840)

# val
config.val.input_dir = "<val-image-dir>"

# DDP
# config.dist_url = 'tcp://localhost:27030'
# config.world_size = 2

# aug
config.aug = AttrDict()
config.aug.img_dim = 640 if config.network=='mobilenet' else 840
config.aug.rgb_means = (104, 117, 123)

# test
config.test.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
config.test.disable_git = True
config.test.input_dir = "<test-image-dir>"
config.test.confidence_threshold = 0.02
config.test.nms_threshold = 0.4
config.test.top_k = 5000
config.test.keep_top_k = 1
config.test.max_edge = 2000
config.test.rgb_means = (104, 117, 123)

config.test.align_type = ['align', 'no_align', 'warp_crop']
config.test.test_dirs = []
config.test.test_prefix = []
config.test.db_dirs = []
config.test.db_prefix = []

# face
config.face = AttrDict()
config.face.disable_git = True
config.face.device = 'cuda'
config.face.disable_git = True
config.face.embedding_size = 512
config.face.threshold = 0.3
config.face.transform = trans.Compose([
    trans.ToTensor(),
    trans.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])
config.face.jit_model_path = "<face-trained-model-path>"
