import math
import torch

import torch.optim as optim
from torchvision import transforms as trans

from deepvac import config, AttrDict
from deepvac.loss import MultiBoxLoss
from modules.model_retina import RetinaFaceMobileNet, RetinaFaceResNet, RetinaFaceRegNet, RetinaFaceRepVGG
from modules.model_is import MobileFaceNet, Resnet50IR
from synthesis.synthesis import RetinaTrainDataset, detection_collate, RetinaValDataset, RetinaTestDataset
from aug.aug import RetinaAug

## ------------------ common ------------------
config.train.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
config.train.output_dir = 'output'
config.train.log_every = 100
config.train.disable_git = False

# load script and quantize model path
config.train.jit_model_path = "<your-script-or-quantize-model-path>"

# # # training # # #
## -------------------- training ------------------
config.train.epoch_num = 250 # 100 for resnet, 250 for others
config.train.save_num = 3
config.train.cls_num = 2
config.train.shuffle = True
config.train.batch_size = 4
config.train.fileline_data_path_prefix = "<train-image-dir>"
config.train.fileline_path = "<train-list-path>"
config.train.image_size = (640, 640) # (840, 840) for resnet, (640, 640) for others

config.train.model_path = "<pretrained-model-path>"

## -------------------- tensorboard ------------------
#config.train.tensorboard_port = "6007"
#config.train.tensorboard_ip = None


## -------------------- script and quantize ------------------
#config.train.trace_model_dir = "./trace.pt"
#config.train.static_quantize_dir = "./script.sq"
#config.train.dynamic_quantize_dir = "./quantize.sq"

## -------------------- aug ------------------
config.aug.facial_img_dim = 640 # 840 for resnet, 640 for others
config.aug.rgb_means = (104, 117, 123)

## -------------------- net and criterion ------------------
config.train.net = RetinaFaceResNet()# RetinaFaceMobileNet(), RetinaFaceRegNet(), RetinaFaceRepVGG(), RetinaFaceResNet()
config.train.criterion = MultiBoxLoss(config.train.cls_num, 0.35, True, 0, True, 7, 0.35, False, config.train.device)

## -------------------- optimizer and scheduler ------------------
config.train.optimizer = optim.SGD(
        config.train.net.parameters(),
        lr=1e-3,
        momentum=0.9,
        weight_decay=5e-4,
        nesterov=False
    )
#config.train.scheduler = optim.lr_scheduler.MultiStepLR(config.train.optimizer, [50, 70, 90], 0.1) # resnet
config.train.scheduler = optim.lr_scheduler.MultiStepLR(config.train.optimizer, [100, 150, 190, 220], 0.1) # others

## -------------------- loader ------------------
config.train.input_dir = "<test-image-dir>"
config.train.num_workers = 4
config.train.collate_fn = detection_collate
config.train.train_dataset = RetinaTrainDataset(config.train, augument=RetinaAug(config.aug))
config.train.train_loader = torch.utils.data.DataLoader(
    config.train.train_dataset,
    batch_size=config.train.batch_size,
    num_workers=config.train.num_workers,
    shuffle=config.train.shuffle,
    collate_fn=config.train.collate_fn
)

config.train.val_dataset = RetinaValDataset(config.train)
config.train.val_loader = torch.utils.data.DataLoader(config.train.val_dataset, batch_size=1, pin_memory=False)

config.train.test_dataset = RetinaTestDataset(config.train)
config.train.test_loader = torch.utils.data.DataLoader(config.train.test_dataset, batch_size=1, pin_memory=False)


## ------------------ ddp --------------------
# config.dist_url = 'tcp://localhost:27030'
# config.world_size = 2

## ------------------ test --------------------
config.train.post_process = AttrDict()
config.train.post_process.confidence_threshold = 0.02
config.train.post_process.nms_threshold = 0.4
config.train.post_process.top_k = 5000
config.train.post_process.keep_top_k = 1
config.train.post_process.max_edge = 2000
config.train.post_process.rgb_means = (104, 117, 123)

config.train.post_process.align_type = ['align', 'no_align', 'warp_crop']
config.train.post_process.test_dirs = []
config.train.post_process.test_prefix = []
config.train.post_process.db_dirs = []
config.train.post_process.db_prefix = []

## ------------------ face --------------------
config.train.accept = AttrDict()
config.train.accept.disable_git = True
config.train.accept.device = 'cuda'
config.train.accept.test_loader = ''
config.train.accept.embedding_size = 512
config.train.accept.net = MobileFaceNet(config.train.accept.embedding_size)
config.train.accept.threshold = 0.3
config.train.accept.transform = trans.Compose([
    trans.ToTensor(),
    trans.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])
config.train.accept.jit_model_path = "<face-trained-model-path>"
