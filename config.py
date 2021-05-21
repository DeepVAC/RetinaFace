import math
import torch
import torch.optim as optim
from torchvision import transforms as trans

from deepvac import config, AttrDict, fork
from deepvac.aug import RetinaAugComposer
from deepvac.loss import MultiBoxLoss

from modules.model_retina import RetinaFaceMobileNet, RetinaFaceResNet, RetinaFaceRegNet, RetinaFaceRepVGG
from modules.model_is import MobileFaceNet, Resnet50IR
from data.dataloader import RetinaTrainDataset, detection_collate, RetinaValDataset, RetinaTestDataset

## -------------------- aug ------------------
config.aug.ResizeSubtractMeanFacialAug = AttrDict()
config.aug.ResizeSubtractMeanFacialAug.img_dim = 640 # 840 for resnet, 640 for others
config.aug.ResizeSubtractMeanFacialAug.rgb_means = (104, 117, 123)

config.aug.CropFacialWithBoxesAndLmksAug = AttrDict()
config.aug.CropFacialWithBoxesAndLmksAug.img_dim = 640 # 840 for resnet, 640 for others

## -------------------- datasets ------------------
config.datasets.RetinaTrainDataset = AttrDict()
config.datasets.RetinaTrainDataset.composer = RetinaAugComposer(config)
config.datasets.RetinaTrainDataset.fileline_path = "<your train list>"
config.datasets.RetinaTrainDataset.sample_path_prefix = "<your train image path prefix>"

config.datasets.RetinaValDataset = AttrDict()
config.datasets.RetinaValDataset.max_edge = 2000
config.datasets.RetinaValDataset.rgb_means = (104, 117, 123)

config.datasets.RetinaTestDataset = AttrDict()
config.datasets.RetinaTestDataset.max_edge = 2000
config.datasets.RetinaTestDataset.rgb_means = (104, 117, 123)

## -------------------- script and quantize ------------------
#config.cast.TraceCast = AttrDict()
#config.cast.TraceCast.model_dir = "./trace.pt"
# config.cast.TraceCast.static_quantize_dir = "./script.sq"
#config.cast.TraceCast.dynamic_quantize_dir = "./quantize.sq"

## ------------------ common ------------------
config.core.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
config.core.output_dir = 'output'
config.core.log_every = 10
config.core.disable_git = False

# load script and quantize model path
# config.core.jit_model_path = "<your-script-or-quantize-model-path>"
## -------------------- training ------------------
config.core.epoch_num = 250 # 100 for resnet, 250 for others
config.core.save_num = 3
config.core.cls_num = 2
config.core.shuffle = True
config.core.batch_size = 4
config.core.image_size = (640, 640) # (840, 840) for resnet, (640, 640) for others
# config.core.model_path = "<pretrained-model-path>"

## -------------------- tensorboard ------------------
#config.core.tensorboard_port = "6007"
#config.core.tensorboard_ip = None

## -------------------- net and criterion ------------------
config.core.net = RetinaFaceMobileNet()# RetinaFaceMobileNet(), RetinaFaceRegNet(), RetinaFaceRepVGG(), RetinaFaceResNet()
config.core.criterion = MultiBoxLoss(config, config.core.cls_num, 0.35, True, 0, True, 7, 0.35, False, config.core.device)

## -------------------- optimizer and scheduler ------------------
config.core.optimizer = optim.SGD(
        config.core.net.parameters(),
        lr=1e-3,
        momentum=0.9,
        weight_decay=5e-4,
        nesterov=False
    )
config.core.scheduler = optim.lr_scheduler.MultiStepLR(config.core.optimizer, [100, 150, 190, 220], 0.1) # others

config.core.num_workers = 4
config.core.collate_fn = detection_collate
config.core.train_dataset = RetinaTrainDataset(config)
config.core.train_loader = torch.utils.data.DataLoader(
    config.core.train_dataset,
    batch_size=config.core.batch_size,
    num_workers=config.core.num_workers,
    shuffle=config.core.shuffle,
    collate_fn=config.core.collate_fn
)

config.core.sample_path = '<your test/val image dir>'
config.core.val_dataset = RetinaValDataset(config, config.core.sample_path)
config.core.val_loader = torch.utils.data.DataLoader(config.core.val_dataset, batch_size=1, pin_memory=False)

config.core.test_dataset = RetinaTestDataset(config, config.core.sample_path)
config.core.test_loader = torch.utils.data.DataLoader(config.core.test_dataset, batch_size=1, pin_memory=False)

## ------------------ ddp --------------------
# config.core.dist_url = 'tcp://localhost:27030'
# config.core.world_size = 2

## ------------------ for post_process --------------------
config.core.post_process = AttrDict()

config.core.post_process.confidence_threshold = 0.02
config.core.post_process.nms_threshold = 0.4
config.core.post_process.top_k = 5000
config.core.post_process.keep_top_k = 1

config.core.post_process.align_type = ['align', 'no_align', 'warp_crop']
config.core.post_process.test_dirs = ['./ipc/ds'] # ds image dir list
config.core.post_process.test_prefix = ['ipc'] # ds prefix list
config.core.post_process.db_dirs = ['./ipc/db'] # db image dir list
config.core.post_process.db_prefix = ['ipc'] # ds prefix list

## ------------------ for processAccept --------------------
rec_config = fork(config,['core'])
rec_config.core.is_forward_only = True
rec_config.core.test_loader = ''
rec_config.core.embedding_size = 512
rec_config.core.net = MobileFaceNet(rec_config.core.embedding_size)
rec_config.core.threshold = 0.3
rec_config.core.transform = trans.Compose([
    trans.ToTensor(),
    trans.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])
rec_config.core.jit_model_path = "<face-recognition-trained-model-path>"
