import math
import torch
import torch.optim as optim
from torchvision import transforms as trans

from deepvac import AttrDict, new
from deepvac.aug import RetinaAugComposer
from deepvac.loss import MultiBoxLoss

from modules.model_retina import RetinaFaceMobileNet, RetinaFaceResNet, RetinaFaceRegNet, RetinaFaceRepVGG
from modules.model_is import MobileFaceNet, Resnet50IR
from data.dataloader import RetinaTrainDataset, detection_collate, RetinaValDataset, RetinaTestDataset

config = new('RetinaTrain')
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
# config.cast.TraceCast = AttrDict()
# config.cast.TraceCast.model_dir = "./trace.pt"
# config.cast.TraceCast.static_quantize_dir = "./script.sq"
# config.cast.TraceCast.dynamic_quantize_dir = "./quantize.sq"

## ------------------ common ------------------
config.core.RetinaTrain.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
config.core.RetinaTrain.output_dir = 'output'
config.core.RetinaTrain.log_every = 100
config.core.RetinaTrain.disable_git = False

# load script and quantize model path
# config.core.RetinaTrain.jit_model_path = "<your-script-or-quantize-model-path>"
## -------------------- training ------------------
config.core.RetinaTrain.epoch_num = 250 # 100 for resnet, 250 for others
config.core.RetinaTrain.save_num = 3
config.core.RetinaTrain.cls_num = 2
config.core.RetinaTrain.image_size = (640, 640) # (840, 840) for resnet, (640, 640) for others
# config.core.RetinaTrain.model_path = "<pretrained-model-path>"

## -------------------- tensorboard ------------------
# config.core.RetinaTrain.tensorboard_port = "6007"
# config.core.RetinaTrain.tensorboard_ip = None

## -------------------- net and criterion ------------------
config.core.RetinaTrain.net = RetinaFaceMobileNet()# RetinaFaceMobileNet(), RetinaFaceRegNet(), RetinaFaceRepVGG(), RetinaFaceResNet()
config.core.RetinaTrain.criterion = MultiBoxLoss(config, config.core.RetinaTrain.cls_num, 0.35, True, 0, True, 7, 0.35, False, config.core.RetinaTrain.device)

## -------------------- optimizer and scheduler ------------------
config.core.RetinaTrain.optimizer = optim.SGD(
        config.core.RetinaTrain.net.parameters(),
        lr=1e-3,
        momentum=0.9,
        weight_decay=5e-4,
        nesterov=False
    )
config.core.RetinaTrain.scheduler = optim.lr_scheduler.MultiStepLR(config.core.RetinaTrain.optimizer, [100, 150, 190, 220], 0.1) # others

config.sample_path = '<your test/val image dir>'

config.core.RetinaTrain.shuffle = True
config.core.RetinaTrain.batch_size = 16
config.core.RetinaTrain.num_workers = 4
config.core.RetinaTrain.collate_fn = detection_collate

config.core.RetinaTrain.train_dataset = RetinaTrainDataset(config)
config.core.RetinaTrain.train_loader = torch.utils.data.DataLoader(
    config.core.RetinaTrain.train_dataset,
    batch_size=config.core.RetinaTrain.batch_size,
    num_workers=config.core.RetinaTrain.num_workers,
    shuffle=config.core.RetinaTrain.shuffle,
    collate_fn=config.core.RetinaTrain.collate_fn
)

config.core.RetinaTrain.val_dataset = RetinaValDataset(config, config.sample_path)
config.core.RetinaTrain.val_loader = torch.utils.data.DataLoader(config.core.RetinaTrain.val_dataset, batch_size=1, pin_memory=False)

config.core.RetinaTrain.test_dataset = RetinaTestDataset(config, config.sample_path)
config.core.RetinaTrain.test_loader = torch.utils.data.DataLoader(config.core.RetinaTrain.test_dataset, batch_size=1, pin_memory=False)

## ------------------ ddp --------------------
config.core.RetinaTrain.dist_url = 'tcp://localhost:27030'
config.core.RetinaTrain.world_size = 2

## ------------------ for post_process and retina_test --------------------
config.core.RetinaTest = config.core.RetinaTrain.clone()
config.core.RetinaTest.model_path = "<pretrained-model-path>"
config.core.RetinaTest.confidence_threshold = 0.02
config.core.RetinaTest.nms_threshold = 0.4
config.core.RetinaTest.top_k = 5000
config.core.RetinaTest.keep_top_k = 1

## ------------------ for face end-to-end test --------------------
config.core.FaceTest = config.core.RetinaTrain.clone()
config.core.FaceTest.align_type = ['align', 'no_align', 'warp_crop']
config.core.FaceTest.test_dirs = ['./ipc/ds'] # ds image dir list
config.core.FaceTest.test_prefix = ['ipc'] # ds prefix list
config.core.FaceTest.db_dirs = ['./ipc/db'] # db image dir list
config.core.FaceTest.db_prefix = ['ipc'] # ds prefix list

## ------------------ for face recognition in processAccept --------------------
config.core.FaceRecTest = config.core.RetinaTrain.clone()
config.core.FaceRecTest.is_forward_only = True
config.core.FaceRecTest.test_loader = ''
config.embedding_size = 512
config.core.FaceRecTest.net = MobileFaceNet(config.embedding_size)
config.core.FaceRecTest.threshold = 0.3
config.core.FaceRecTest.transform = trans.Compose([
    trans.ToTensor(),
    trans.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])
config.core.FaceRecTest.jit_model_path = "<face-recognition-trained-model-path>"
