# RetinaFace
DeepVAC-compliant RetinaFace implementation

# 简介
本项目实现了符合DeepVAC规范的RetinaFace 。

### 项目依赖

- deepvac >= 0.5.7
- pytorch >= 1.8.0
- torchvision >= 0.7.0
- opencv-python
- numpy

# 如何运行本项目

## 1. 阅读[DeepVAC规范](https://github.com/DeepVAC/deepvac)
可以粗略阅读，建立起第一印象。

## 2. 准备运行环境
使用Deepvac规范指定[Docker镜像](https://github.com/DeepVAC/deepvac#2-%E7%8E%AF%E5%A2%83%E5%87%86%E5%A4%87)。

## 3. 准备数据集

- 获取WIDER Face数据集      
[WIDER Face Training Images](https://share.weiyun.com/5WjCBWV)
[WIDER Face Testing Images](https://share.weiyun.com/5vSUomP)
[Face annotations](http://mmlab.ie.cuhk.edu.hk/projects/WIDERFace/support/example/Submission_example.zip)

- 若想进一步了解WIDER Face数据集，可参考官网上的信息。    
[WIDER Face官网](http://shuoyang1213.me/WIDERFACE)

- 解压WIDER Face数据集

- 数据集配置
在config.py文件中作如下配置：     
```python
# line 26
config.datasets.RetinaTrainDataset.fileline_path = <train-image-dir>
# line 27
config.datasets.RetinaTrainDataset.sample_path_prefix = <train-list-path>
# line 76
config.sample_path = <test/val-image-dir>
```  

- 如果是自己的数据集，那么必须要跟widerface的标注格式一致

## 4. 训练相关配置
- 指定预训练模型路径(config.core.RetinaTrain.model_path)      
- 指定Backbone网络结构, 支持ResNet50, MobileNetV3, RegNet, RepVGG(config.core.RetinaTrain.net)
- 指定loss函数(config.core.RetinaTrain.criterion)
- 指定训练分类数量(config.core.RetinaTrain.class_num)    
- 指定优化器optimizer(config.core.RetinaTrain.optimizer)
- 指定学习率策略scheduler(config.core.RetinaTrain.scheduler)   

```python
config.core.RetinaTrain.model_path = ''
config.core.RetinaTrain.class_num = 2
config.core.RetinaTrain.shuffle = True
config.core.RetinaTrain.batch_size = 24
config.core.RetinaTrain.net = RetinaFaceMobileNet()
config.core.RetinaTrain.criterion = MultiBoxLoss(config.train.cls_num, 0.35, True, 0, True, 7, 0.35, False, config.train.device)
config.core.RetinaTrain.optimizer = torch.optim.SGD(
        config.core.RetinaTrain.net.parameters(),
        lr=1e-3,
        momentum=0.9,
        weight_decay=5e-4,
        nesterov=False
    )
config.core.RetinaTrain.scheduler = optim.lr_scheduler.MultiStepLR(config.core.RetinaTrain.optimizer, [100, 150, 190, 220], 0.1)

```
## 5. 训练

### 5.1 单卡训练
执行命令：

```bash
python3 train.py
```

### 5.2 分布式训练

在config.py中修改如下配置：
```python
#dist_url，单机多卡无需改动，多机训练一定要修改
config.core.RetinaTrain.dist_url = "tcp://localhost:27030"

#rank的数量，一定要修改
config.core.RetinaTrain.world_size = 2
```
然后执行命令：

```bash
python train.py --rank 0 --gpu 0
python train.py --rank 1 --gpu 1
```


## 6. 测试

- 测试相关配置

```python
# config.core.RetinaTest is config used for post_process and retina_test.
config.core.RetinaTest.model_path = "<pretrained-model-path>"
config.core.RetinaTest.confidence_threshold = 0.02
config.core.RetinaTest.nms_threshold = 0.4
config.core.RetinaTest.top_k = 5000
config.core.RetinaTest.keep_top_k = 1

# config.core.FaceTest is config used for face end-to-end test.
# align type
config.core.FaceTest.align_type = ['align', 'no_align', 'warp_crop']
# db/ds path and prefix(name)
config.core.FaceTest.test_dirs = ['']
config.core.FaceTest.test_prefix = ['']
config.core.FaceTest.db_dirs = ['']
config.core.FaceTest.db_prefix = ['']

# config.core.FaceRecTest is config used in face recognition module.
config.core.FaceRecTest.jit_model_path = "<face-recognition-trained-model-path>"

```

- 加载模型(*.pth)

```python
config.core.RetinaTest.model_path = <trained-model-path>
```

- 运行测试脚本：

```bash
python3 test.py
```
## 7. 使用trace模型/script模型
如果训练过程中开启config.cast.TraceCast（或者config.cast.ScriptCast)开关，可以在测试过程中转化torchscript模型     

- 转换torchscript模型(*.pt)     

```python
# trace
config.cast.TraceCast = AttrDict()
config.cast.TraceCast.model_dir = "./trace.pt"

# script
config.cast.ScriptCast = AttrDict()
config.cast.ScriptCast.model_dir = "./script.pt"
```

按照步骤6完成测试，torchscript模型将保存至model_dir指定文件位置      

- 加载torchscript模型

```python
config.core.RetinaTest.jit_model_path = <torchscript-model-path>
```

## 8. 使用静态量化模型
如果训练过程中未开启config.cast.TraceCast开关，可以在测试过程中转化静态量化模型     
- 转换静态模型(*.sq)     

```python
# trace
config.cast.TraceCast.static_quantize_dir = "./trace.sq"

# script
config.cast.ScriptCast.static_quantize_dir = "./script.sq"
```
按照步骤6完成测试，静态量化模型将保存至config.static_quantize_dir指定文件位置      

- 加载静态量化模型

```python
config.core.RetinaTest.jit_model_path = <static-quantize-model-path>
```
- 动态量化模型对应的配置参数为config.cast.TraceCast.dynamic_quantize_dir(或者config.cast.ScriptCast.dynamic_quantize_dir)

## 9. 更多功能
如果要在本项目中开启如下功能：
- 预训练模型加载
- checkpoint加载
- 使用tensorboard
- 启用TorchScript
- 转换ONNX
- 转换NCNN
- 转换CoreML
- 开启量化
- 开启自动混合精度训练

请参考[DeepVAC](https://github.com/DeepVAC/deepvac)

