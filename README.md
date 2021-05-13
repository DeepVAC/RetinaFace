# RetinaFace
DeepVAC-compliant RetinaFace implementation

# 简介
本项目实现了符合DeepVAC规范的RetinaFace 。

### 项目依赖

- deepvac >= 0.2.6
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
config.train.fileline_data_path_prefix = <train-image-dir>
config.train.fileline_path = <train-list-path>
config.train.input_dir = <test-image-dir>
```
- 验证/测试集可自己设定，设置config.train.input_dir参数即可。   

- 如果是自己的数据集，那么必须要跟widerface的标注格式一致

## 4. 训练相关配置
- 指定预训练模型路径(config.train.model_path)      
- 指定Backbone网络结构, 支持ResNet50, MobileNetV3, RegNet, RepVGG(config.train.net)
- 指定loss函数(config.train.criterion)
- 指定训练分类数量(config.train.class_num)    
- 指定学习率策略相关参数(config.train.momentum, config.train.weight_decay, config.train.lr, config.train.gamma)
- dataloader相关配置(config.train.collate_fn, config.train.train_dataset, config.train.train_loader, config.train.val_dataset, config.train.val_loader, config.train.test_dataset, config.train.test_loader)     

```python
config.model_path = ''
config.class_num = 2
config.momentum = 0.9
config.weight_decay = 5e-4
config.lr = 1e-3
config.gamma = 0.1
config.train.shuffle = True
config.train.batch_size = 24

config.train.input_dir = "<test-image-dir>"
config.train.collate_fn = detection_collate
config.train.train_dataset = RetinaTrainDataset(config.train, augument=RetinaAug(config.aug))
config.train.train_loader = torch.utils.data.DataLoader(
    config.train.train_dataset,
    batch_size=config.train.batch_size,
    num_workers=config.train.num_workers,
    shuffle= False if is_ddp else config.train.shuffle,
    collate_fn=config.train.collate_fn
)

config.train.val_dataset = RetinaValDataset(config.train)
config.train.val_loader = torch.utils.data.DataLoader(config.train.val_dataset, batch_size=1, pin_memory=False)

config.train.test_dataset = RetinaTestDataset(config.train)
config.train.test_loader = torch.utils.data.DataLoader(config.train.test_dataset, batch_size=1, pin_memory=False)

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
config.train.dist_url = "tcp://localhost:27030"

#rank的数量，一定要修改
config.train.world_size = 2
```
然后执行命令：

```bash
python train.py --rank 0 --gpu 0
python train.py --rank 1 --gpu 1
```


## 6. 测试

- 测试相关配置

```python
config.train.input_dir = <test-image-dir>
config.train.confidence_threshold = 0.02
config.train.nms_threshold = 0.4
config.train.top_k = 5000
config.train.keep_top_k = 1
config.train.max_edge = 2000
config.train.rgb_means = (104, 117, 123)
```

- 加载模型(*.pth)

```python
config.train.model_path = <trained-model-path>
```

- 运行测试脚本：

```bash
python3 test.py
```
## 7. 使用trace模型
如果训练过程中开启config.train.trace_model_dir开关，可以在测试过程中转化torchscript模型     

- 转换torchscript模型(*.pt)     

```python
config.train.trace_model_dir = "output/trace.pt"
```

按照步骤6完成测试，torchscript模型将保存至config.torchscript_model_dir指定文件位置      

- 加载torchscript模型

```python
config.jit_model_path = <torchscript-model-path>
```

## 8. 使用静态量化模型
如果训练过程中未开启config.static_quantize_dir开关，可以在测试过程中转化静态量化模型     
- 转换静态模型(*.sq)     

```python
config.static_quantize_dir = "output/trace.sq"
```
按照步骤6完成测试，静态量化模型将保存至config.static_quantize_dir指定文件位置      

- 加载静态量化模型

```python
config.jit_model_path = <static-quantize-model-path>
```
- 动态量化模型对应的配置参数为config.dynamic_quantize_dir     

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

