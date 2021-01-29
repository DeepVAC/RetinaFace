# RetinaFace
DeepVAC-compliant RetinaFace implementation

### 项目依赖

deepvac, pytorch, opencv-python, numpy

### 配置文件

**选择网络结构**

修改config.py文件，指定模型结构，支持ResNet50和MobileNetV3

```
config.network = 'resnet50' or 'mobilenet'
```

**准备数据**

修改config.py文件，指定训练集对应的标注txt文件和训练集目录（暂且没有加入验证集的逻辑）
指定网络结构，支持ResNet50以及MobileNetV3


```
config.train.fileline_data_path_prefix = '/ your train image dir /'
config.train.fileline_path = '/ your train.txt path /'
```

### 训练


```
python3 train.py
```

### 测试

**指定测试模型路径**

修改config.py指定模型路径


```
config.test.model_path = 'model path'
```

**运行测试脚本**

```
python3 test.py
```

### 项目参考

有关配置文件和ddp的更多细节，请参考：https://github.com/DeepVAC/deepvac
