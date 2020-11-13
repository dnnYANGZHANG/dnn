# dnn



## Ghost Bottleneck

模型来自

- ShuffleNet: https://github.com/jaxony/ShuffleNet
- GhostNet: https://github.com/huawei-noah/ghostnet



### Version 0.1

时间: 2020.11.13 10:34

Github commit: ghostV0.1

原结构: 

```
Inp ---> GM1 --> GM2 --------------+--> Out
     |                             |
     |shortcut                     |
     |--> BN --> conv1x1 --> BN ---|
```
新结构:
```
Inp ---> CS --> GC --> GM1 --> GM2 ----------+--> Out
     |                                       |
     |shortcut                               |
     |-----> GC --> BN --> conv1x1 --> BN ---|
     
Inp: input, CS: channel shuffle, GC: group conv, GM: GhostModule
```
注: 省略了`Se`层

#### Usage
```python
# zhq: define a ghost GhostBottleneck
# [kernel_size, hidden_channel, output_channel, se_ratio, stride, groups]
# hidden_channel, output_channel can be divided by 8
layers = []
blockInfo1 = [3, 16, 16, 0, 1, 4]
layers.append(GhostBottleneck(blockInfo1))
blockInfo2 = [3, 48, 24, 0, 2, 4]
layers.append(GhostBottleneck(blockInfo2))
layers = nn.Sequential(*layers)
```

### Version 0.2
时间: 2020.11.13 14:56

改动: 
 - 原先的结构默认几乎所有的padding=1, 通过stride=2做downsampling. 

   ```
   GhostBottleneck(....)
     |-- GhostModule: 
              |-- Conv2d: padding=1, 3x3 
              |-- ....
              |-- Conv2d: padding=1, 1x1, group=all
              |-- ....
     |-- GhostModule: padding=1
              |-- Conv2d: padding=1, 3x3
              |-- ....
              |-- Conv2d: padding=1, 1x1
              |-- ....
     |-- Shortcut: 
              |-- Conv2d: padding=1, 3x3
              |-- ....
              |-- Conv2d: padding=0, 1x1
   ```

   现在改结构让他可以接受padding=0或1

   ```
   GhostBottleneck(...., padding=padding, group=group)
     |-- Conv2d: padding=1, 3x3, group=group
     |-- GhostModule: padding=padding
              |-- Conv2d: padding=padding, 3x3, 
              |-- ....
              |-- Conv2d: padding=1, 3x3, group=all
              |-- ....
     |-- GhostModule: padding=1
              |-- Conv2d: padding=1, 3x3, 
              |-- ....
              |-- Conv2d: padding=1, 3x3, group=all
              |-- ....
     |-- Shortcut: 
              |-- Conv2d: padding=padding, 3x3, group=group
              |-- ....
              |-- Conv2d: padding=0, 1x1
   ```

 - 创建可用于`cifar-10`训练的ghostNet, 位于`./ghostNet_cifar10.py`

#### Usage

- 新结构`GhostBottleneck`的使用

  ```python
  # zhq: define a ghost GhostBottleneck
  # [kernel_size, hidden_channel, output_channel, se_ratio, stride, groups, padding]
  # hidden_channel, output_channel can be divided by 8
  # 新增最后一个参数padding, 补0为0, 要补1为1
  layers = []
  blockInfo1 = [3, 16, 16, 0, 1, 4, 0]
  layers.append(GhostBottleneck(blockInfo1))
  blockInfo2 = [3, 48, 24, 0, 2, 4, 1]
  layers.append(GhostBottleneck(blockInfo2))
  layers = nn.Sequential(*layers)
  ```
- `ghostNet`在`cifar10`中的使用
  
  - `python ghostNet_cifar10.py`即可打印结构
  - `sh run.sh`开始训练
  - 目前跑了epoch=19, batch_size=16, learning_rate=1e-3, **testing acc=0.777**
