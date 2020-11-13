# dnn



### Ghost Bottleneck

模型来自

- ShuffleNet: https://github.com/jaxony/ShuffleNet
- GhostNet: https://github.com/huawei-noah/ghostnet



##### version 0.1

时间: 2020.11.13 10:34

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

##### usage

```python
# zhq: define a ghost GhostBottleneck
# [kernel_size, hidden_channel, output_channel, se_ratio, stride, groups]
# hidden_channel, output_channel can be divided by 8
layers = []
blockInfo1 = [3, 16, 16, 0, 1, 4]
layers.append(blockInfo1)
blockInfo2 = [3, 48, 24, 0, 2, 4]
layers.append(blockInfo2)
layers = nn.Sequential(*layers)
```

