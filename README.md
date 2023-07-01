# transformer-CIFAR100
在Convolution-enhanced image Transformer（CeiT）上进行CIFAR100的训练，baseline 为 ResNet18

### 文件下载
包含 CeiT 和 ResNet 两个模型的checkpoint和log文件

https://pan.baidu.com/s/16wdkYhzc6bqb5jdGNrEYyQ?pwd=k8j8


### CeiT训练：
```
python train.py -c configs/default.yaml --name CeiT
```

### CeiT测试
```
python test.py -c configs/default.yaml --name CeiT -p checkpoint/CeiT_checkpoint.pyt
```

### Resnet训练
```
python train.py -net resnet18 -gpu
```

### ResNet测试：
```
python test.py -net resnet18 -weights checkpoint/resnet18.pth -gpu
```
