# transformer-CIFAR100
在Convolution-enhanced image Transformer（CeiT）上进行CIFAR100的训练，baseline 为 ResNet18

### Install requirement
```
pip install -r requirements.txt
```

### CeiT训练：
Train :
```
python train.py -c configs/default.yaml --name CeiT
```
### CeiT测试
Test :
```
python test.py -c configs/default.yaml --name CeiT -p checkpoint/CeiT_checkpoint.pyt
```

### Resnet训练
train:
```
python train.py -net resnet18 -gpu
```
### ResNet测试：
```
python test.py -net resnet18 -weights checkpoint/resnet18.pth -gpu
```

可下载训练好的模型进行测试
Github 代码地址：https://github.com/ivorytan/transformer-resnet-cifar100
模型下载地址：链接：https://pan.baidu.com/s/1kw0LgUn37Jvkwx_lGYM1Eg 提取码：549c
