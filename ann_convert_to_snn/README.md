#安弈：实验了在vit，densenet，efficientnet，resnet上的ann转换成snn并且比较效果,
运行先运行CIFAR10_efficient.py形式的函数生成训练好的模型，然后运行convert_CIFAR10.py文件，修改model_name以测试不同网络

ann网络已提前训练好了，看结果之间运行就行
但是efficientnet，vgg，resnet的结果太大了没法上传至github，如果需要运行我可以私发给你

1. 对于efficientnet使用网络参数为
cfgs = [
        # t, c, n, s, fused
        [1,  16,  2, 1, 1],
        [4,  32,  4, 2, 1],
        [4,  64,  4, 2, 1],
        [4, 128,  6, 2, 0],
        [6, 256,  9, 1, 0],    ]
2. densenet参数设置为
                 growth_rate=12, block_config=(8, 12, 16), compression=0.5,
                 num_init_features=24, bn_size=4, drop_rate=0,
                 num_classes=10, small_inputs=False, efficient=False
3.resnet使用参数为
	resnet34，block设置为[3, 4, 6, 3]
所有实验的优化器均为torch.optim.SGD
损失函数均为CrossEntropyLoss
batch-size均为128，初始学习率为0.01

