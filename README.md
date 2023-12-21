#目前安弈，杜保汛，董一航，吴圣栋，商汇川均已实现vgg的转换以及调优,

#安弈：实验了在vit，densenet，efficientnet，resnet上的ann转换成snn并且比较效果,
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
batch-size均为128，初始学习率为0.01,具体代码在ann_convert_to_snn里，详情可以见该文件夹里的readme
   结论也写在了那个readme里


   
#商汇川：实现了ann转snn后在coco的目标检测任务

#董一航：训练了VIT_16架构并进行了转化与效果比较，但是个人训练的VIT_16的ACC过低，于是下载了对应的预训练权重，转化为了VIT_SNN并与原VIT进行了比较；训练Unet图像生成模型，并进行convert转换为SNN；初步训练resNet34，并进行convert转换为SNN

#吴圣栋：实现了resnet上的ann转换成snn，使用参数为resnet18，block设置为[2，2，2，2]

#杜保汛：使用MAE训练cifar10后转snn进行测试,使用MAE训练后的结果为94.06，转成SNN后测试结果为96.37，验证了SNN优良的性能。


