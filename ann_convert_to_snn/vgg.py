import torch.nn as nn
import torch
 
class VGG(nn.Module):
    def __init__(self, features, num_classes=4, init_weights=False):
        super(VGG, self).__init__()
        self.features = features			# 卷积层提取特征
        self.classifier = nn.Sequential(	# 全连接层进行分类
            nn.Dropout(p=0.1),
            nn.Linear(512*8*8, 2048),
            nn.ReLU(True),
            nn.Dropout(p=0.1),
            nn.Linear(2048, 2048),
            nn.ReLU(True),
            nn.Linear(2048, num_classes)
        )
        if init_weights:
            self._initialize_weights()  #初始化权重
 
    def forward(self, x):
        # N x 3 x 256 x 256
        x = self.features(x)
        # N x 512 x 8 x 8
        x = torch.flatten(x, start_dim=1)
        # N x 512*8*8
        x = self.classifier(x)
        return x
 
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                # nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


cfgs = {
    'vgg11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],											# 模型A
    'vgg13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],									# 模型B
    'vgg16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],					# 模型D
    'vgg19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'], 	# 模型E
}
 
# 卷积层提取特征
def make_features(cfg: list): # 传入的是具体某个模型的参数列表
    layers = []
    in_channels = 3		# 输入的原始图像(rgb三通道)
    for v in cfg:
        # 如果是最大池化层，就进行池化
        if v == "M":
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        # 不然就是卷积层
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            layers += [conv2d, nn.ReLU(True)]
            in_channels = v
    return nn.Sequential(*layers)  # 单星号(*)将参数以元组(tuple)的形式导入
 
 
def vgg(model_name="vgg11", **kwargs):  # 双星号(**)将参数以字典的形式导入
    try:
        cfg = cfgs[model_name]
    except:
        print("Warning: model number {} not in cfgs dict!".format(model_name))
        exit(-1)
    model = VGG(make_features(cfg), **kwargs) #**kwargs是你传入的字典数据
    return model

