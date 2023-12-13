import torch
from torch import nn
import matplotlib.pyplot as plt
from braincog.base.node.node import BaseNode ,LIFNode, IzhNode
from braincog.base.strategy.surrogate import *
from braincog.base.utils.visualization import spike_rate_vis, spike_rate_vis_1d

class TestNode(BaseNode):
    def __init__(self,threshold=1., tau=2, act_fun=QGateGrad, *args, **kwargs):
        super().__init__(threshold, *args, **kwargs)
        self.tau = tau
        if isinstance(act_fun, str):
            act_fun=eval(act_fun)
        self.act_fun = act_fun(alpha=2., requires_grad=False)
    def integral(self, inputs):
        self.mem = self.mem+((inputs-self.mem)/self.tau)*self.dt
    def calc_spike(self):
        self.spike= self.act_fun(self.mem-self.get_thres())
        self.mem=self.mem*(1-self.spike.detach())


from functools import partial
from timm.models import register_model
from braincog.base.node.node import *
from braincog.base.encoder.encoder import *
from braincog.model_zoo.base_module import BaseModule, BaseConvModule
from braincog.datasets import is_dvs_data
from braincog.datasets.datasets import get_dvsc10_data
from torch import optim
from opacus import PrivacyEngine

@register_model#timm里注册
class SNN5(BaseModule):
    def __init__(self,
                 num_classes=10,
                 step=8,
                 node_type=LIFNode,
                 encode_type='direct',
                 *args,
                 **kwargs):
        super().__init__(step,encode_type,*args, **kwargs)
        self.num_classes=num_classes
        self.layer_by_layer=True
        self.node=node_type
        if issubclass(self.node, BaseNode):
            self.node = partial(self.node, **kwargs, step=step)

        self.dataset=kwargs['dataset']if'dataset' in kwargs else 'dvsc10'
        if not is_dvs_data(self.dataset):
            init_channel=3
        else:
            init_channel=2

        self.feature = nn.Sequential(
            BaseConvModule(16, 16, kernel_size=(3,3), padding=(1,1), node=self.node),
            BaseConvModule(16, 64, kernel_size=(5,5), padding=(2,2), node=self.node),
            nn.AvgPool2d(2),
            BaseConvModule(64, 128, kernel_size=(5,5), padding=(2,2), node=self.node),
            nn.AvgPool2d(2),
            BaseConvModule(128, 256, kernel_size=(3,3), padding=(1,1), node=self.node),
            nn.AvgPool2d(2),
            BaseConvModule(256, 512, kernel_size=(3,3), padding=(1,1), node=self.node),
            nn.AvgPool2d(2),
            )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512*3*3,self.num_classes),
        )

    def forward(self,inputs):
        inputs = self.encoder(inputs)
        #print(inputs.shape)
        self.reset()

        if self.layer_by_layer:
            inputs=inputs.view(32, 16, 48, 48)
            x = self.feature(inputs)
            x = self.fc(x)
            #print(x.shape)
            #x = rearrange(x, '(t b) c -> t b c', t=self.step).mean(0)
            return x
        else:
            outputs = []
            for t in range(self.step):
                x = inputs[t]
                x = self.feature(x)
                x = self.fc(x)
                outputs.append(x)

            return sum(outputs)/len(outputs)
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

def load_cifar10(batch_size, train=True, shuffle=True):
    # 定义数据预处理步骤
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # 归一化
    ])

    # 加载CIFAR-10数据集
    dataset = torchvision.datasets.CIFAR10(root='./data', train=train, 
                                           download=True, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return dataloader

# 设置参数
#batch_size = 64

# 加载训练和测试数据
#train_loader = load_cifar10(batch_size, train=True)
#test_loader = load_cifar10(batch_size, train=False, shuffle=False)





model=SNN5()
train_loader, test_loader, train_data, test_data=get_dvsc10_data(batch_size=32,step=8)
print('data has been loaded')
"""
it=iter(train_loader)
inputs, label= it.next()
print(input.shape)
"""



model.to('cuda')
optimizer = optim.AdamW(model.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[40], gamma=0.1, last_epoch=-1)
disable_noise = False
delta = 1e-5
privacy_engine=PrivacyEngine()
"""
it = iter(train_loader)
inputs, labels =it.next()
print(inputs.shape, labels.shape)

print(spike_rate_vis(inputs[0,:,0]))

model=SNN5(layer_by_layer=True, datasets='dvsc10').cuda()
print(model)
"""
def train(model, device, train_loader, optimizer, epoch, privacy_engine):
    for param in model.parameters():
        param.requires_grad = True
    criterion = nn.CrossEntropyLoss().to(device)
    for i in range(epoch):
        losses = []
        #loss.requires_grad_(True)
        model.train()
        correct = 0
        for _batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            output = output.argmax(dim=1)
            #print(output)
            #print(target)
            loss = criterion(output.float(), target.float())
            loss.requires_grad_(True)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
            pred = output
            correct += pred.eq(target.view_as(pred)).sum().item()
        print(f"Train Epoch: {i} \t"
            f"Loss: {np.mean(losses):.6f} ")
        print("Accuracy: {}/{} ({:.2f}%)\n".format(
            correct,
            len(train_loader.dataset),
            100.0 * correct / len(train_loader.dataset), ))
    """
    if not disable_noise:
        epsilon = privacy_engine.get_epsilon(delta=delta)
        print(
            f"Train Epoch: {epoch} \t"
            f"Loss: {np.mean(losses):.6f} "
        )
        print("Accuracy: {}/{} ({:.2f}%)\n".format(
            correct,
            len(train_loader.dataset),
            100.0 * correct / len(train_loader.dataset), ))
        print(
              f"(ε = {epsilon:.2f}, δ = {delta})"
              )
    else:
        print(f"Train Epoch: {epoch} \t Loss: {np.mean(losses):.6f}")
    """
    return 100.0 * correct / len(train_loader.dataset)

print('begin training')

train(model, device='cuda', train_loader=train_loader, optimizer=optimizer, epoch=100, privacy_engine=privacy_engine)


def test(model, device, test_loader):
    model.eval()
    criterion = nn.CrossEntropyLoss().to(device)
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()  # sum up batch loss
            pred = output.argmax(
                dim=1, keepdim=True
            )  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader)

    print(
        "\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n".format(
            test_loss,
            correct,
            len(test_loader.dataset),
            100.0 * correct / len(test_loader.dataset),
        )
    )
    return correct / len(test_loader.dataset)












        


























