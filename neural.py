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

x=torch.rand(1,10,10)
lif=TestNode(threshold=0.3,tau=2.)#阈值是0.3，每一个时刻输入都呈现2倍衰减
lif.n_reset()#所有神经元使用前都需要重置
spike=lif(x)
spike_rate_vis(x)

print(spike)
spike_rate_vis(spike)
"""
"""
lif=IzhNode(0.5,tau=2)#依直科维奇生物神经元模型，需要较大电流
x=torch.rand(100)
spike=[]

lif.n_reset()
for t in range(1000):
    spike.append(lif(50*x))

spike = torch.stack(spike)
spike_rate_vis_1d(spike)


lif = TestNode(threshold=.5)
lif.n_reset()
lif.requires_fp=True

mem=[]
spike=[]
for t in range(100):
    x=torch.rand(1)
    spike.append(lif(x))
    mem.append(lif.mem)

mem = torch.stack(mem)
spike= torch.stack(spike)

outputs = torch.max(mem,spike).detach().cpu().numpy()

plt.plot(outputs)
plt.show()











