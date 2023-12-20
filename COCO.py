import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# 加载COCO数据集
train_dataset = datasets.CocoDetection(root='./data', annFile='./data/annotations/instances_train2017.json',
                                       transform=transforms.ToTensor())
test_dataset = datasets.CocoDetection(root='./data', annFile='./data/annotations/instances_val2017.json',
                                      transform=transforms.ToTensor())

num_epochs = 10  # 设置训练轮数
learning_rate = 0.001  # 设置学习率
batch_size = 128
# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(models.vgg16(pretrained=True).parameters(), lr=learning_rate)

# 训练模型
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
for epoch in range(num_epochs):
    for i, (images, targets) in enumerate(train_loader):
        optimizer.zero_grad()
        outputs = models.vgg16(images)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
    print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch + 1, num_epochs, loss.item()))

# 加载测试数据集
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

# 计算模型准确率
model = models.vgg16(pretrained=True)
model.eval()  # 设置模型为评估模式
correct = 0
total = 0
with torch.no_grad():  # 不计算梯度，减少内存使用
    for images, targets in test_loader:
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)  # 获取预测结果
        total += targets.size(0)  # 增加总样本数
        correct += (predicted == targets).sum().item()  # 计算正确样本数

accuracy = correct / total  # 计算准确率
print('Accuracy of the model on the test images: {} %'.format(accuracy * 100))

# 保存训练好的模型
torch.save(models.vgg16(pretrained=True).state_dict(), './COCOmodel.pth')