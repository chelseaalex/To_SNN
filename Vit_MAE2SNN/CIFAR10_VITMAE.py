import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from transformers import ViTMAEForPreTraining, ViTMAEConfig
from torch.utils.data import DataLoader
from tqdm import tqdm
from braincog.utils import setup_seed
from torch.cuda.amp import GradScaler, autocast
import torch.nn as nn

class CustomViTMAEForClassification(ViTMAEForPreTraining):
    def __init__(self, config, num_labels=10):
        super().__init__(config)
        self.classifier = nn.Linear(config.hidden_size, num_labels)
    
    def forward(self, pixel_values):
        outputs = self.vit(pixel_values=pixel_values)
        cls_output = outputs.last_hidden_state[:, 0]
        logits = self.classifier(cls_output)
        return logits

def main():
    setup_seed(42)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    batch_size = 128
    num_epochs = 3
    num_classes = 10  # CIFAR-10有10个类别

    # CIFAR-10 数据预处理
    normalize = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    transform_train = transforms.Compose([
        transforms.Resize((224, 224)),  # 假设图像大小为224x224
        transforms.RandomCrop(224, padding=8),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize
    ])
    transform_test = transforms.Compose([
        transforms.Resize((224, 224)),  # 假设图像大小为224x224
        transforms.ToTensor(), 
        normalize
    ])

    # 加载CIFAR-10数据集
    train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8)
    test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=8)

    # 初始化模型
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = ViTMAEConfig.from_pretrained('facebook/vit-mae-large', cache_dir="./cache",local_files_only=True)
    model = CustomViTMAEForClassification(config, num_labels=num_classes).to(device)
    
    train(model,num_epochs,device,train_loader,test_loader)
    # 测试模型
    model.load_state_dict(torch.load("./vit_mae_finetuned.pth", map_location=device))
    acc = evaluate_accuracy(test_loader, model, device)
    print(f"Test Accuracy: {acc}")

def evaluate_accuracy(data_iter, net, device=None, only_onebatch=False):
    if device is None and isinstance(net, torch.nn.Module):
        device = list(net.parameters())[0].device
    acc_sum, n = 0.0, 0
    with torch.no_grad():
        for X, y in tqdm(data_iter):
            net.eval()
            logits = net(X.to(device))
            acc_sum += (logits.argmax(dim=1) == y.to(device)).float().sum().cpu().item()
            net.train()
            n += y.shape[0]
            if only_onebatch: break
    return acc_sum / n

def train(model,num_epochs,device,train_loader,test_loader):
    # 微调模型
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    scaler = GradScaler()

    model.train()
    best = 0
    for epoch in tqdm(range(num_epochs)): 
        for images, labels in tqdm(train_loader):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            
            with autocast():  # 开始自动转换为FP16
                logits = model(pixel_values=images)
                loss = criterion(logits, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
        test_acc = evaluate_accuracy(test_loader, model)
        print(f'Epoch {epoch+1}, Loss: {loss.item()}, Train acc {test_acc}, Best acc {best}')
        
        if test_acc > best:
            best = test_acc
            torch.save(model.state_dict(), 'vit_mae_finetuned.pth')

if __name__ == "__main__":
    main()
