import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import matplotlib
matplotlib.use('agg')
import numpy as np
from tqdm import tqdm
from CIFAR10_VITMAE import CustomViTMAEForClassification, ViTMAEConfig  # Using ViT-MAE instead of VGG16
from braincog.utils import setup_seed
from braincog.base.conversion import Convertor
import argparse
from torch.utils.data import DataLoader


parser = argparse.ArgumentParser(description='Conversion')
parser.add_argument('--T', default=64, type=int, help='simulation time')
parser.add_argument('--p', default=0.99, type=float, help='percentile for data normalization. 0-1')
parser.add_argument('--gamma', default=5, type=int, help='burst spike and max spikes IF can emit')
parser.add_argument('--channelnorm', default=False, type=bool, help='use channel norm')
parser.add_argument('--lipool', default=True, type=bool, help='LIPooling')
parser.add_argument('--smode', default=True, type=bool, help='replace ReLU to IF')
parser.add_argument('--soft_mode', default=True, type=bool, help='soft reset or not')
parser.add_argument('--device', default='4', type=str, help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
parser.add_argument('--cuda', default=True, type=bool, help='use cuda.')
parser.add_argument('--model_name', default='vitmae', type=str, help='model name. vgg16 or resnet20 or vitmae')
parser.add_argument('--merge', default=True, type=bool, help='merge conv and bn')
parser.add_argument('--train_batch', default=1, type=int, help='batch size for get max')
parser.add_argument('--batch_num', default=1, type=int, help='number of train batch')
parser.add_argument('--spicalib', default=0, type=int, help='allowance for spicalib')
parser.add_argument('--batch_size', default=128, type=int, help='batch size for testing')
parser.add_argument('--seed', default=42, type=int, help='seed')
args = parser.parse_args()


def evaluate_snn(test_iter, snn, device=None, duration=50):
    accs = []
    snn.eval()

    for ind, (test_x, test_y) in tqdm(enumerate(test_iter)):
        test_x = test_x.to(device)
        test_y = test_y.to(device)
        n = test_y.shape[0]
        out = 0
        with torch.no_grad():
            snn.reset()
            acc = []
            # for t in tqdm(range(duration)):
            for t in range(duration):
                out += snn(test_x)
                result = torch.max(out, 1).indices
                result = result.to(device)
                acc_sum = (result == test_y).float().sum().item()
                acc.append(acc_sum / n)

        accs.append(np.array(acc))
    accs = np.array(accs).mean(axis=0)

    i, show_step = 1, []
    while 2 ** i <= duration:
        show_step.append(2 ** i - 1)
        i = i + 1

    for iii in show_step:
        print("timestep", str(iii).zfill(3) + ':', accs[iii])
    print("best acc: ", max(accs))


if __name__ == '__main__':
    print("Setting Arguments.. : ", args)
    print("----------------------------------------------------------")
    setup_seed(seed=args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    num_classes = 10
    
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    
    if args.model_name == 'vitmae':
        config = ViTMAEConfig.from_pretrained('facebook/vit-mae-large',cache_dir="./cache",local_files_only=True)
        net = CustomViTMAEForClassification(config, num_labels=num_classes)
        print("loading pth..")
        net.load_state_dict(torch.load("./vit_mae_finetuned.pth", map_location=device))

    # CIFAR-10 数据预处理
    normalize = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    transform_train = transforms.Compose([
        transforms.Resize((config.image_size, config.image_size)),
        transforms.RandomCrop(config.image_size, padding=8),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize
    ])
    transform_test = transforms.Compose([
        transforms.Resize((config.image_size, config.image_size)),
        transforms.ToTensor(), 
        normalize
    ])

    # 加载CIFAR-10数据集
    train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,num_workers=8)
    test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,num_workers=8)


    net.eval()
    net = net.to(device)

    converter = Convertor(dataloader=test_loader,
                          device=device,
                          p=args.p,
                          channelnorm=args.channelnorm,
                          lipool=args.lipool,
                          gamma=args.gamma,
                          soft_mode=args.soft_mode,
                          merge=args.merge,
                          batch_num=args.batch_num,
                          spicalib=args.spicalib
                          )
    snn = converter(net)
    print("----------------------------------------------------------")

    evaluate_snn(test_loader, snn, device, duration=args.T)