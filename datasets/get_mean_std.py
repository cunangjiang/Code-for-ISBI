from torchvision.datasets import ImageFolder
import torch
from torchvision import transforms
from PIL import Image

# 自定义 transform，将灰度图像 ("L") 转换为 RGB
transform_to_rgb = transforms.Compose([
    transforms.Lambda(lambda img: img.convert("RGB")),  # 将灰度图像转换为RGB
    transforms.ToTensor()  # 转换为张量
])

def getStat(train_data):
    '''
    Compute mean and variance for training data
    :param train_data: 自定义类Dataset(或ImageFolder即可)
    :return: (mean, std)
    '''
    print('Compute mean and variance for training data.')
    print(len(train_data))
    train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=1, shuffle=False, num_workers=0,
        pin_memory=True)
    mean = torch.zeros(3)
    std = torch.zeros(3)
    for X, _ in train_loader:
        for d in range(3):
            mean[d] += X[:, d, :, :].mean()
            std[d] += X[:, d, :, :].std()
    mean.div_(len(train_data))
    std.div_(len(train_data))
    return list(mean.numpy()), list(std.numpy())

if __name__ == '__main__':
    # 读取 BraTS2020 数据集，并应用转换
    train_dataset = ImageFolder(root=r'datasets/BraTs2020_t1_t2_tiny/val', transform=transform_to_rgb)
    
    # 计算均值和标准差
    print(getStat(train_dataset))
