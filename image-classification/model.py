import torch
import torch.nn as nn

class Dataset(torch.utils.data.Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        x = self.data['image'][index]
        y = self.data['label'][index]
        return x, y

class Block(nn.Module):
    def __init__(self, in_channels, out_channels, stride, kernel_size):
        super(Block, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, stride=stride)
        self.bn = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = x + self.conv(x)
        x = x + self.bn(x)
        x = x + self.conv2(x)
        x = x + self.relu(x)
        x = x + self.bn2(x)
        return x
    

class Model(nn.Module):
    
    def __init__(self, in_channels, out_channels, kernel_size, stride, num_of_blocks):
        super(Model, self).__init__()
        self.blocks = nn.ModuleList([Block(in_channels, out_channels, stride, kernel_size) for _ in range(num_of_blocks)])
        #add end layer

    def forward(self, x):
        for block in self.blocks:
            x = x + block(x)

        return x