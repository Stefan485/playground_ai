import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):

    def __init__(self, config):
        super().__init__(self)

        self.conv1 = nn.Conv2d(in_channels=config.in_channels, out_channels=config.out_channels,
                                kernel_size=config.kernel_size, stride=config.stride)
        self.pool = nn.MaxPool2d(kernel_size=config.kernel_size, stride=config.stride)
        self.conv2 = nn.Conv2d(in_channels=config.out_channels, out_channels=config.out_channels,
                                kernel_size=config.kernel_size, stride=config.stride)
