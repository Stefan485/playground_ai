import torch
import torch.nn as nn


#In transformers implementation there is a choice for the activation function used, but here I am using ReLU (default in transformers)
class ResNetConvLayer(nn.Module):

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, stride: int = 1, activation: str = 'relu'):
        super().__init__()
        self.convolution = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride)
        self.normalization = nn.BatchNorm2d(out_channels)
        self.activation = nn.ReLU() if activation == 'relu' else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.convolution(x)
        x = x + self.normalization(x)
        x = x + self.activation(x)
        return x
    

class ResNetEmbeddings(nn.Module):

    def __init__(self, num_chanels: int, embedding_size: int, kernel_size: int = 7, stride: int = 2):
        super().__init__()

        self.embedder = ResNetConvLayer(num_chanels, embedding_size, kernel_size, stride)
        self.pooler = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.num_chanels = num_chanels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.embedder(x)
        x = self.pooler(x)
        return x
    

class ResNetShortcut(nn.Module):

    def __init__(self, in_channels: int, out_channels: int, stride: int = 2):
        super().__init__()
        self.convolution = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False)
        self.normalization = nn.BatchNorm2d(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.convolution(x)
        x = self.normalization(x)
        return x
    

class ResNetBasicLayer(nn.Module):

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1, activation: str = 'relu'):
        super().__init__()
        should_apply_shortcut = in_channels != out_channels or stride != 1

        self.shortcut = (ResNetShortcut(in_channels, out_channels, stride) if should_apply_shortcut else nn.Identity())

        self.layer = nn.Sequential(ResNetConvLayer(in_channels, out_channels, stride=stride),
                                    ResNetConvLayer(out_channels, out_channels, activation=activation))
        
        self.activation = nn.ReLU() if activation == 'relu' else nn.Identity()

    def forward(self, x):
        residual = x
        x = self.layer(x)
        residual = self.shortcut(residual)
        x += residual
        x = self.activation(x)
        return x


class ResNetBottleNeckLayer(nn.Module):

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1, activation: str = 'relu',
                  reduction: int = 4, downsample: bool = False):
        super().__init__()
        should_apply_shortcut = in_channels != out_channels or stride != 1
        reduces_channels = out_channels // reduction
        self.shortcut(ResNetShortcut(in_channels, out_channels, stride) if should_apply_shortcut else nn.Identity())

        self.layer = nn.Sequential(ResNetConvLayer(in_channels, reduces_channels, kernel_size=1, stride=stride if downsample else 1),
                                    ResNetConvLayer(reduces_channels, reduces_channels, stride=stride, stride=stride if not downsample else 1),
                                    ResNetConvLayer(reduces_channels, out_channels, kernel_size=1, activation=None))
        
        self.activation = nn.ReLU() if activation == 'relu' else nn.Identity()

    def forward(self, x):
        residual = x
        x = self.layer(x)
        residual = self.shortcut(residual)
        x += residual
        x = self.activation(x)
        return x
    

class ResNetStage(nn.Module):

    def __init__(self, in_channels: int, out_channels: int, num_blocks: int,
                  stride: int = 1, depth: int = 2, layer_type: str = 'basic', downsample_in_bottleneck: bool = False):
        super().__init__()

        layer = ResNetBasicLayer if layer_type == 'basic' else ResNetBottleNeckLayer

        if layer_type == "bottleneck":
            first_layer = layer(in_channels, out_channels, stride, downsample=downsample_in_bottleneck)
        
        else:
            first_layer = layer(in_channels, out_channels, stride=stride)
        
        self.layers = nn.Sequential(first_layer, *[layer(out_channels, out_channels) for _ in range(depth - 1)])

    def forward(self, x: torch.Tensor) -> torch.Tensor: 

        for layer in self.layers:
            x = layer(x)

        return x
    

class ResNetEncoder(nn.Module):
    def __init__(self, config: ResNetConfig):
        super().__init__()
        self.stages = nn.ModuleList([])
        # based on `downsample_in_first_stage` the first layer of the first stage may or may not downsample the input
        self.stages.append(
            ResNetStage(
                config,
                config.embedding_size,
                config.hidden_sizes[0],
                stride=2 if config.downsample_in_first_stage else 1,
                depth=config.depths[0],
            )
        )
        in_out_channels = zip(config.hidden_sizes, config.hidden_sizes[1:])
        for (in_channels, out_channels), depth in zip(in_out_channels, config.depths[1:]):
            self.stages.append(ResNetStage(config, in_channels, out_channels, depth=depth))

    def forward(
        self, hidden_state: Tensor, output_hidden_states: bool = False, return_dict: bool = True
    ) -> BaseModelOutputWithNoAttention:
        hidden_states = () if output_hidden_states else None

        for stage_module in self.stages:
            if output_hidden_states:
                hidden_states = hidden_states + (hidden_state,)

            hidden_state = stage_module(hidden_state)

        if output_hidden_states:
            hidden_states = hidden_states + (hidden_state,)

        if not return_dict:
            return tuple(v for v in [hidden_state, hidden_states] if v is not None)

        return BaseModelOutputWithNoAttention(
            last_hidden_state=hidden_state,
            hidden_states=hidden_states,
        )

