import torch.nn as nn
import torch

class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout(0.3),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1)
        )
        
        self.decoder = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 3, kernel_size=3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

class SRGAN(nn.Module):
    def __init__(self):
        super(SRGAN, self).__init__()
        self.generator = ESPCN(scale_factor=4)
        self.discriminator = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=3//2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=3//2, stride=2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, padding=3//2),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=3//2, stride=2),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, kernel_size=3, padding=3//2),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=3//2, stride=2),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 512, kernel_size=3, padding=3//2),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=3//2, stride=2),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(512, 1024, kernel_size=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(1024, 1, kernel_size=1),
            nn.Flatten(),
            nn.Linear(1, 3 * 256 * 256),
            nn.Unflatten(1, (3, 256, 256))
        )

    def forward(self, x):
        x = self.generator(x)
        x = self.discriminator(x)
        return x
    
class ESPCN(nn.Module):
    def __init__(self, scale_factor=1, noOfConvBlocks=2, noOfChannels=64):
        super(ESPCN, self).__init__()
        layers = []
        for idx in range(noOfConvBlocks):
            if idx == 0:
                layers.append(nn.Conv2d(3, noOfChannels, kernel_size=5, padding=5 // 2))
                layers.append(nn.Tanh())
            else:
                layers.append(nn.Conv2d(noOfChannels, noOfChannels//2, kernel_size=3, padding=3 // 2))
                layers.append(nn.Tanh())
                noOfChannels //= 2

        self.feature_extraction = nn.Sequential(
            *layers
        )
        self.sub_pixel_conv = nn.Conv2d(int(noOfChannels), 3 * (scale_factor ** 2), kernel_size=3, padding=3//2)
        self.pixel_shuffle = nn.Sequential(
            nn.PixelShuffle(scale_factor),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.feature_extraction(x)
        x = self.sub_pixel_conv(x)
        x = self.pixel_shuffle(x)
        return x

class ChannelAttention(nn.Module):
    def __init__(self, in_channels, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_channels // ratio, in_channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=(kernel_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv(x)
        return self.sigmoid(x)

class SANBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(SANBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()
        self.ca = ChannelAttention(out_channels)
        self.sa = SpatialAttention()

    def forward(self, x):
        out = self.conv1(x)
        out = self.relu(out)
        out = self.ca(out) * out
        out = self.sa(out) * out
        return out

class SAN(nn.Module):
    def __init__(self, scale_factor=1, num_blocks=16):
        super(SAN, self).__init__()
        self.init_conv = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        self.blocks = nn.ModuleList([SANBlock(64, 64) for _ in range(num_blocks)])
        self.conv2 = nn.Conv2d(64, 3 * (scale_factor ** 2), kernel_size=3, stride=1, padding=1)
        self.pixel_shuffle = nn.PixelShuffle(scale_factor)

    def forward(self, x):
        out = self.init_conv(x)
        for block in self.blocks:
            out = block(out)
        out = self.conv2(out)
        out = self.pixel_shuffle(out)
        return out

class VDSR(nn.Module):
    def __init__(self):
        super(VDSR, self).__init__()

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.residual_blocks = self.make_residual_blocks()
        self.conv2 = nn.Conv2d(64, 3, kernel_size=3, padding=1)

    def make_residual_blocks(self, num_blocks=18):
        layers = []
        for _ in range(num_blocks):
            layers.append(nn.Conv2d(64, 64, kernel_size=3, padding=1))
            layers.append(nn.ReLU(inplace=True))
        return nn.Sequential(*layers)

    def forward(self, x):
        residual = x
        out = self.relu(self.conv1(x))
        out = self.residual_blocks(out)
        out = self.conv2(out)
        out = torch.add(out, residual)
        return out