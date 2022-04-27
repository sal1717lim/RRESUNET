from torch import nn
import torch
from utils import init_weights


class ResidualConv(nn.Module):
    def __init__(self, input_dim, output_dim, stride, padding):
        super(ResidualConv, self).__init__()

        self.conv_block = nn.Sequential(
            nn.BatchNorm2d(input_dim),
            nn.ReLU(),
            nn.Conv2d(
                input_dim, output_dim, kernel_size=3, stride=stride, padding=padding, bias=False, padding_mode="reflect"
            ),
            nn.BatchNorm2d(output_dim),
            nn.ReLU(),
            nn.Conv2d(output_dim, output_dim, kernel_size=3, padding=1, padding_mode="reflect"),
        )
        self.conv_skip = nn.Sequential(
            nn.Conv2d(input_dim, output_dim, kernel_size=3, stride=stride, padding=1, bias=False,
                      padding_mode="reflect"),
            nn.BatchNorm2d(output_dim),
        )

    def forward(self, x):
        return self.conv_block(x) + self.conv_skip(x)


class Upsample(nn.Module):
    def __init__(self, output_dim):
        super(Upsample, self).__init__()

        self.upsample = nn.Upsample(output_dim, mode='bilinear')

    def forward(self, x):
        return self.upsample(x)


class Generator(nn.Module):
    def __init__(self, channel=6, filters=[16, 32, 64, 128], init_weight=True):
        super(Generator, self).__init__()
        self.input_layer = nn.Sequential(
            nn.Conv2d(channel, filters[0], kernel_size=3, padding=1, bias=False, padding_mode="reflect"),
            nn.BatchNorm2d(filters[0]),
            nn.ReLU(),
            nn.Conv2d(filters[0], filters[0], kernel_size=3, padding=1, bias=False, padding_mode="reflect"),
        )
        self.input_skip = nn.Sequential(
            nn.Conv2d(channel, filters[0], kernel_size=3, padding=1, padding_mode="reflect"),
            nn.BatchNorm2d(filters[0])

        )

        self.residual_conv_1 = ResidualConv(filters[0], filters[1], 2, 1)
        self.residual_conv_2 = ResidualConv(filters[1], filters[2], 2, 1)

        self.bridge = ResidualConv(filters[2], filters[3], 2, 1)

        self.upsample_1 = Upsample((64, 64))
        self.up_residual_conv1 = ResidualConv(filters[3] + filters[2], filters[2], 1, 1)

        self.upsample_2 = Upsample((128, 128))
        self.up_residual_conv2 = ResidualConv(filters[2] + filters[1], filters[1], 1, 1)

        self.upsample_3 = Upsample((256, 256))
        self.up_residual_conv3 = ResidualConv(filters[1] + filters[0], filters[0], 1, 1)

        self.output_layer = nn.Sequential(
            nn.Conv2d(filters[0], 3, 1, 1),
            nn.Tanh(),
        )
        #initialising the wwights using the normal distro
        if init_weight:
            init_weights(self)
            print("ResUnet weights initialised using Normal distribution around 0 with std of 0.02")

    def forward(self, x,xp):
        # Encode
        x = torch.cat([x,xp], dim=1)
        x1 = self.input_layer(x) + self.input_skip(x)
        x2 = self.residual_conv_1(x1)
        x3 = self.residual_conv_2(x2)
        # Bridge
        x4 = self.bridge(x3)
        # Decode

        x4 = self.upsample_1(x4)

        x5 = torch.cat([x4, x3], dim=1)

        x6 = self.up_residual_conv1(x5)

        x6 = self.upsample_2(x6)
        x7 = torch.cat([x6, x2], dim=1)

        x8 = self.up_residual_conv2(x7)

        x8 = self.upsample_3(x8)
        x9 = torch.cat([x8, x1], dim=1)

        x10 = self.up_residual_conv3(x9)

        output = self.output_layer(x10)

        return output


def test():
    x = torch.randn((1, 3, 256, 256))
    model = Generator(channel=3)
    preds = model(x)
    print(model)
    print(preds.shape)


if __name__ == "__main__":
    test()
