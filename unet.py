import torch.nn as nn
import torch
from torch.nn import functional as F
from networks.layers import BasicConv3d, FastSmoothSeNormConv3d, RESseNormConv3d, UpConv


class FastSmoothSENormDeepUNet_supervision_skip_no_drop(nn.Module):
    """The model presented in the paper. This model is one of the multiple models that we tried in our experiments
    that it why it has such an awkward name."""

    def __init__(self, in_channels=1, n_cls=33, n_filters=12, reduction=2, return_logits=False):
        super(FastSmoothSENormDeepUNet_supervision_skip_no_drop, self).__init__()
        self.in_channels = in_channels
        self.n_cls = 1 if n_cls == 2 else n_cls
        self.n_filters = n_filters
        self.return_logits = return_logits

        self.block_1_1_left = RESseNormConv3d(in_channels, n_filters, reduction, kernel_size=7, stride=1, padding=3)
        self.block_1_2_left = RESseNormConv3d(n_filters, n_filters, reduction, kernel_size=3, stride=1, padding=1)

        self.pool_1 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.block_2_1_left = RESseNormConv3d(n_filters, 2 * n_filters, reduction, kernel_size=3, stride=1, padding=1)
        self.block_2_2_left = RESseNormConv3d(2 * n_filters, 2 * n_filters, reduction, kernel_size=3, stride=1, padding=1)
        self.block_2_3_left = RESseNormConv3d(2 * n_filters, 2 * n_filters, reduction, kernel_size=3, stride=1, padding=1)

        self.pool_2 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.block_3_1_left = RESseNormConv3d(2 * n_filters, 4 * n_filters, reduction, kernel_size=3, stride=1, padding=1)
        self.block_3_2_left = RESseNormConv3d(4 * n_filters, 4 * n_filters, reduction, kernel_size=3, stride=1, padding=1)
        self.block_3_3_left = RESseNormConv3d(4 * n_filters, 4 * n_filters, reduction, kernel_size=3, stride=1, padding=1)

        self.pool_3 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.block_4_1_left = RESseNormConv3d(4 * n_filters, 8 * n_filters, reduction, kernel_size=3, stride=1, padding=1)
        self.block_4_2_left = RESseNormConv3d(8 * n_filters, 8 * n_filters, reduction, kernel_size=3, stride=1, padding=1)
        self.block_4_3_left = RESseNormConv3d(8 * n_filters, 8 * n_filters, reduction, kernel_size=3, stride=1, padding=1)

        self.pool_4 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.block_5_1_left = RESseNormConv3d(8 * n_filters, 16 * n_filters, reduction, kernel_size=3, stride=1, padding=1)
        self.block_5_2_left = RESseNormConv3d(16 * n_filters, 16 * n_filters, reduction, kernel_size=3, stride=1, padding=1)
        self.block_5_3_left = RESseNormConv3d(16 * n_filters, 16 * n_filters, reduction, kernel_size=3, stride=1, padding=1)

        self.upconv_4 = nn.ConvTranspose3d(16 * n_filters, 8 * n_filters, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.block_4_1_right = FastSmoothSeNormConv3d((8 + 8) * n_filters, 8 * n_filters, reduction, kernel_size=3, stride=1, padding=1)
        self.block_4_2_right = FastSmoothSeNormConv3d(8 * n_filters, 8 * n_filters, reduction, kernel_size=3, stride=1, padding=1)
        self.vision_4 = UpConv(8 * n_filters, n_filters, reduction, scale=8)

        self.upconv_3 = nn.ConvTranspose3d(8 * n_filters, 4 * n_filters, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.block_3_1_right = FastSmoothSeNormConv3d((4 + 4) * n_filters, 4 * n_filters, reduction, kernel_size=3, stride=1, padding=1)
        self.block_3_2_right = FastSmoothSeNormConv3d(4 * n_filters, 4 * n_filters, reduction, kernel_size=3, stride=1, padding=1)
        self.vision_3 = UpConv(4 * n_filters, n_filters, reduction, scale=4)

        self.upconv_2 = nn.ConvTranspose3d(4 * n_filters, 2 * n_filters, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.block_2_1_right = FastSmoothSeNormConv3d((2 + 2) * n_filters, 2 * n_filters, reduction, kernel_size=3, stride=1, padding=1)
        self.block_2_2_right = FastSmoothSeNormConv3d(2 * n_filters, 2 * n_filters, reduction, kernel_size=3, stride=1, padding=1)
        self.vision_2 = UpConv(2 * n_filters, n_filters, reduction, scale=2)

        self.upconv_1 = nn.ConvTranspose3d(2 * n_filters, 1 * n_filters, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.block_1_1_right = FastSmoothSeNormConv3d((1 + 1) * n_filters, n_filters, reduction, kernel_size=3, stride=1, padding=1)
        self.block_1_2_right = FastSmoothSeNormConv3d(n_filters, n_filters, reduction, kernel_size=3, stride=1, padding=1)

        self.conv1x1 = nn.Conv3d(1 * n_filters, self.n_cls, kernel_size=1, stride=1, padding=0)

    def forward(self, x):

        ds0 = self.block_1_2_left(self.block_1_1_left(x))
        ds1 = self.block_2_3_left(self.block_2_2_left(self.block_2_1_left(self.pool_1(ds0))))
        ds2 = self.block_3_3_left(self.block_3_2_left(self.block_3_1_left(self.pool_2(ds1))))
        ds3 = self.block_4_3_left(self.block_4_2_left(self.block_4_1_left(self.pool_3(ds2))))
        x = self.block_5_3_left(self.block_5_2_left(self.block_5_1_left(self.pool_4(ds3))))

        x = self.block_4_2_right(self.block_4_1_right(torch.cat([self.upconv_4(x), ds3], 1)))
        sv4 = self.vision_4(x)

        x = self.block_3_2_right(self.block_3_1_right(torch.cat([self.upconv_3(x), ds2], 1)))
        sv3 = self.vision_3(x)

        x = self.block_2_2_right(self.block_2_1_right(torch.cat([self.upconv_2(x), ds1], 1)))
        sv2 = self.vision_2(x)

        x = self.block_1_1_right(torch.cat([self.upconv_1(x), ds0], 1))
        x = x + sv4 + sv3 + sv2
        x = self.block_1_2_right(x)

        x = self.conv1x1(x)

        return x


class BaselineUNet(nn.Module):
    def __init__(self, in_channels=1, n_cls=33, n_filters=12):
        super(BaselineUNet, self).__init__()
        self.in_channels = in_channels
        self.n_cls = 1 if n_cls == 2 else n_cls
        self.n_filters = n_filters

        self.block_1_1_left = BasicConv3d(in_channels, n_filters, kernel_size=3, stride=1, padding=1)
        self.block_1_2_left = BasicConv3d(n_filters, n_filters, kernel_size=3, stride=1, padding=1)

        self.pool_1 = nn.MaxPool3d(kernel_size=2, stride=2)  # 64, 1/2
        self.block_2_1_left = BasicConv3d(n_filters, 2 * n_filters, kernel_size=3, stride=1, padding=1)
        self.block_2_2_left = BasicConv3d(2 * n_filters, 2 * n_filters, kernel_size=3, stride=1, padding=1)

        self.pool_2 = nn.MaxPool3d(kernel_size=2, stride=2)  # 128, 1/4
        self.block_3_1_left = BasicConv3d(2 * n_filters, 4 * n_filters, kernel_size=3, stride=1, padding=1)
        self.block_3_2_left = BasicConv3d(4 * n_filters, 4 * n_filters, kernel_size=3, stride=1, padding=1)

        self.pool_3 = nn.MaxPool3d(kernel_size=2, stride=2)  # 256, 1/8
        self.block_4_1_left = BasicConv3d(4 * n_filters, 8 * n_filters, kernel_size=3, stride=1, padding=1)
        self.block_4_2_left = BasicConv3d(8 * n_filters, 8 * n_filters, kernel_size=3, stride=1, padding=1)

        self.upconv_3 = nn.ConvTranspose3d(8 * n_filters, 4 * n_filters, kernel_size=3, stride=2, padding=1,
                                           output_padding=1)
        self.block_3_1_right = BasicConv3d((4 + 4) * n_filters, 4 * n_filters, kernel_size=3, stride=1, padding=1)
        self.block_3_2_right = BasicConv3d(4 * n_filters, 4 * n_filters, kernel_size=3, stride=1, padding=1)

        self.upconv_2 = nn.ConvTranspose3d(4 * n_filters, 2 * n_filters, kernel_size=3, stride=2, padding=1,
                                           output_padding=1)
        self.block_2_1_right = BasicConv3d((2 + 2) * n_filters, 2 * n_filters, kernel_size=3, stride=1, padding=1)
        self.block_2_2_right = BasicConv3d(2 * n_filters, 2 * n_filters, kernel_size=3, stride=1, padding=1)

        self.upconv_1 = nn.ConvTranspose3d(2 * n_filters, n_filters, kernel_size=3, stride=2, padding=1,
                                           output_padding=1)
        self.block_1_1_right = BasicConv3d((1 + 1) * n_filters, n_filters, kernel_size=3, stride=1, padding=1)
        self.block_1_2_right = BasicConv3d(n_filters, n_filters, kernel_size=3, stride=1, padding=1)

        self.conv1x1 = nn.Conv3d(n_filters, self.n_cls, kernel_size=1, stride=1, padding=0)
        # self.avgpool = nn.AdaptiveAvgPool3d((1, None, None))

    def forward(self, x, softmax=False):
        ds0 = self.block_1_2_left(self.block_1_1_left(x))
        ds1 = self.block_2_2_left(self.block_2_1_left(self.pool_1(ds0)))
        ds2 = self.block_3_2_left(self.block_3_1_left(self.pool_2(ds1)))
        x = self.block_4_2_left(self.block_4_1_left(self.pool_3(ds2)))

        x = self.block_3_2_right(self.block_3_1_right(torch.cat([self.upconv_3(x), ds2], 1)))
        x = self.block_2_2_right(self.block_2_1_right(torch.cat([self.upconv_2(x), ds1], 1)))
        x = self.block_1_2_right(self.block_1_1_right(torch.cat([self.upconv_1(x), ds0], 1)))
        x = self.conv1x1(x)
        #
        # if softmax is True:
        #     x = torch.nn.functional.softmax(x, dim=1)

        return x


class Down(nn.Module):
    def __init__(self, inc, m, n, c, pool=True):
        super(Down, self).__init__()
        self.m = m
        self.n = n
        self.c = c
        self.pool = pool

        layers = []

        if self.m > 0:
            for i in range(self.m):
                layers.append(nn.Conv3d(inc, inc, kernel_size=(1, 3, 3), stride=(1, 1, 1), padding=(0, 1, 1)))
                layers.append(nn.ReLU())

        if self.n > 0:
            for i in range(self.n):
                layers.append(nn.Conv3d(inc, inc, kernel_size=(1, 3, 3), stride=(1, 1, 1), padding=(0, 1, 1)))

                layers.append(nn.Conv3d(inc, inc, kernel_size=(3, 1, 1), stride=(1, 1, 1), padding=(0, 0, 0)))
                layers.append(nn.ReLU())

        layers.append(nn.Conv3d(inc, c, kernel_size=(3, 1, 1), stride=(1, 1, 1), padding=(1, 0, 0)))

        self.layers = nn.Sequential(*layers)

        if pool:
            self.avgpool = nn.AvgPool3d(kernel_size=(1, 2, 2))

    def forward(self, x):
        x = self.layers(x)

        if self.pool is True:
            down_x = self.avgpool(x)
            return x, down_x

        return x


class Up(nn.Module):
    def __init__(self, inc, m, c, upscale=True):
        super(Up, self).__init__()

        self.upscale = upscale
        if upscale is True:
            self.tranconv = nn.ConvTranspose3d(inc, c, kernel_size=(1, 2, 2), stride=(1, 2, 2))
            self.reducedim = nn.Conv3d(c, c, kernel_size=(7, 1, 1), stride=(1, 1, 1), padding=(0, 0, 0))

        self.relu = nn.ReLU()

        layers = []
        if m > 0:
            for i in range(m):
                layers.append(nn.Conv3d(c, c, kernel_size=(1, 3, 3), stride=(1, 1, 1), padding=(0, 1, 1)))
                layers.append(nn.ReLU())
        self.convs = nn.Sequential(*layers)

    def forward(self, x, down_x):
        if self.upscale is True:
            x = torch.cat((x, self.tranconv(down_x)), dim=2)
            x = self.reducedim(x)

        x = x + self.convs(self.relu(x))

        return x


class Fc(nn.Module):
    def __init__(self, inc=256, c=1024):
        super(Fc, self).__init__()

        self.conv1 = nn.Conv3d(inc, c, kernel_size=(1, 8, 8), stride=(1, 1, 1), padding=(0, 0, 0))
        self.relu = nn.ReLU()
        self.conv2 = nn.Linear(c, c)
        self.conv3 = nn.Linear(c, c)
        self.conv4 = nn.Linear(c, 8 * 8 * 256)

    def forward(self, x):
        x = self.conv1(x)
        x = x.view(x.shape[0], x.shape[1])
        x = x + self.conv2(self.relu(x))
        x = x + self.conv3(self.relu(x))
        x = self.conv4(self.relu(x))
        x = x.view(x.shape[0], 256, 1, 8, 8)
        return x


class UNet3D(nn.Module):
    def __init__(self, inc=1, num_class=33):
        super(UNet3D, self).__init__()

        self.num_class = num_class

        self.down1 = Down(1, 3, 0, 32)
        self.down2 = Down(32, 3, 0, 32)
        self.down3 = Down(32, 3, 0, 64)
        self.down4 = Down(64, 1, 2, 64)
        self.down5 = Down(64, 1, 2, 128)
        self.down6 = Down(128, 1, 2, 128)
        self.down7 = Down(128, 0, 4, 256, pool=False)

        self.fc = Fc()

        self.up1 = Up(256, 4, 256, False)
        self.up2 = Up(256, 4, 128)
        self.up3 = Up(128, 4, 128)
        self.up4 = Up(128, 3, 64)
        self.up5 = Up(64, 3, 64)
        self.up6 = Up(64, 3, 32)
        self.up7 = Up(32, 1, 32)

        self.classifier = nn.Sequential(
            nn.Conv3d(32, num_class, kernel_size=(1, 1, 1)),
            nn.Sigmoid()
        )

    def forward(self, x):
        x1, down_x1 = self.down1(x)
        x2, down_x2 = self.down2(down_x1)
        x3, down_x3 = self.down3(down_x2)
        x4, down_x4 = self.down4(down_x3)
        x5, down_x5 = self.down5(down_x4)
        x6, down_x6 = self.down6(down_x5)
        x7 = self.down7(down_x6)

        x8 = self.fc(x7)

        x = self.up1(x7, x8)
        x = self.up2(x6, x)
        x = self.up3(x5, x)
        x = self.up4(x4, x)
        x = self.up5(x3, x)
        x = self.up6(x2, x)
        x = self.up7(x1, x)

        x = self.classifier(x)

        return x


if __name__ == '__main__':
    model = BaselineUNet()

    x = torch.zeros((1, 1, 24, 512, 512))

    x = model(x)
