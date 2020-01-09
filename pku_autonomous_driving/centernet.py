from torch import nn
import torch
import numpy as np
import torch.nn.functional as F


class double_conv(nn.Module):
    """(conv => BN => ReLU) * 2"""

    def __init__(self, in_ch, out_ch):
        super(double_conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.conv(x)
        return x

class conv1x1(nn.Module):
    """(conv => BN => ReLU) * 2"""

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.conv(x)
        return x

class bottleneck_conv(nn.Module):
    """(conv => BN => ReLU) * 3"""

    def __init__(self, in_ch, mid_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, mid_ch, 1, padding=1),
            nn.BatchNorm2d(mid_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_ch, mid_ch, 3, padding=1),
            nn.BatchNorm2d(mid_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_ch, out_ch, 1, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class up(nn.Module):
    def __init__(self, in_ch, out_ch, bilinear=True):
        super(up, self).__init__()

        #  would be a nice idea if the upsampling could be learned too,
        #  but my machine do not have enough memory to handle all those weights
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_ch // 2, in_ch // 2, 2, stride=2)

        self.conv = double_conv(in_ch, out_ch)

    def forward(self, x1, x2=None):
        x1 = self.up(x1)

        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]  # -> -307
        diffX = x2.size()[3] - x1.size()[3]  # -> -700

        x1 = F.pad(x1, (diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2))

        # for padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd

        if x2 is not None:
            x = torch.cat([x2, x1], dim=1)
        else:
            x = x1
        x = self.conv(x)
        return x


def get_mesh(batch_size, shape_x, shape_y):
    mg_x, mg_y = np.meshgrid(np.linspace(0, 1, shape_y), np.linspace(0, 1, shape_x))
    mg_x = np.tile(mg_x[None, None, :, :], [batch_size, 1, 1, 1]).astype("float32")
    mg_y = np.tile(mg_y[None, None, :, :], [batch_size, 1, 1, 1]).astype("float32")
    mesh = torch.cat([torch.tensor(mg_x), torch.tensor(mg_y)], 1)
    return mesh


class CentResnet(nn.Module):
    """Mixture of previous classes"""

    def __init__(self, base_model, n_classes, use_pos_feature=True):
        super(CentResnet, self).__init__()
        self.base_model = base_model
        self.use_pos_feature = use_pos_feature

        pos_channels = 2 if use_pos_feature else 0
        self.conv0 = conv1x1(256, 256)
        self.conv1 = conv1x1(512, 256)
        self.conv2 = conv1x1(1024, 256)
        self.conv3 = bottleneck_conv(2048 + 3 * 256 + pos_channels, 256, 1024)

        self.mp2 = nn.MaxPool2d(2)
        self.mp4 = nn.MaxPool2d(4)
        self.mp8 = nn.MaxPool2d(8)

        self.up1 = up(1024 + 256, 512)  # + 1024
        self.up2 = up(512 + 256, 256)  # + 1024
        self.outc = nn.Conv2d(256, n_classes, 1)

    def forward(self, x):
        batch_size = x.shape[0]

        # Run frontend network
        feats4, feats8, feats16, feats32 = self.base_model(x)
        feats4_32 = self.conv0(self.mp8(feats4))
        feats8_32 = self.conv1(self.mp4(feats8))
        feats16_32 = self.conv2(self.mp2(feats16))
        feats = torch.cat([feats4_32, feats8_32, feats16_32, feats32], 1)

        # Add positional info
        if self.use_pos_feature:
            mesh = get_mesh(batch_size, feats.shape[2], feats.shape[3]).to(feats.device)
            feats = torch.cat([feats, mesh], 1)
        feats = self.conv3(feats)

        feats = self.up1(feats, feats16[:,:256,:,:])
        feats = self.up2(feats, feats8[:,:256,:,:])
        feats = self.outc(feats)
        return feats
