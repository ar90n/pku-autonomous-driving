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

        # Lateral layers convert resnet outputs to a common feature size
        # self.lat8 = nn.Conv2d(256, 256, 1)
        self.lat16 = nn.Conv2d(512, 256, 1)
        self.lat32 = self.lat16  # nn.Conv2d(512, 512, 1)
        self.bn8 = nn.GroupNorm(16, 256)
        self.bn16 = nn.GroupNorm(16, 256)
        self.bn32 = nn.GroupNorm(16, 256)

        pos_channels = 2 if use_pos_feature else 0
        self.conv0 = double_conv(3 + pos_channels, 64)
        self.conv1 = double_conv(64, 128)
        self.conv2 = double_conv(128, 512)
        self.conv3 = double_conv(512, 1024)
        self.conv4 = bottleneck_conv(1024, 256, 1024)
        self.conv5 = nn.Conv2d(512, 256, 1)

        self.mp = nn.MaxPool2d(2)

        self.up1 = up(1280 + pos_channels, 512)  # + 1024
        self.up2 = up(512 + 512, 256)
        self.outc = nn.Conv2d(256, n_classes, 1)

    def forward(self, x):
        # Run frontend network
        feats32 = self.base_model(x)
        # lat8 = F.relu(self.bn8(self.lat8(feats8)))
        # lat16 = F.relu(self.bn16(self.lat16(feats16)))
        lat32 = F.relu(self.bn32(self.lat32(feats32)))

        # Add positional info
        if self.use_pos_feature:
            batch_size = x.shape[0]
            mesh1 = get_mesh(batch_size, x.shape[2], x.shape[3]).to(x.device)
            x0 = torch.cat([x, mesh1], 1)

            mesh2 = get_mesh(batch_size, lat32.shape[2], lat32.shape[3]).to(lat32.device)
            feats = torch.cat([lat32, mesh2], 1)
        else:
            x0 = x
            feats = lat32

        x1 = self.mp(self.conv0(x0))
        x2 = self.mp(self.conv1(x1))
        x3 = self.mp(self.conv2(x2))
        x4 = self.mp(self.conv3(x3))

        x5 = self.conv4(x4)

        x = self.up1(x5, feats)
        x = self.conv5(x)
        x = self.outc(x)
        return x


#    def forward(self, x):
#        batch_size = x.shape[0]
#        mesh1 = get_mesh(batch_size, x.shape[2], x.shape[3]).to(x.device)
#        x0 = torch.cat([x, mesh1], 1)  # -> [1, 5, 700, 1600]
#        x1 = self.mp(self.conv0(x0))   # -> [1, 64, 350, 800]
#        x2 = self.mp(self.conv1(x1))   # -> [1, 128, 175, 400]
#        x3 = self.mp(self.conv2(x2))   # -> [1, 512, 87, 200]
#        x4 = self.mp(self.conv3(x3))   # -> [1, 1024, 43, 100]
#
#        # feats = self.base_model.extract_features(x)
#        # Run frontend network
#        feats32 = self.base_model(x)   # -> [1, 256, 175, 400]
#        # lat8 = F.relu(self.bn8(self.lat8(feats8)))
#        # lat16 = F.relu(self.bn16(self.lat16(feats16)))
#        lat32 = F.relu(self.bn32(self.lat32(feats32)))  # -> [1, 256, 175, 400]
#
#        # Add positional info
#        mesh2 = get_mesh(batch_size, lat32.shape[2], lat32.shape[3]).to(lat32.device)
#        feats = torch.cat([lat32, mesh2], 1)   # -> [1, 258, 175, 400]
#        x = self.up1(feats, x4)  # -> [1, 512, 43, 100]
#        x = self.up2(x, x3)  # -> [1, 256, 87, 200]
#        x = self.outc(x)   # -> [1, 8, 87, 200]
#        return x
