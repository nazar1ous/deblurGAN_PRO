import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.MIMO_FPN.layers import *

class EBlock(nn.Module):
    def __init__(self, out_channel, num_res=8):
        super(EBlock, self).__init__()

        layers = [ResBlock(out_channel, out_channel) for _ in range(num_res)]

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class DBlock(nn.Module):
    def __init__(self, channel, num_res=8):
        super(DBlock, self).__init__()

        layers = [ResBlock(channel, channel) for _ in range(num_res)]
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class AFF(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(AFF, self).__init__()
        self.conv = nn.Sequential(
            BasicConv(in_channel, out_channel, kernel_size=1, stride=1, relu=True),
            BasicConv(out_channel, out_channel, kernel_size=3, stride=1, relu=False)
        )

    def forward(self, x1, x2, x4):
        x = torch.cat([x1, x2, x4], dim=1)
        return self.conv(x)


class SCM(nn.Module):
    def __init__(self, out_plane):
        super(SCM, self).__init__()
        self.main = nn.Sequential(
            BasicConv(3, out_plane // 4, kernel_size=3, stride=1, relu=True),
            BasicConv(out_plane // 4, out_plane // 2, kernel_size=1, stride=1, relu=True),
            BasicConv(out_plane // 2, out_plane // 2, kernel_size=3, stride=1, relu=True),
            BasicConv(out_plane // 2, out_plane - 3, kernel_size=1, stride=1, relu=True)
        )

        self.conv = BasicConv(out_plane, out_plane, kernel_size=1, stride=1, relu=False)

    def forward(self, x):
        x = torch.cat([x, self.main(x)], dim=1)
        return self.conv(x)


class FAM(nn.Module):
    def __init__(self, channel):
        super(FAM, self).__init__()
        self.merge = BasicConv(channel, channel, kernel_size=3, stride=1, relu=False)

    def forward(self, x1, x2):
        x = x1 * x2
        out = x1 + self.merge(x)
        return out


class DAM(nn.Module):
    def __init__(self, channel):
        super(DAM, self).__init__()
        self.diff_conv = BasicConv(3, channel, kernel_size=3, stride=1, relu=False)
        self.orig_feature_conv = BasicConv(channel, channel, kernel_size=3, stride=1, relu=False)
        self.selective_conv = BasicConv(2 * channel, channel, kernel_size=3, stride=1, relu=False)

    def forward(self, output_img, input_img, output_feature):
        _diff = output_img - input_img
        _attention_masks = torch.sigmoid(self.diff_conv(_diff))
        # _updated_feature = self.orig_feature_conv(output_feature)
        _updated_feature = _attention_masks * self.orig_feature_conv(output_feature)
        _updated_feature = torch.cat([_updated_feature, output_feature], dim=1)
        return self.selective_conv(_updated_feature)


class MIMOFPN(nn.Module):
    def __init__(self, num_res=4):
        super(MIMOFPN, self).__init__()

        print("hi from MIMOFPN")

        print(f"{num_res=}")

        base_channel = 32

        self.Encoder = nn.ModuleList([
            EBlock(base_channel, num_res),
            EBlock(base_channel * 2, num_res),
            EBlock(base_channel * 4, num_res),
            EBlock(base_channel * 8, num_res),
        ])

        self.feat_extract = nn.ModuleList([
            BasicConv(3, base_channel, kernel_size=3, relu=True, stride=1),
            BasicConv(base_channel, base_channel * 2, kernel_size=3, relu=True, stride=2),
            BasicConv(base_channel * 2, base_channel * 4, kernel_size=3, relu=True, stride=2),
            BasicConv(base_channel * 4, base_channel * 8, kernel_size=3, relu=True, stride=2),

            BasicConv(base_channel * 8, base_channel * 4, kernel_size=4, relu=True, stride=2, transpose=True),
            BasicConv(base_channel * 4, base_channel * 2, kernel_size=4, relu=True, stride=2, transpose=True),
            BasicConv(base_channel * 2, base_channel, kernel_size=4, relu=True, stride=2, transpose=True),
            BasicConv(base_channel, 3, kernel_size=3, relu=False, stride=1)
        ])

        self.Decoder = nn.ModuleList([
            DBlock(base_channel * 8, num_res),
            DBlock(base_channel * 4, num_res),
            DBlock(base_channel * 2, num_res),
            DBlock(base_channel, num_res)
        ])

        self.merger = nn.Sequential(
            BasicConv(base_channel * 14, base_channel * 2, kernel_size=1, relu=True, stride=1),
            BasicConv(base_channel * 2, base_channel * 2, kernel_size=3, relu=True, stride=1),
            *[ResBlock(base_channel * 2, base_channel * 2) for _ in range(num_res)]
        )

        self.Convs = nn.ModuleList([
            BasicConv(base_channel * 8, base_channel * 4, kernel_size=1, relu=True, stride=1),
            BasicConv(base_channel * 4, base_channel * 2, kernel_size=1, relu=True, stride=1),
            BasicConv(base_channel * 2, base_channel, kernel_size=1, relu=True, stride=1),
        ])

        self.ConvsOut = nn.ModuleList(
            [
                BasicConv(base_channel * 4, 3, kernel_size=3, relu=False, stride=1),
                BasicConv(base_channel * 2, 3, kernel_size=3, relu=False, stride=1),
            ]
        )

        self.DAMs = nn.ModuleList(
            [
                DAM(base_channel * 4),
                DAM(base_channel * 2),
            ]
        )

        self.AFFs = nn.ModuleList([
            AFF(base_channel * 14, base_channel * 4),
            AFF(base_channel * 7, base_channel * 2)
        ])

        self.random = nn.Sequential(
            *[ResBlock(32,32) for i in range(50)]
        )

    def forward(self, x):
        # print("MIMOFPN")

        # Encoder
        x_ = self.feat_extract[0](x)  # 32x256
        res1 = self.Encoder[0](x_)  # 32x256


        # self.random(res1)

        z = self.feat_extract[1](res1)  # 64x128
        res2 = self.Encoder[1](z)  # 64x128

        z = self.feat_extract[2](res2)  # 128x64
        res3 = self.Encoder[2](z)  # 128x64

        z = self.feat_extract[3](res3)  # 256x32
        res4 = self.Encoder[3](z)  # 256x32

        # skip connections
        z21 = F.interpolate(res1, scale_factor=0.5)
        z23 = F.interpolate(res3, scale_factor=2)

        z32 = F.interpolate(res2, scale_factor=0.5)
        z34 = F.interpolate(res4, scale_factor=2)

        res2 = self.AFFs[1](z21, res2, z23)
        res3 = self.AFFs[0](z32, res3, z34)

        # Decoder
        dres4 = self.Decoder[0](res4)
        z = self.feat_extract[4](dres4)  # transpose

        z = torch.cat([z, res3], dim=1)
        z = self.Convs[0](z)  # 1x1
        dres3 = self.Decoder[1](z)
        z = self.feat_extract[5](dres3)  # transpose

        z = torch.cat([z, res2], dim=1)
        z = self.Convs[1](z)  # 1x1
        dres2 = self.Decoder[2](z)

        # merge
        dres4 = F.interpolate(dres4, scale_factor=4)
        dres3 = F.interpolate(dres3, scale_factor=2)
        z = torch.cat([dres4, dres3, dres2], dim=1)
        z = self.merger(z)
        z = self.feat_extract[6](z)  # transpose

        z = torch.cat([res1, z], dim=1)
        z = self.Convs[2](z)  # 1x1
        z = self.Decoder[3](z)

        z = self.feat_extract[7](z)  # transpose

        output_256_256 = z + x

        return output_256_256  # , output_128_128, output_64_64

