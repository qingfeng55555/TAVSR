#!/usr/bin/env python
import math
from re import S
from telnetlib import PRAGMA_HEARTBEAT
import numpy as np
import torch.nn.init as init
import torch
import torch.nn as nn
import torch.nn.functional as F
# Torchvision Implement DCNv1
import torchvision
from contextlib import contextmanager
from common import *
from torchvision import transforms
from model import *
from timm.models.layers import DropPath, to_2tuple, trunc_normal_


# from ops.dcn.deform_conv import ModulatedDeformConv


class RU(nn.Module):
    def __init__(self, H, W, in_chans=1, embed_dim_U=32,
                 embed_dim=60, base_ks=3, deform_ks=3, scale=2):
        super(RU, self).__init__()
        self.in_nc = in_chans
        self.embed_dim = embed_dim
        self.shuffler = PixelShuffle(1 / scale)

        ### shallow feature extraction
        self.conv_ref = nn.Conv2d(self.in_nc * (scale ** 2), self.in_nc, 3, 1, 1)
        self.conv_ref_1 = nn.Conv2d(self.in_nc * (scale ** 2), embed_dim_U * 2, 3, 1, 1)
        self.conv_fus = nn.Conv2d(self.in_nc * 2, embed_dim, 3, 1, 1)
        self.conv_lqs = nn.Conv2d(self.in_nc * 5, embed_dim, 3, 1, 1)

        ### deep feature extraction --> offset
        self.h = H
        self.w = W
        self.embed_dim_u = embed_dim_U
        self.u_transformer_net_3 = U_transformer(H=self.h, W=self.w, dd_in=in_chans * 2, img_size=64,
                                                 embed_dim=self.embed_dim_u,
                                                 win_size=8, token_projection='linear', token_mlp='leff',
                                                 modulator=True)

        ### deep feature extraction --> align_feature
        self.u_transformer_net_4 = U_transformer(H=self.h, W=self.w, dd_in=in_chans * 2, img_size=64,
                                                 embed_dim=self.embed_dim_u,
                                                 win_size=8, token_projection='linear', token_mlp='leff',
                                                 modulator=True)
        # self.conv_channel_ref = nn.Conv2d(3, embed_dim_U, 3, 1, 1)

        ### ref + lq DCN

        self.deform_ks = deform_ks
        self.size_dk = deform_ks ** 2
        self.offset_mask_ref = nn.Conv2d(
            embed_dim_U * 2, self.embed_dim_u * 2 * 3 * self.size_dk, base_ks, padding=base_ks // 2
        )
        self.deform_conv_ref = torchvision.ops.DeformConv2d(64, 64, 3, stride=1, padding=1,
                                                            dilation=1, groups=8, bias=True)

        ### fusion_model ref + lq
        self.fus_1 = fusion_model(embed_dim_U * 2)

        ### U_transformer_net --> offset
        self.h = H
        self.w = W
        self.embed_dim_u = embed_dim_U
        self.u_transformer_net_1 = U_transformer(H=self.h, W=self.w, dd_in=in_chans * 5, img_size=64,
                                                 embed_dim=self.embed_dim_u,
                                                 win_size=8, token_projection='linear', token_mlp='leff',
                                                 modulator=True)

        ## U_transformer_net --> align_feature
        self.u_transformer_net_2 = U_transformer(H=self.h, W=self.w, dd_in=in_chans * 5, img_size=64,
                                                 embed_dim=self.embed_dim_u,
                                                 win_size=8, token_projection='linear', token_mlp='leff',
                                                 modulator=True)
        self.conv_channel = nn.Conv2d(60, embed_dim_U * 2, 3, 1, 1)

        ### lqs DCN

        self.offset_mask_lqs = nn.Conv2d(
            self.embed_dim_u * 2, self.in_nc * 5 * 3 * self.size_dk, base_ks, padding=base_ks // 2
        )
        self.deform_conv_lqs = torchvision.ops.DeformConv2d(5, 60, 3, stride=1, padding=1,
                                                            dilation=1, groups=5, bias=True)

        ### fusion_model lqs
        self.fus_2 = fusion_model(embed_dim_U * 2)

        ### PixelShuffle
        self.shuffler_up = PixelShuffle(scale)

    def forward(self, lqs, ref):
        ### ref cat lq = fus
        feats_ref = self.shuffler(ref)
        ref_feat = self.conv_ref_1(feats_ref)
        feats_ref = self.conv_ref(feats_ref)

        B, T, _, H, W = lqs.size()
        x_center = lqs[:, T // 2, :, :, :]
        x_center = x_center.unsqueeze(1)
        feats_ref = feats_ref.unsqueeze(1)
        fus = torch.cat([x_center, feats_ref], dim=1).view(B, -1, H, W)
        # fus = self.conv_fus(fus)

        ### deep transformer net --> offset
        y = self.u_transformer_net_3(fus)

        ### deep transformer net --> align_feature
        align_ref = self.u_transformer_net_4(fus)
        # align_ref = self.conv_channel_ref(align_ref)

        ### compute fus offset and mask
        off_msk_ref = self.offset_mask_ref(y)  # torch.Size([8, 81, 128, 128])
        off_ref = off_msk_ref[:, :self.embed_dim_u * 2 * 2 * self.deform_ks * self.deform_ks,
                  ...]  # torch.Size([8, 54, 128, 128])
        msk_ref = torch.sigmoid(
            off_msk_ref[:, self.embed_dim_u * 2 * 2 * self.deform_ks * self.deform_ks:, ...]
            # torch.Size([8, 27, 128, 128])
        )

        ### perform deformable convolutional fusion
        fused_ref_feat = F.relu(
            self.deform_conv_ref(ref_feat, off_ref, msk_ref),
            inplace=True
        )

        # fuse align_feature and dcn_feature
        fused_ref_feat = self.fus_1(self.fus_1(self.fus_1(fused_ref_feat, align_ref), align_ref), align_ref)

        ### lqs fus
        input_lqs = lqs.view(B, -1, H, W)

        ### U_tranformer net --> offset
        z = self.u_transformer_net_1(input_lqs)

        ### U_tranformer net --> align_feature
        align_lqs = self.u_transformer_net_2(input_lqs)
        # align_lqs = self.conv_channel(align_lqs)

        ### compute lqs offset and mask
        off_msk_lqs = self.offset_mask_lqs(z)
        off_lqs = off_msk_lqs[:, :self.in_nc * 5 * 2 * self.deform_ks * self.deform_ks, ...]
        msk_lqs = torch.sigmoid(
            off_msk_lqs[:, self.in_nc * 5 * 2 * self.deform_ks * self.deform_ks:, ...]
        )

        # perform deformable convolutional fusion
        fused_lqs_feat = F.relu(
            self.deform_conv_lqs(input_lqs, off_lqs, msk_lqs),
            inplace=True
        )
        fused_lqs_feat = self.conv_channel(fused_lqs_feat)
        # fuse align_feature and dcn_feature
        fused_lqs_feat = self.fus_2(self.fus_2(self.fus_2(fused_lqs_feat, align_lqs), align_lqs), align_lqs)

        ### shuffler
        ref_shuffler = self.shuffler_up(fused_ref_feat)
        lqs_shuffler = self.shuffler_up(fused_lqs_feat)

        return fused_ref_feat, fused_lqs_feat, ref_shuffler, lqs_shuffler


class Decoder(nn.Module):
    def __init__(self, scale=2):
        super(Decoder, self).__init__()
        # self.conv_fusion1 = nn.Conv2d(120, 64, kernel_size=3, stride=1, padding=1, bias=True)
        # self.conv_fusion2 = nn.Conv2d(96, 64, kernel_size=3, stride=1, padding=1, bias=True)
        self.shuffler = PixelShuffle(scale)
        # self.enh_1 = ResBlock(in_feat=64, out_feat=64, kernel_size=3)
        # self.enh_2 = ResBlock(in_feat=64, out_feat=64, kernel_size=3)

        self.fus = fusion_model(64)

    def forward(self, fea_lr, fea_ref):
        feats = self.fus(self.fus(self.fus(fea_lr, fea_ref), fea_ref), fea_ref)

        out = self.shuffler(feats)

        return out


class R3N(nn.Module):
    def __init__(self, H, W, scale=2):
        super(R3N, self).__init__()

        self.scale = scale
        self.h = H
        self.w = W
        self.encoder = RU(H=self.h, W=self.w, in_chans=1, embed_dim=60)

        self.decoder = Decoder(scale=self.scale)
        self.tail = conv3x3(64, 1)

    def forward(self, x, ref):
        B, T, C, H, W = x.size()
        x_center = x[:, T // 2, :, :, :]
        base = F.interpolate(x_center, scale_factor=self.scale, mode='bicubic', align_corners=False)
        # x = x.view(B, -1, H, W)

        # feature extraction and DCNBlock alignment
        feats_ref, feats_lr, ref_sheffle, lqs_sheffle = self.encoder(x, ref)

        # feature fusion and reconstruction
        out = self.decoder(feats_lr, feats_ref)

        out = self.tail(out)
        ref_resnet = self.tail(ref_sheffle)
        lqs_unet = self.tail(lqs_sheffle)

        ref_resnet = ref_resnet + base
        lqs_unet = lqs_unet + base
        out = out + base

        return out, base, ref_resnet, lqs_unet


if __name__ == "__main__":
    # torch.cuda.set_device(1)
    net = R3N(H=208, W=120).cuda()
    from thop import profile
    import cv2
    import os

    os.environ['CUDA_VISIBLE_DEVICES'] = '2'
    # img = cv2.imread('/home/newdata/data/ref_pic/test_ref_qp27/001/001.png', 1).astype(np.uint8)
    # img = cv2.resize(img, (1024, 1024))
    # img = img / 255.
    # img = torch.from_numpy(img)
    # img = torch.tensor(img, dtype=torch.float32)
    # img = img.unsqueeze(0)  # 在第一维度上增加一个维度，作为batch size大小
    # img = img.permute(0, 3, 1, 2)
    input = torch.randn(2, 5, 1, 208, 120).cuda()
    input_1 = torch.randn(2, 1, 416, 240).cuda()
    # dd = img[0, :, :, :]
    # SR_img = transforms.ToPILImage()(dd.cpu())
    # SR_img.save(f'/home/weiliu/project-pycharm/stdf-pytorch/ref_pic/ref.png')
    flops, params = profile(net, inputs=(input, input_1))
    total = sum([param.nelement() for param in net.parameters()])
    print('   Number of params: %.2fM' % (total / 1e6))
    print('   Number of FLOPs: %.2fGFLOPs' % (flops / 1e9))