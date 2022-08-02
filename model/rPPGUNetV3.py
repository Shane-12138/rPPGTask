import torch
import torch.nn as nn
import numpy as np
from utils import trunc_normal_, lecun_normal_

frames = 128
drop_all = 0.0  # dropout 的概率


class ScaledDotProductAttention(nn.Module):
    def __init__(self, d_model, d_qkv):
        super(ScaledDotProductAttention, self).__init__()
        assert d_model % d_qkv == 0, 'D_conv dimensions must be divisible by the d_q.'
        self.d_model = d_model
        self.d_q = d_qkv
        self.d_k = d_qkv
        self.d_v = d_qkv
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, input):
        n_heads = int(self.d_model / self.d_q)

        input_b, input_n, input_d = input.shape

        q = self.w_q(input).view(input_b, -1, n_heads, self.d_q).transpose(1, 2)
        k = self.w_k(input).view(input_b, -1, n_heads, self.d_k).transpose(1, 2)
        v = self.w_v(input).view(input_b, -1, n_heads, self.d_v).transpose(1, 2)
        scores = (torch.matmul(q, k.transpose(-1, -2))) / np.sqrt(k.shape[-1])
        attn = self.softmax(scores)
        output = torch.matmul(attn, v)
        output = output.transpose(1, 2).reshape(input_b, input_n, input_d)

        return output


class Attention(nn.Module):
    def __init__(self, d_model, d_qkv):
        super(Attention, self).__init__()
        self.sdpa = ScaledDotProductAttention(d_model, d_qkv)
        self.norm = nn.LayerNorm(d_model)
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels=d_model, out_channels=d_model, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm1d(d_model),
            nn.ReLU(),
        )

    def forward(self, input):
        output = input.permute(0, 2, 1)
        output = self.norm(output)
        output = self.sdpa(output)
        output = output.permute(0, 2, 1)
        output = self.conv(input + output)
        return output


class EncoderConv(nn.Module):
    def __init__(self, in_c, out_c):
        super(EncoderConv, self).__init__()
        self.encoder = nn.Sequential(
            # 普通卷积
            nn.Conv3d(in_channels=in_c, out_channels=in_c, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(in_c),
            nn.ReLU(),
            nn.Conv3d(in_channels=in_c, out_channels=out_c, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm3d(out_c),
            nn.ReLU(),
        )

    def forward(self, input):
        output = self.encoder(input)
        return output


class Decoder(nn.Module):
    def __init__(self, in_c, out_c):
        super(Decoder, self).__init__()
        self.decoder = nn.Sequential(
            # 普通卷积
            nn.Conv1d(in_channels=in_c * 2, out_channels=in_c * 2, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(in_c * 2),
            nn.ReLU(),
            nn.Conv1d(in_channels=in_c * 2, out_channels=out_c, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm1d(out_c),
            nn.ReLU(),
        )

    def forward(self, x):
        y = self.decoder(x)
        return y


class UpSample4(nn.Module):
    def __init__(self, in_c):
        super(UpSample4, self).__init__()
        # 通道数不变，特征图面积翻倍
        self.up_sample = nn.Sequential(
            nn.ConvTranspose1d(in_channels=in_c, out_channels=in_c, kernel_size=6, stride=4, padding=1),
            nn.BatchNorm1d(in_c),
            nn.ReLU()
        )

    def forward(self, x):
        y = self.up_sample(x)
        return y


class UpSample2(nn.Module):
    def __init__(self, in_c):
        super(UpSample2, self).__init__()
        # 通道数不变，特征图面积翻倍
        self.up_sample = nn.Sequential(
            nn.ConvTranspose1d(in_channels=in_c, out_channels=in_c, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(in_c),
            nn.ReLU()
        )

    def forward(self, x):
        y = self.up_sample(x)
        return y


class ResNet(nn.Module):
    def __init__(self, in_c, mid_c, out_c):
        super(ResNet, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_channels=in_c, out_channels=mid_c, kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1)),
            nn.BatchNorm3d(mid_c),
            nn.ReLU(),
            nn.Conv3d(in_channels=mid_c, out_channels=out_c, kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1)),
            nn.BatchNorm3d(out_c),
            nn.ReLU(),
        )
        self.skip = nn.Sequential(
            nn.Conv3d(in_channels=in_c, out_channels=out_c, kernel_size=(1, 5, 5), stride=(1, 4, 4), padding=(0, 1, 1)),
            nn.BatchNorm3d(out_c),
            nn.ReLU(),
        )

    def forward(self, input):
        out_conv = self.conv(input)
        out_skip = self.skip(input)
        return out_conv + out_skip


class MlpBVP(nn.Module):
    def __init__(self, in_c):
        super(MlpBVP, self).__init__()
        self.mlp = nn.Sequential(
            nn.AdaptiveAvgPool1d(128),
            nn.Conv1d(in_channels=in_c, out_channels=1, kernel_size=1, stride=1, padding=0),
        )

    def forward(self, x):
        x = self.mlp(x)
        y = x.squeeze(1)

        return y


class MlpHr(nn.Module):
    def __init__(self, in_c):
        super(MlpHr, self).__init__()
        self.mlp = nn.Sequential(
            nn.Conv1d(in_channels=in_c, out_channels=in_c, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm1d(in_c),
            nn.ReLU(),
            nn.Conv1d(in_channels=in_c, out_channels=in_c, kernel_size=1, stride=1, padding=0),
        )
        self.fc = nn.Sequential(
            nn.Linear(in_features=frames*in_c, out_features=in_c),
            nn.ReLU(),
            nn.Linear(in_features=in_c, out_features=1)
        )

    def forward(self, x):

        y = self.mlp(x)

        y = y.view(x.shape[0], -1)
        y = self.fc(y)

        return y


class rPPPGUNet(nn.Module):
    def __init__(self):
        super(rPPPGUNet, self).__init__()

        self.encoder_1 = EncoderConv(in_c=3, out_c=32)
        self.encoder_2 = EncoderConv(in_c=32, out_c=64)
        self.encoder_3 = EncoderConv(in_c=64, out_c=128)
        self.encoder_4 = EncoderConv(in_c=128, out_c=256)

        self.pool_1 = nn.Sequential(
            nn.Conv3d(in_channels=32, out_channels=32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=3, stride=2, padding=1)
        )
        self.pool_2 = nn.Sequential(
            nn.Conv3d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=3, stride=2, padding=1)
        )
        self.pool_3 = nn.Sequential(
            nn.Conv3d(in_channels=128, out_channels=128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm3d(128),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=3, stride=2, padding=1)
        )
        self.pool_4 = nn.Sequential(
            nn.Conv3d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(256),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=3, stride=2, padding=1)
        )

        self.en_de = nn.Sequential(
            nn.Conv3d(in_channels=256, out_channels=256, kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1)),
            nn.BatchNorm3d(256),
            nn.ReLU(),
        )

        self.en_de_4 = nn.Sequential(
            nn.Conv3d(in_channels=256, out_channels=256, kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1)),
            nn.BatchNorm3d(256),
            nn.ReLU(),
            nn.Conv3d(in_channels=256, out_channels=256, kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1)),
            nn.BatchNorm3d(256),
            nn.ReLU(),
        )
        self.en_de_3 = nn.Sequential(
            ResNet(in_c=128, mid_c=160, out_c=192),
            ResNet(in_c=192, mid_c=160, out_c=128),
        )
        self.en_de_2 = nn.Sequential(
            ResNet(in_c=64, mid_c=96, out_c=128),
            ResNet(in_c=128, mid_c=128, out_c=96),
            nn.Conv3d(in_channels=96, out_channels=64, kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1)),
            nn.BatchNorm3d(64),
            nn.ReLU(),
        )
        self.en_de_1 = nn.Sequential(
            ResNet(in_c=32, mid_c=64, out_c=96),
            ResNet(in_c=96, mid_c=128, out_c=160),
            ResNet(in_c=160, mid_c=192, out_c=224),
            nn.Conv3d(in_channels=224, out_channels=256, kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1)),
            nn.BatchNorm3d(256),
            nn.ReLU(),
        )

        self.decoder_4 = Decoder(in_c=256, out_c=128)
        self.attn_4 = Attention(d_model=128, d_qkv=32)

        self.decoder_3 = Decoder(in_c=128, out_c=64)
        self.attn_3 = Attention(d_model=64, d_qkv=16)

        self.decoder_2 = Decoder(in_c=64, out_c=32)
        self.attn_2 = Attention(d_model=32, d_qkv=8)

        self.decoder_1 = nn.Sequential(
            nn.Conv1d(in_channels=288, out_channels=288, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm1d(288),
            nn.ReLU(),
        )
        self.attn_1 = Attention(d_model=288, d_qkv=36)

        self.up_sample_4 = UpSample2(256)
        self.up_sample_3 = UpSample4(128)
        self.up_sample_2 = UpSample2(64)
        self.up_sample_1 = UpSample4(32)

        self.mlp_bvp = MlpBVP(288)

        self.init_weights()

    def init_weights(self):
        self.apply(_init_vit_weights)
        print("init weight success!\n")

    def forward(self, video):

        en_out_1 = self.encoder_1(video)

        en_in_2 = self.pool_1(en_out_1)
        en_out_2 = self.encoder_2(en_in_2)

        en_in_3 = self.pool_2(en_out_2)
        en_out_3 = self.encoder_3(en_in_3)

        en_in_4 = self.pool_3(en_out_3)
        en_out_4 = self.encoder_4(en_in_4)
        en_in_de = self.pool_4(en_out_4)

        en_de_out = self.en_de(en_in_de)
        en_de_out = en_de_out.squeeze(-1).squeeze(-1)

        de_in_4_1 = self.en_de_4(en_out_4)
        de_in_4_1 = de_in_4_1.squeeze(-1).squeeze(-1)
        de_in_4_2 = self.up_sample_4(en_de_out)
        de_in_4 = torch.cat((de_in_4_1, de_in_4_2), 1)
        de_out_4 = self.decoder_4(de_in_4)
        de_out_4 = self.attn_4(de_out_4)

        de_in_3_1 = self.en_de_3(en_out_3)
        de_in_3_1 = de_in_3_1.squeeze(-1).squeeze(-1)
        de_in_3_2 = self.up_sample_3(de_out_4)
        de_in_3 = torch.cat((de_in_3_1, de_in_3_2), 1)
        de_out_3 = self.decoder_3(de_in_3)
        de_out_3 = self.attn_3(de_out_3)

        de_in_2_1 = self.en_de_2(en_out_2)
        de_in_2_1 = de_in_2_1.squeeze(-1).squeeze(-1)
        de_in_2_2 = self.up_sample_2(de_out_3)
        de_in_2 = torch.cat((de_in_2_1, de_in_2_2), 1)
        de_out_2 = self.decoder_2(de_in_2)
        de_out_2 = self.attn_2(de_out_2)

        de_in_1_1 = self.en_de_1(en_out_1)
        de_in_1_1 = de_in_1_1.squeeze(-1).squeeze(-1)
        de_in_1_2 = self.up_sample_1(de_out_2)
        de_in_1 = torch.cat((de_in_1_1, de_in_1_2), 1)
        de_out_1 = self.decoder_1(de_in_1)
        de_out_1 = self.attn_1(de_out_1)

        out_bvp = self.mlp_bvp(de_out_1)

        return out_bvp


def _init_vit_weights(module: nn.Module):
    if isinstance(module, nn.Linear):
        trunc_normal_(module.weight, std=.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, (nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.ConvTranspose1d, nn.ConvTranspose2d, nn.ConvTranspose3d)):
        lecun_normal_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, (nn.LayerNorm, nn.GroupNorm, nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
        nn.init.ones_(module.weight)
        nn.init.zeros_(module.bias)