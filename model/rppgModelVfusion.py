import torch
import torch.nn as nn
import numpy as np
from einops import rearrange
from utils import trunc_normal_, lecun_normal_
from config import Config

model_config = Config()
drop_all = model_config.dropout  # dropout 的概率
frames = model_config.frame_size
layers = model_config.n_layers


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
        self.fc = nn.Linear(d_model, d_model)
        self.drop = nn.Dropout(drop_all)

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
        output = self.fc(output)
        output = self.drop(output)

        return output


class MlpBVP(nn.Module):
    def __init__(self, d_model):
        super(MlpBVP, self).__init__()
        self.mlp = nn.Sequential(
            nn.AdaptiveAvgPool3d((frames, 1, 1)),
            nn.Conv3d(in_channels=d_model, out_channels=1, kernel_size=1, stride=1, padding=0),
        )

    def forward(self, x):

        x = self.mlp(x)
        y = x.squeeze(3).squeeze(3).squeeze(1)

        return y


class MlpHr(nn.Module):
    def __init__(self, d_model):
        super(MlpHr, self).__init__()
        self.mlp = nn.Sequential(
            nn.Conv3d(in_channels=d_model, out_channels=d_model, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm3d(d_model),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool3d((frames, 1, 1)),
        )
        self.fc = nn.Sequential(
            nn.Linear(in_features=frames*d_model, out_features=d_model),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=d_model, out_features=1)
        )

    def forward(self, x):

        y = self.mlp(x)

        y = y.view(x.shape[0], -1)
        y = self.fc(y)

        return y


class FusionBlock(nn.Module):
    def __init__(self, d_model):
        super(FusionBlock, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv3d(in_channels=d_model*2, out_channels=d_model, kernel_size=(1, 3, 3), stride=(1, 1, 1), padding=(0, 1, 1)),
            nn.BatchNorm3d(d_model),
            nn.ReLU(inplace=True),
        )

    def forward(self, x_res, x_attn):
        x_all = torch.cat((x_res, x_attn), 1)
        y = self.conv(x_all)

        return y


class TimeSpacaAttention(nn.Module):
    def __init__(self, d_model, d_qkv):
        super(TimeSpacaAttention, self).__init__()
        self.sdpa_s = ScaledDotProductAttention(d_model, d_qkv)

        self.norm_s = nn.LayerNorm(d_model)

        self.sdpa_t = ScaledDotProductAttention(d_model, d_qkv)

        self.norm_t = nn.LayerNorm(d_model)

        self.conv_in = nn.Sequential(
            nn.Conv3d(in_channels=d_model, out_channels=d_model, kernel_size=(1, 3, 3), stride=(1, 1, 1), padding=(0, 1, 1)),
            nn.BatchNorm3d(d_model),
            nn.ReLU(inplace=True),
        )

        self.fusion = FusionBlock(d_model)

    def forward(self, input_attn):
        B, C, T, Ph, Pw = input_attn.shape

        # 层归一化
        output_s = rearrange(input_attn, 'b c t ph pw -> (b t) (ph pw) c')
        output_s = self.norm_s(output_s)

        output_s = self.sdpa_s(output_s)
        output_s = rearrange(output_s, '(b t) (ph pw) c -> b c t ph pw', b=B, c=C, t=T, ph=Ph, pw=Pw)
        output_s = output_s + input_attn

        # 层归一化
        output_t = rearrange(output_s, 'b c t ph pw -> (b ph pw) t c')
        output_t = self.norm_t(output_t)

        output_t = self.sdpa_t(output_t)
        output_t = rearrange(output_t, '(b ph pw) t c -> b c t ph pw', b=B, c=C, t=T, ph=Ph, pw=Pw)
        output_t = output_t + output_s

        output_in = self.conv_in(input_attn)

        output = self.fusion(output_in, output_t)
        return output


class ConvAttn(nn.Module):
    def __init__(self, d_conv, n_layers, d_qkv):
        super(ConvAttn, self).__init__()
        self.attn_layers = nn.ModuleList([TimeSpacaAttention(d_conv, d_qkv) for _ in range(n_layers)])

    def forward(self, input_conv):
        output_attn = input_conv
        for layer in self.attn_layers:
            output_attn = layer(output_attn)

        return output_attn


class ResBlock(nn.Module):
    # 只进行残差连接提高维度，降低尺寸，不进行注意力计算
    def __init__(self, d_conv):
        super(ResBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1)),
            nn.Conv3d(in_channels=d_conv, out_channels=d_conv*2, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm3d(d_conv*2),
            nn.ReLU(inplace=True),
        )
        self.skip_conv = nn.Sequential(
            nn.Conv3d(in_channels=d_conv, out_channels=d_conv * 2, kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1)),
            nn.BatchNorm3d(d_conv*2),
            nn.ReLU(inplace=True),
        )

    def forward(self, input_conv):
        input_skip = input_conv
        output_conv = self.conv(input_conv)
        output_skip = self.skip_conv(input_skip)
        output = output_conv + output_skip

        return output


class ResAttnBlock(nn.Module):
    # 既进行残差连接，又进行注意力计算
    def __init__(self, in_conv, out_conv, n_layers, d_qkv):
        super(ResAttnBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1)),
            nn.Conv3d(in_channels=in_conv, out_channels=out_conv, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm3d(out_conv),
            nn.ReLU(inplace=True),
        )
        self.skip_conv = nn.Sequential(
            nn.Conv3d(in_channels=in_conv, out_channels=out_conv, kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1)),
            nn.BatchNorm3d(out_conv),
            nn.ReLU(inplace=True),
        )
        self.conv_attn = ConvAttn(out_conv, n_layers, d_qkv)

    def forward(self, input_conv):
        input_skip = input_conv
        output_conv = self.conv(input_conv)
        output_skip = self.skip_conv(input_skip)
        output = output_conv + output_skip

        output_attn = self.conv_attn(output)

        return output_attn


class rPPGTR(nn.Module):
    def __init__(self):
        super(rPPGTR, self).__init__()

        self.conv_patch = nn.Sequential(
            nn.Conv3d(in_channels=3, out_channels=32, kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1)),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
            nn.Conv3d(in_channels=32, out_channels=64, kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1)),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
        )

        self.conv_block_1 = ResAttnBlock(64, 128, n_layers=layers, d_qkv=32)
        self.conv_block_2 = ResAttnBlock(128, 192, n_layers=layers, d_qkv=48)
        self.conv_block_3 = ResAttnBlock(192, 256, n_layers=layers, d_qkv=64)

        self.mlp_bvp = MlpBVP(256)
        self.mlp_hr = MlpHr(256)

        self.init_weights()

    def init_weights(self):
        self.apply(_init_vit_weights)
        print("init weight success!\n")

    def forward(self, video):

        output_cnn = self.conv_patch(video)

        output_cnn = self.conv_block_1(output_cnn)

        output_cnn = self.conv_block_2(output_cnn)

        output_cnn = self.conv_block_3(output_cnn)

        output_bvp = self.mlp_bvp(output_cnn)
        output_hr = self.mlp_hr(output_cnn)

        # output_bvp: [B, 128]
        # output_hr: [B, 1]
        return output_bvp, output_hr


def _init_vit_weights(module: nn.Module):
    if isinstance(module, nn.Linear):
        trunc_normal_(module.weight, std=.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, (nn.Conv2d, nn.Conv3d)):
        lecun_normal_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, (nn.LayerNorm, nn.GroupNorm, nn.BatchNorm2d, nn.BatchNorm3d)):
        nn.init.ones_(module.weight)
        nn.init.zeros_(module.bias)