import os
import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops
import mindspore.ops.functional as F
from mindspore.common.initializer import initializer as init
import math
import mindvision.classification.models as models
import numpy as np
from mindvision.utils.load_pretrained_model import LoadPretrainedModel
from mindvision.classification.utils.model_urls import model_urls
from mindvision.dataset.download import DownLoad


class DoubleConv(nn.Cell):
    def __init__(self, in_channels, out_channels, mid_channels=None, norm_num_groups=32):
        super(DoubleConv, self).__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.SequentialCell(
            nn.Conv2d(in_channels=in_channels, out_channels=mid_channels, kernel_size=3, padding=1, pad_mode='pad', has_bias=True, weight_init='HeUniform'),
            nn.GroupNorm(num_groups=norm_num_groups, num_channels=mid_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels=mid_channels, out_channels=out_channels, kernel_size=3, padding=1, pad_mode='pad', has_bias=True, weight_init='HeUniform'),
            nn.GroupNorm(num_groups=norm_num_groups, num_channels=out_channels),
            nn.ReLU(),
        )

    def construct(self, x):
        return self.double_conv(x)


class Down(nn.Cell):
    def __init__(self, in_channels, out_channels, norm_num_groups=32):
        super(Down, self).__init__()
        self.maxpool_conv = nn.SequentialCell(
            nn.MaxPool2d(kernel_size=2, stride=2),
            DoubleConv(in_channels=in_channels, out_channels=out_channels, norm_num_groups=norm_num_groups)
        )

    def construct(self, x):
        return self.maxpool_conv(x)


class Up(nn.Cell):
    def __init__(self, in_channels, out_channels, bilinear=True, norm_num_groups=32):
        super(Up, self).__init__()
        self.bilinear = bilinear
        if bilinear:
            self.conv = DoubleConv(in_channels=in_channels, out_channels=out_channels, mid_channels=in_channels // 2, norm_num_groups=norm_num_groups)
        else:
            self.up = nn.Conv2dTranspose(in_channels=in_channels, out_channels=in_channels//2, kernel_size=2, stride=2, has_bias=True, weight_init='HeUniform')
            self.conv = DoubleConv(in_channels=in_channels, out_channels=out_channels, norm_num_groups=norm_num_groups)

    def construct(self, x1, x2):
        _, _, H, W = x1.shape
        if self.bilinear:
            x1 = ops.ResizeBilinear((H * 2, W * 2))(x1)
        else:
            x1 = self.up(x1)
        x = F.concat([x1, x2], axis=1)
        return self.conv(x)


class Attention(nn.Cell):
    def __init__(self, dim=256, num_heads=8, bias=True):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        self.temperature = 1 / math.sqrt(dim / num_heads)
        self.q_conv = nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=1, has_bias=bias, weight_init='HeUniform')
        self.k_conv = nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=1, has_bias=bias, weight_init='HeUniform')
        self.v_conv = nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=1, has_bias=bias, weight_init='HeUniform')
        self.out_conv = nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=1, has_bias=bias, weight_init='HeUniform')
        self.gamma = ms.Parameter(default_input=init('zeros', [1], ms.float32))
        self.fusion_conv = nn.Conv2d(in_channels=2 * dim, out_channels=dim, kernel_size=1, has_bias=bias, weight_init='HeUniform')

    def construct(self, x):
        b, c, h, w = x.shape
        num = h * w

        q = self.q_conv(x)
        k = self.k_conv(x)
        v = self.v_conv(x)

        q = q.reshape(b * self.num_heads, -1, num).transpose(0, 2, 1)  # (Bh, N, E)
        k = k.reshape(b * self.num_heads, -1, num)                     # (Bh, E, N)
        v = v.reshape(b * self.num_heads, -1, num).transpose(0, 2, 1)  # (Bh, N, E)

        # (Bh, N, E) x (Bh, E, N) -> (Bh, N, N)
        attn = ops.BatchMatMul()(q, k) * self.temperature
        attn = ops.Softmax(axis=-1)(attn)
        out = ops.BatchMatMul()(attn, v)  # (Bh, N, E)
        out = out.transpose(0, 2, 1).reshape(b, c, h, w)
        out = self.out_conv(out)

        final_out = ops.concat([out * self.gamma, x], axis=1)
        final_out = self.fusion_conv(final_out)
        return final_out


class AttentionChannel(nn.Cell):
    def __init__(self, dim=256, num_heads=8, bias=True):
        super(AttentionChannel, self).__init__()
        self.num_heads = num_heads
        self.temperature = ms.Parameter(default_input=init('ones', [num_heads, 1, 1], ms.float32))
        self.q_conv = nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=3, stride=1, padding=1, group=dim, has_bias=bias, pad_mode='pad', weight_init='HeUniform')
        self.k_conv = nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=3, stride=1, padding=1, group=dim, has_bias=bias, pad_mode='pad', weight_init='HeUniform')
        self.v_conv = nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=3, stride=1, padding=1, group=dim, has_bias=bias, pad_mode='pad', weight_init='HeUniform')
        self.out_conv = nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=1, has_bias=bias)
        self.gamma = ms.Parameter(default_input=init('zeros', [1], ms.float32))
        self.fusion_conv = nn.Conv2d(in_channels=2 * dim, out_channels=dim, kernel_size=1, has_bias=bias, weight_init='HeUniform')

    def construct(self, x):
        b, c, h, w = x.shape
        num = h * w

        q = self.q_conv(x)
        k = self.k_conv(x)
        v = self.v_conv(x)

        q = q.reshape(b * self.num_heads, -1, num)  # (Bh, E, N)
        k = k.reshape(b * self.num_heads, -1, num).transpose(0, 2, 1)  # (Bh, N, E)
        v = v.reshape(b * self.num_heads, -1, num)  # (Bh, E, N)

        # (Bh, E, N) x (Bh, N, E) -> (Bh, E, E)
        attn = ops.BatchMatMul()(q, k) * self.temperature
        attn = ops.Softmax(axis=-1)(attn)
        out = ops.BatchMatMul()(attn, v)  # (Bh, E, N)
        out = out.reshape(b, c, h, w)
        out = self.out_conv(out)

        final_out = ops.concat([out * self.gamma, x], axis=1)
        final_out = self.fusion_conv(final_out)
        return final_out

class resnet(nn.Cell):
    def __init__(self, arch='resnet18', input_dim=5, num_classes=2, pretrained=True):
        super(resnet, self).__init__()
        model = getattr(models, arch)(pretrained=pretrained)
        out_channels = model.backbone.conv1.features[0].out_channels
        model.backbone.conv1.features[0] = nn.Conv2d(in_channels=input_dim, out_channels=out_channels, kernel_size=7, stride=2, pad_mode='pad', padding=3, has_bias=False)

        if pretrained:
            url = model_urls[arch]
            path = os.path.join('./', 'LoadPretrainedModel')
            os.makedirs(path, exist_ok=True)
            DownLoad().download_url(url, path)
            param_dict = ms.load_checkpoint(os.path.join(path, os.path.basename(url)))
            src_param = param_dict['backbone.conv1.features.0.weight'].asnumpy()

            new_param = np.zeros((out_channels, 1, 7, 7), dtype=np.float32)
            for i in range(out_channels):
                new_param[i] = 0.299 * src_param[i, 0] + 0.587 * src_param[i, 1] + 0.114 * src_param[i, 2]
            new_param = new_param.repeat(input_dim - 3, axis=1)
            new_param = np.concatenate((src_param, new_param), axis=1)

            param_dict['backbone.conv1.features.0.weight'] = ms.Parameter(ms.Tensor(new_param), name='backbone.conv1.features.0.weight')
            ms.load_param_into_net(model, param_dict)

        model.head.dense = nn.Dense(in_channels=model.head.dense.in_channels, out_channels=num_classes, has_bias=True, weight_init='HeUniform')
        self.model = model

    def construct(self, x):
        return self.model(x)

if __name__ == '__main__':
    model = resnet(arch='resnet18', input_dim=5, num_classes=2)
    print(model)

    x = ms.Tensor(np.ones((2, 5, 256, 256), dtype=np.float32))
    out = model(x)
    print(out.shape)
    #print(model.model.backbone.conv1.features[0])
