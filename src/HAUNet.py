import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops
import numpy as np
from src.networks import DoubleConv, Up, Down, Attention, AttentionChannel
import math

class HAUNet(nn.Cell):
    def __init__(self, input_dim=4):
        super(HAUNet, self).__init__()

        hidden_dim = 32
        self.bilinear = True
        self.inc = DoubleConv(input_dim, hidden_dim, norm_num_groups=hidden_dim // 2)
        self.down1 = Down(hidden_dim, hidden_dim * 2)
        self.down2 = Down(hidden_dim * 2, hidden_dim * 4)
        self.down3 = Down(hidden_dim * 4, hidden_dim * 8)
        factor = 2 if self.bilinear else 1
        self.down4 = Down(hidden_dim * 8, hidden_dim * 16 // factor)
        self.up1 = Up(hidden_dim * 16, hidden_dim * 8 // factor, self.bilinear)
        self.up2 = Up(hidden_dim * 8, hidden_dim * 4 // factor, self.bilinear)
        self.up3 = Up(hidden_dim * 4, hidden_dim * 2 // factor, self.bilinear, norm_num_groups=hidden_dim // 2)
        self.up4 = Up(hidden_dim * 2, hidden_dim, self.bilinear, norm_num_groups=hidden_dim // 2)
        self.outc = nn.Conv2d(hidden_dim, 1, kernel_size=1, has_bias=True, weight_init='HeUniform')

        self.att1 = nn.SequentialCell(
            Attention(dim=hidden_dim * 16 // factor, num_heads=1),
            AttentionChannel(dim=hidden_dim * 16 // factor, num_heads=1)
        )
        self.att2 = nn.SequentialCell(
            Attention(dim=hidden_dim * 16 // factor, num_heads=1),
            AttentionChannel(dim=hidden_dim * 16 // factor, num_heads=1)
        )
        self.att3 = nn.SequentialCell(
            Attention(dim=hidden_dim * 16 // factor, num_heads=1),
            AttentionChannel(dim=hidden_dim * 16 // factor, num_heads=1)
        )

    def construct(self, deshadow_img, fg_instance):
        input_mask = ops.concat([deshadow_img, fg_instance], axis=1)
        # encoder
        f1 = self.inc(input_mask)
        f2 = self.down1(f1)
        f3 = self.down2(f2)
        f4 = self.down3(f3)
        f5 = self.down4(f4)

        # attention
        f5 = self.att1(f5)
        f5 = self.att2(f5)
        f5 = self.att3(f5)

        # decoder
        x = self.up1(f5, f4)
        x = self.up2(x, f3)
        x = self.up3(x, f2)
        x = self.up4(x, f1)
        pre_alpha = self.outc(x)
        pre_alpha = ops.Sigmoid()(pre_alpha)  # [0, 1]
        return pre_alpha


class SetCriterion(nn.Cell):
    def __init__(self, weight_dict):
        super(SetCriterion, self).__init__()
        self.weight_dict = weight_dict
        self.BCE_loss = ops.BinaryCrossEntropy(reduction='none')

    def mask_loss(self, inputs, targets, num_masks):
        weight = ops.Ones()(inputs.shape, ms.float32)
        loss = self.BCE_loss(inputs, targets, weight)
        return loss.mean(1).sum() / num_masks

    def dice_loss(self, inputs, targets, num_masks):
        numerator = 2 * (inputs * targets).sum(axis=-1)
        denominator = inputs.sum(axis=-1) + targets.sum(axis=-1)
        loss = 1 - (numerator + 1) / (denominator + 1)
        return loss.sum() / num_masks

    def construct(self, output_masks, target_masks):
        target_masks[target_masks > 0] = 1
        target_masks[target_masks < 0] = 0

        output_masks = output_masks.reshape(output_masks.shape[0], -1)
        target_masks = target_masks.reshape(target_masks.shape[0], -1)
        loss_mask = self.mask_loss(output_masks, target_masks, target_masks.shape[0])
        loss_dice = self.dice_loss(output_masks, target_masks, target_masks.shape[0])
        loss_dict = {
            "loss_mask": loss_mask,
            "loss_dice": loss_dice,
        }
        loss = sum(loss_dict[k] * self.weight_dict[k] for k in loss_dict.keys() if k in self.weight_dict)
        return loss



class IoUMetrics(nn.Metric):
    def __init__(self):
        super(IoUMetrics, self).__init__()
        self.clear()

    def clear(self):
        self._iou_sum = 0
        self._samples_num = 0

    @nn.rearrange_inputs
    def update(self, pre_mask, fg_shadow):
        fg_shadow[fg_shadow > 0.0] = 1.0
        fg_shadow[fg_shadow < 1.0] = 0.0
        for i in range(pre_mask.shape[0]):
            # iou
            tar_mask = fg_shadow[i].asnumpy()  # [0, 1]
            out_mask = pre_mask[i].asnumpy()
            out_mask[out_mask > 0.5] = 1.0
            out_mask[out_mask < 1.0] = 0.0
            union = out_mask + tar_mask
            union[union > 1.0] = 1.0
            inter = out_mask * tar_mask
            IoU = np.sum(inter) / np.sum(union)
            self._iou_sum += IoU

        self._samples_num += pre_mask.shape[0]

    def eval(self):
        IoU = self._iou_sum / self._samples_num
        res = {"IoU": IoU}
        return res


def build_model(args):
    model = HAUNet(input_dim=4)

    weight_dict = {"loss_mask": args.mask_weight,
                   "loss_dice": args.dice_weight}
    criterion = SetCriterion(weight_dict=weight_dict)
    evaluator = IoUMetrics()
    return model, criterion, evaluator

if __name__ == '__main__':
    model = HAUNet()
    print(model)
    x = ms.Tensor(np.ones((2, 4, 256, 256), dtype=np.float32))
    out = model(x)
    print(out.shape)