import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops
import numpy as np
from src.networks import DoubleConv, Up, Down, resnet
from src.HAUNet import HAUNet
from src.ssim import SSIM
import src.util as util
import math
from skimage.measure import compare_mse
import os

ms.set_context(mode=ms.PYNATIVE_MODE)
#os.environ['CUDA_VISIBLE_DEVICES'] = '1'


class IFNet_Param(nn.Cell):
    def __init__(self):
        super(IFNet_Param, self).__init__()
        # param
        self.model_param = resnet(arch='resnet18', input_dim=5, num_classes=6)

    def construct(self, deshadow_img, fg_instance, pred_mask):
        input_param = ops.concat([deshadow_img, fg_instance, pred_mask * 2 - 1], axis=1)  # [-1, 1]
        pred_param = self.model_param(input_param)
        return pred_param

class SetCriterion(nn.Cell):
    def __init__(self, weight_dict):
        super(SetCriterion, self).__init__()
        self.weight_dict = weight_dict
        self.l1_loss = ms.nn.loss.L1Loss()
        self.mse_loss = ms.nn.loss.MSELoss()

    def construct(self, pred_param, shadow_param):
        # param
        src_avg = pred_param[..., [0, 2, 4]]
        src_min = pred_param[..., [1, 3, 5]]

        tar_avg = shadow_param[..., [0, 2, 4]]
        tar_min = shadow_param[..., [1, 3, 5]]

        loss_avg = self.l1_loss(src_avg, tar_avg)
        loss_min = self.l1_loss(src_min, tar_min)

        loss_dict = {
            "loss_avg": loss_avg,
            "loss_min": loss_min,
        }
        loss = sum(loss_dict[k] * self.weight_dict[k] for k in loss_dict.keys() if k in self.weight_dict)
        return loss

class ShadowMetrics(nn.Metric):
    def __init__(self):
        super(ShadowMetrics, self).__init__()
        self.clear()
        self.l1_loss = ms.nn.loss.L1Loss()


    def clear(self):
        self._l1_min_sum = 0
        self._l1_avg_sum = 0
        self._samples_num = 0

    @nn.rearrange_inputs
    def update(self, pred_param, shadow_param):
        for i in range(pred_param.shape[0]):
            src_avg = pred_param[..., [0, 2, 4]]
            src_min = pred_param[..., [1, 3, 5]]

            tar_avg = shadow_param[..., [0, 2, 4]]
            tar_min = shadow_param[..., [1, 3, 5]]

            loss_min = self.l1_loss(src_min, tar_min)
            self._l1_min_sum += loss_min

            loss_avg = self.l1_loss(src_avg, tar_avg)
            self._l1_avg_sum += loss_avg

        self._samples_num += pred_param.shape[0]

    def eval(self):
        avg_loss = self._l1_avg_sum / self._samples_num
        min_loss = self._l1_min_sum / self._samples_num
        res = {"avg": avg_loss, "min": min_loss}
        return res

def build_model(args):
    model = IFNet_Param()

    weight_dict = {"loss_avg": args.avg_weight,
                   "loss_min": args.min_weight,
                   "loss_fuse": args.fuse_weight}
    criterion = SetCriterion(weight_dict=weight_dict)
    evaluator = ShadowMetrics()
    return model, criterion, evaluator


if __name__ == '__main__':
    model_mask = HAUNet()
    model = IFNet_Param()
    print(model)
    deshadow_img = ms.Tensor(np.ones((2, 3, 256, 256), dtype=np.float32))
    fg_instance = ms.Tensor(np.ones((2, 1, 256, 256), dtype=np.float32))
    pre_mask = model_mask(deshadow_img, fg_instance)
    print(pre_mask.shape)
    pred_param = model(deshadow_img, fg_instance, pre_mask)
    print(pred_param.shape)