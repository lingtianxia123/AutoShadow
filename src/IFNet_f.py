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

class IFNet_fuse(nn.Cell):
    def __init__(self):
        super(IFNet_fuse, self).__init__()
        # confuse
        self.fuse_num = 5
        self.kernel_size = 3

        hidden_dim = 32
        self.bilinear = True
        factor = 2 if self.bilinear else 1
        self.inc = DoubleConv(1 + 3 + self.fuse_num * 3, hidden_dim, norm_num_groups=hidden_dim // 2)
        self.down1 = Down(hidden_dim, hidden_dim * 2)
        self.down2 = Down(hidden_dim * 2, hidden_dim * 4)
        self.down3 = Down(hidden_dim * 4, hidden_dim * 8)
        self.down4 = Down(hidden_dim * 8, hidden_dim * 16 // factor)
        self.up1 = Up(hidden_dim * 16, hidden_dim * 8 // factor, self.bilinear)
        self.up2 = Up(hidden_dim * 8, hidden_dim * 4 // factor, self.bilinear)
        self.up3 = Up(hidden_dim * 4, hidden_dim * 2 // factor, self.bilinear, norm_num_groups=hidden_dim // 2)
        self.up4 = Up(hidden_dim * 2, hidden_dim, self.bilinear, norm_num_groups=hidden_dim // 2)
        self.outc = nn.Conv2d(hidden_dim, ((1 + self.fuse_num) * 3) * 3 * self.kernel_size * self.kernel_size, kernel_size=1, has_bias=True, weight_init='HeUniform')

        self.unfold = nn.Unfold(ksizes=[1, self.kernel_size, self.kernel_size, 1], strides=[1, 1, 1, 1], padding='same', rates=[1, 1, 1, 1])

    def construct(self, deshadow_img, fg_instance, pred_mask, pred_param):
        B , _, H, W = deshadow_img.shape
        # comp
        avg_scale = pred_param[..., [0, 2, 4]].copy()
        min_scale = pred_param[..., [1, 3, 5]].copy()
        avg_scale = avg_scale.reshape(avg_scale.shape[0], 3, 1, 1)
        min_scale = min_scale.reshape(min_scale.shape[0], 3, 1, 1)

        deshadow_img_01 = deshadow_img / 2 + 0.5  # [0, 1]
        dark_min_img = deshadow_img_01 * min_scale
        dark_avg_img = deshadow_img_01 * avg_scale
        dark_img_list = ops.concat([dark_min_img, dark_avg_img, deshadow_img_01], axis=1)
        # low
        num_scale = self.fuse_num // 2
        temp_img = (dark_avg_img - dark_min_img) / num_scale
        for i in range(num_scale - 1):
            dark_img_list = ops.concat([dark_img_list, dark_min_img + temp_img * (i + 1)], axis=1)
        # high
        num_scale = self.fuse_num // 2 + 1
        temp_img = (deshadow_img_01 - dark_avg_img) / num_scale
        for i in range(num_scale - 1):
            dark_img_list = ops.concat([dark_img_list, dark_avg_img + temp_img * (i + 1)], axis=1)

        out = dark_img_list  # [0, 1]
        out_matrix = self.unfold(out)

        input_confuse = ops.concat([dark_img_list, pred_mask], axis=1)  # [0, 1]
        input_confuse = input_confuse * 2 - 1
        # encoder
        f1 = self.inc(input_confuse)
        f2 = self.down1(f1)
        f3 = self.down2(f2)
        f4 = self.down3(f3)
        f5 = self.down4(f4)
        # decoder
        x = self.up1(f5, f4)
        x = self.up2(x, f3)
        x = self.up3(x, f2)
        x = self.up4(x, f1)
        kernel = self.outc(x)

        fuse_img = self.confuse(out_matrix, kernel, deshadow_img_01, self.fuse_num + 1, self.kernel_size)
        fuse_img = fuse_img * 2 - 1
        return fuse_img


        # kernel = kernel.reshape(B, 3, -1, H * W)
        # kernel = ops.Softmax(axis=2)(kernel)
        # kernel = kernel.transpose(0, 3, 1, 2)
        # kernel = kernel.reshape(-1, kernel.shape[2], kernel.shape[3])
        #
        # out_matrix = out_matrix.reshape(B, -1, H * W)
        # out_matrix = out_matrix.transpose(0, 2, 1)
        # out_matrix = out_matrix.reshape(-1, out_matrix.shape[2], 1)
        #
        # img = ops.BatchMatMul()(kernel, out_matrix)
        # img = img.reshape(B, -1, img.shape[1])
        # img = img.transpose(0, 2, 1).reshape(B, -1, H, W)
        # return img



    def confuse(self, matrix, kernel, image, img_num, k_size):
        b, c, h, w = image.shape
        output = []
        matrix = matrix.reshape(matrix.shape[0], matrix.shape[1], -1)
        for i in range(b):
            feature = matrix[i, ...]  # ((1 + n) * 3) * ks * ks, L
            weight = kernel[i, ...]  # ((1 + n) * 3) * 3 * ks * ks, H, W
            feature = feature.expand_dims(axis=0)  # 1, C, L
            weight = weight.view((3, img_num * 3 * k_size * k_size, h * w))
            weight = ops.Softmax(axis=1)(weight)
            iout = feature * weight  # (3, C, L)
            iout = iout.sum(axis=1, keepdims=False)
            iout = iout.view((1, 3, h, w))
            output.append(iout)
        final = ops.concat(output, axis=0)
        return final

class SetCriterion(nn.Cell):
    def __init__(self, weight_dict):
        super(SetCriterion, self).__init__()
        self.weight_dict = weight_dict
        self.l1_loss = ms.nn.loss.L1Loss()
        self.mse_loss = ms.nn.loss.MSELoss()

    def construct(self, fuse_img, shadow_img):
        # fuse
        loss_fuse = self.mse_loss(fuse_img, shadow_img)

        #loss_dict = {"loss_fuse": loss_fuse,}
        return loss_fuse

class ShadowMetrics(nn.Metric):
    def __init__(self):
        super(ShadowMetrics, self).__init__()
        self.clear()
        self.ssim_net = SSIM()

    def clear(self):
        self._iou_sum = 0
        self._samples_num = 0
        self._g_rmse_sum = 0
        self._l_rmse_sum = 0
        self._g_ssim_sum = 0
        self._l_ssim_sum = 0

    @nn.rearrange_inputs
    def update(self, pred_mask, fuse_img, fg_shadow, shadow_img):
        fg_shadow[fg_shadow > 0.0] = 1.0
        fg_shadow[fg_shadow < 1.0] = 0.0
        for i in range(pred_mask.shape[0]):
            # iou
            tar_mask = fg_shadow[i].asnumpy()  # [0, 1]
            out_mask = pred_mask[i].asnumpy()
            out_mask[out_mask > 0.5] = 1.0
            out_mask[out_mask < 1.0] = 0.0
            union = out_mask + tar_mask
            union[union > 1.0] = 1.0
            inter = out_mask * tar_mask
            IoU = np.sum(inter) / np.sum(union)
            self._iou_sum += IoU

            # ssim
            tar_tensor = (shadow_img[i:i + 1, :, :, :] / 2 + 0.5) * 255
            out_tensor = (fuse_img[i:i + 1, :, :, :] / 2 + 0.5) * 255
            mask_tensor = fg_shadow[i:i + 1, :, :, :]   # [0, 1]
            self._g_ssim_sum += self.ssim_net(tar_tensor, out_tensor)
            self._l_ssim_sum += self.ssim_net(tar_tensor, out_tensor, mask_tensor)

            # rmse
            tar_img = util.tensor2img(shadow_img[i:i + 1, :, :, :]).astype(np.float32)
            out_img = util.tensor2img(fuse_img[i:i + 1, :, :, :]).astype(np.float32)
            mask_img = util.tensor2img(fg_shadow[i:i + 1, :, :, :] * 2 - 1)   # [0, 255]
            self._g_rmse_sum += math.sqrt(compare_mse(tar_img, out_img))
            self._l_rmse_sum += math.sqrt(compare_mse(tar_img * (mask_img / 255), out_img * (mask_img / 255)) * 256 * 256 / np.sum(mask_img / 255))

        self._samples_num += pred_mask.shape[0]

    def eval(self):
        IoU = self._iou_sum / self._samples_num
        GRMSE = self._g_rmse_sum / self._samples_num
        LRMSE = self._l_rmse_sum / self._samples_num
        GSSIM = self._g_ssim_sum / self._samples_num
        LSSIM = self._l_ssim_sum / self._samples_num

        res = {"IoU": IoU, "GRMSE": GRMSE, "LRMSE": LRMSE, "GSSIM": GSSIM, "LSSIM": LSSIM}
        return res

def build_model(args):
    model = IFNet_fuse()

    weight_dict = {"loss_avg": args.avg_weight,
                   "loss_min": args.min_weight,
                   "loss_fuse": args.fuse_weight}
    criterion = SetCriterion(weight_dict=weight_dict)
    evaluator = ShadowMetrics()
    return model, criterion, evaluator


# if __name__ == '__main__':
#     model_mask = HAUNet()
#     model = IFNet()
#     print(model)
#     deshadow_img = ms.Tensor(np.ones((2, 3, 256, 256), dtype=np.float32))
#     fg_instance = ms.Tensor(np.ones((2, 1, 256, 256), dtype=np.float32))
#     pre_mask = model_mask(deshadow_img, fg_instance)
#     print(pre_mask.shape)
#     fuse_img, pred_param = model(deshadow_img, fg_instance, pre_mask)
#     print(fuse_img.shape)