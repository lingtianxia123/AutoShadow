# Copyright 2020-2021 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""image"""
import numbers
import numpy as np
import mindspore.common.dtype as mstype
from mindspore.common.tensor import Tensor
from mindspore.ops import operations as P
from mindspore.ops import functional as F
from mindspore.ops.primitive import constexpr
from mindspore._checkparam import Rel, Validator as validator
import mindspore.nn as nn


class ImageGradients(nn.Cell):
    r"""
    Returns two tensors, the first is along the height dimension and the second is along the width dimension.

    Assume an image shape is :math:`h*w`. The gradients along the height and the width are :math:`dy` and :math:`dx`,
    respectively.

    .. math::
        dy[i] = \begin{cases} image[i+1, :]-image[i, :], &if\ 0<=i<h-1 \cr
        0, &if\ i==h-1\end{cases}

        dx[i] = \begin{cases} image[:, i+1]-image[:, i], &if\ 0<=i<w-1 \cr
        0, &if\ i==w-1\end{cases}

    Inputs:
        - **images** (Tensor) - The input image data, with format 'NCHW'.

    Outputs:
        - **dy** (Tensor) - vertical image gradients, the same type and shape as input.
        - **dx** (Tensor) - horizontal image gradients, the same type and shape as input.

    Raises:
        ValueError: If length of shape of `images` is not equal to 4.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> net = nn.ImageGradients()
        >>> image = Tensor(np.array([[[[1, 2], [3, 4]]]]), dtype=mindspore.int32)
        >>> output = net(image)
        >>> print(output)
        (Tensor(shape=[1, 1, 2, 2], dtype=Int32, value=
        [[[[2, 2],
           [0, 0]]]]), Tensor(shape=[1, 1, 2, 2], dtype=Int32, value=
        [[[[1, 0],
           [1, 0]]]]))
    """
    def __init__(self):
        super(ImageGradients, self).__init__()

    def construct(self, images):
        check = _check_input_4d(F.shape(images), "images", self.cls_name)
        images = F.depend(images, check)
        batch_size, depth, height, width = P.Shape()(images)
        if height == 1:
            dy = P.Fill()(P.DType()(images), (batch_size, depth, 1, width), 0)
        else:
            dy = images[:, :, 1:, :] - images[:, :, :height - 1, :]
            dy_last = P.Fill()(P.DType()(images), (batch_size, depth, 1, width), 0)
            dy = P.Concat(2)((dy, dy_last))

        if width == 1:
            dx = P.Fill()(P.DType()(images), (batch_size, depth, height, 1), 0)
        else:
            dx = images[:, :, :, 1:] - images[:, :, :, :width - 1]
            dx_last = P.Fill()(P.DType()(images), (batch_size, depth, height, 1), 0)
            dx = P.Concat(3)((dx, dx_last))
        return dy, dx


def _convert_img_dtype_to_float32(img, max_val):
    """convert img dtype to float32"""
    # Usually max_val is 1.0 or 255, we will do the scaling if max_val > 1.
    # We will scale img pixel value if max_val > 1. and just cast otherwise.
    ret = F.cast(img, mstype.float32)
    max_val = F.scalar_cast(max_val, mstype.float32)
    if max_val > 1.:
        scale = 1. / max_val
        ret = ret * scale
    return ret


@constexpr
def _get_dtype_max(dtype):
    """get max of the dtype"""
    np_type = mstype.dtype_to_nptype(dtype)
    if issubclass(np_type, numbers.Integral):
        dtype_max = np.float64(np.iinfo(np_type).max)
    else:
        dtype_max = 1.0
    return dtype_max


@constexpr
def _check_input_4d(input_shape, param_name, func_name):
    if len(input_shape) != 4:
        raise ValueError(f"For '{func_name}', the dimension of '{param_name}' must be 4d, "
                         f"but got {len(input_shape)}.")
    return True


@constexpr
def _check_input_filter_size(input_shape, param_name, filter_size, func_name):
    _check_input_4d(input_shape, param_name, func_name)
    validator.check(param_name + " shape[2]", input_shape[2], "filter_size", filter_size, Rel.GE, func_name)
    validator.check(param_name + " shape[3]", input_shape[3], "filter_size", filter_size, Rel.GE, func_name)


@constexpr
def _check_input_dtype(input_dtype, param_name, allow_dtypes, cls_name):
    validator.check_type_name(param_name, input_dtype, allow_dtypes, cls_name)


def _conv2d(in_channels, out_channels, kernel_size, weight, stride=1, padding=0):
    return nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride,
                  weight_init=weight, padding=padding, pad_mode='same')


def _create_window(size, sigma):
    x_data, y_data = np.mgrid[-size // 2 + 1:size // 2 + 1, -size // 2 + 1:size // 2 + 1]
    x_data = np.expand_dims(x_data, axis=-1).astype(np.float32)
    x_data = np.expand_dims(x_data, axis=-1) ** 2
    y_data = np.expand_dims(y_data, axis=-1).astype(np.float32)
    y_data = np.expand_dims(y_data, axis=-1) ** 2
    sigma = 2 * sigma ** 2
    g = np.exp(-(x_data + y_data) / sigma)
    return np.transpose(g / np.sum(g), (2, 3, 0, 1))


def _split_img(x):
    _, c, _, _ = F.shape(x)
    img_split = P.Split(1, c)
    output = img_split(x)
    return output, c


def _compute_per_channel_loss(c1, c2, img1, img2, conv):
    """computes ssim index between img1 and img2 per single channel"""
    dot_img = img1 * img2
    mu1 = conv(img1)
    mu2 = conv(img2)
    mu1_sq = mu1 * mu1
    mu2_sq = mu2 * mu2
    mu1_mu2 = mu1 * mu2
    sigma1_tmp = conv(img1 * img1)
    sigma1_sq = sigma1_tmp - mu1_sq
    sigma2_tmp = conv(img2 * img2)
    sigma2_sq = sigma2_tmp - mu2_sq
    sigma12_tmp = conv(dot_img)
    sigma12 = sigma12_tmp - mu1_mu2
    a = (2 * mu1_mu2 + c1)
    b = (mu1_sq + mu2_sq + c1)
    v1 = 2 * sigma12 + c2
    v2 = sigma1_sq + sigma2_sq + c2
    ssim = (a * v1) / (b * v2)

    cs = v1 / v2
    return ssim, cs


def _compute_multi_channel_map(c1, c2, img1, img2, conv, concat, mean):
    """computes ssim index between img1 and img2 per color channel"""
    split_img1, c = _split_img(img1)
    split_img2, _ = _split_img(img2)
    multi_ssim = ()
    multi_cs = ()
    for i in range(c):
        ssim_per_channel, cs_per_channel = _compute_per_channel_loss(c1, c2, split_img1[i], split_img2[i], conv)
        multi_ssim += (ssim_per_channel,)
        multi_cs += (cs_per_channel,)

    multi_ssim = concat(multi_ssim)
    multi_cs = concat(multi_cs)
    # ssim = mean(multi_ssim, (2, 3))
    # cs = mean(multi_cs, (2, 3))
    return multi_ssim, multi_cs


class SSIM(nn.Cell):
    r"""
    Returns SSIM index between two images.

    Its implementation is based on Wang, Z., Bovik, A. C., Sheikh, H. R., & Simoncelli, E. P. (2004). `Image quality
    assessment: from error visibility to structural similarity <https://ieeexplore.ieee.org/document/1284395>`_.
    IEEE transactions on image processing.

    SSIM is a measure of the similarity of two pictures.
    Like PSNR, SSIM is often used as an evaluation of image quality. SSIM is a number between 0 and 1.The larger it is,
    the smaller the gap between the output image and the undistorted image, that is, the better the image quality.
    When the two images are exactly the same, SSIM=1.

    .. math::

        l(x,y)&=\frac{2\mu_x\mu_y+C_1}{\mu_x^2+\mu_y^2+C_1}, C_1=(K_1L)^2.\\
        c(x,y)&=\frac{2\sigma_x\sigma_y+C_2}{\sigma_x^2+\sigma_y^2+C_2}, C_2=(K_2L)^2.\\
        s(x,y)&=\frac{\sigma_{xy}+C_3}{\sigma_x\sigma_y+C_3}, C_3=C_2/2.\\
        SSIM(x,y)&=l*c*s\\&=\frac{(2\mu_x\mu_y+C_1)(2\sigma_{xy}+C_2}{(\mu_x^2+\mu_y^2+C_1)(\sigma_x^2+\sigma_y^2+C_2)}.

    Args:
        max_val (Union[int, float]): The dynamic range of the pixel values (255 for 8-bit grayscale images).
          Default: 1.0.
        filter_size (int): The size of the Gaussian filter. Default: 11. The value must be greater than or equal to 1.
        filter_sigma (float): The standard deviation of Gaussian kernel. Default: 1.5.
          The value must be greater than 0.
        k1 (float): The constant used to generate c1 in the luminance comparison function. Default: 0.01.
        k2 (float): The constant used to generate c2 in the contrast comparison function. Default: 0.03.

    Inputs:
        - **img1** (Tensor) - The first image batch with format 'NCHW'. It must be the same shape and dtype as img2.
        - **img2** (Tensor) - The second image batch with format 'NCHW'. It must be the same shape and dtype as img1.

    Outputs:
        Tensor, has the same dtype as img1. It is a 1-D tensor with shape N, where N is the batch num of img1.

    Raises:
        TypeError: If `max_val` is neither int nor float.
        TypeError: If `k1`, `k2` or `filter_sigma` is not a float.
        TypeError: If `filter_size` is not an int.
        ValueError: If `max_val` or `filter_sigma` is less than or equal to 0.
        ValueError: If `filter_size` is less than 0.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import numpy as np
        >>> import mindspore.nn as nn
        >>> from mindspore import Tensor
        >>> net = nn.SSIM()
        >>> img1 = Tensor(np.ones([1, 3, 16, 16]).astype(np.float32))
        >>> img2 = Tensor(np.ones([1, 3, 16, 16]).astype(np.float32))
        >>> output = net(img1, img2)
        >>> print(output)
        [1.]
    """
    def __init__(self, max_val=1.0, filter_size=11, filter_sigma=1.5, k1=0.01, k2=0.03):
        super(SSIM, self).__init__()
        validator.check_value_type('max_val', max_val, [int, float], self.cls_name)
        validator.check_number('max_val', max_val, 0.0, Rel.GT, self.cls_name)
        self.max_val = max_val
        self.filter_size = validator.check_int(filter_size, 1, Rel.GE, 'filter_size', self.cls_name)
        self.filter_sigma = validator.check_positive_float(filter_sigma, 'filter_sigma', self.cls_name)
        self.k1 = validator.check_value_type('k1', k1, [float], self.cls_name)
        self.k2 = validator.check_value_type('k2', k2, [float], self.cls_name)
        window = _create_window(filter_size, filter_sigma)
        self.conv = _conv2d(in_channels=1, out_channels=1, kernel_size=filter_size, weight=Tensor(window), stride=1, padding=0)
        self.conv.weight.requires_grad = False
        self.reduce_mean = P.ReduceMean()
        self.reduce_sum = P.ReduceSum()
        self.concat = P.Concat(axis=1)

    def construct(self, img1, img2, mask=None):
        _check_input_dtype(F.dtype(img1), "img1", [mstype.float32, mstype.float16], self.cls_name)
        _check_input_filter_size(F.shape(img1), "img1", self.filter_size, self.cls_name)
        P.SameTypeShape()(img1, img2)
        dtype_max_val = _get_dtype_max(F.dtype(img1))
        max_val = F.scalar_cast(self.max_val, F.dtype(img1))
        max_val = _convert_img_dtype_to_float32(max_val, dtype_max_val)
        img1 = _convert_img_dtype_to_float32(img1, dtype_max_val)
        img2 = _convert_img_dtype_to_float32(img2, dtype_max_val)

        c1 = (self.k1 * max_val) ** 2
        c2 = (self.k2 * max_val) ** 2

        ssim_map, _ = _compute_multi_channel_map(c1, c2, img1, img2, self.conv, self.concat, self.reduce_mean)

        if mask is not None:
            mask_sum = mask.sum()
            fg_ssim_map = ssim_map * mask
            fg_ssim_map_sum = fg_ssim_map.sum(3).sum(2)
            fg_ssim = fg_ssim_map_sum / mask_sum

            fg_ssim_mu = fg_ssim.mean()
            return fg_ssim_mu
            #return fg_ssim_mu.asnumpy()

        ssim_mu = ssim_map.mean()
        return ssim_mu
        #return ssim_mu.asnumpy()

