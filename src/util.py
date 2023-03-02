import numpy as np
import mindspore as ms


def tensor2img(input_tensor, type=np.uint8):
    if len(input_tensor.shape) < 3: return None

    if not isinstance(input_tensor, ms.Tensor):
        return input_tensor

    image_numpy = input_tensor[0].asnumpy()
    image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0
    image_numpy[image_numpy < 0] = 0
    image_numpy[image_numpy > 255] = 255
    return image_numpy.astype(type)