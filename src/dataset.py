import os
from pathlib import Path
import numpy as np
import mindspore.dataset as ds
from PIL import Image
import cv2

class DESOBA_Dataset:
    def __init__(self, root=''):
        self.root = root

        self.birdy_deshadoweds = []
        self.birdy_all_deshadoweds = []
        self.birdy_shadoweds = []
        self.birdy_fg_instances = []
        self.birdy_fg_shadows = []
        self.birdy_bg_instances = []
        self.birdy_bg_shadows = []
        self.birdy_shadow_params = []
        self.birdy_imlists = []
        for imname in os.listdir(os.path.join(self.root, 'deshadoweds')):
            name = imname[:-len(imname.split('-')[-1]) - 1]
            self.birdy_deshadoweds.append(os.path.join(self.root, 'deshadoweds', imname))
            self.birdy_all_deshadoweds.append(os.path.join(self.root, 'all_deshadoweds', name + '.png'))
            self.birdy_shadoweds.append(os.path.join(self.root, 'shadoweds', name + '.png'))
            self.birdy_fg_instances.append(os.path.join(self.root, 'fg_instance', imname))
            self.birdy_fg_shadows.append(os.path.join(self.root, 'fg_shadow', imname))
            self.birdy_bg_instances.append(os.path.join(self.root, 'bg_instance', imname))
            self.birdy_bg_shadows.append(os.path.join(self.root, 'bg_shadow', imname))
            self.birdy_shadow_params.append(os.path.join(self.root, 'SOBA_params_am', name + '.png.txt'))
            self.birdy_imlists.append(imname)

        self.data_size = len(self.birdy_deshadoweds)

        print('shadow_param_fine_am datasize', self.data_size)

    @property
    def column_names(self):
        column_names = ['shadow_img', 'deshadow_img', 'fg_instance', 'fg_shadow', 'shadow_param']
        return column_names

    def __getitem__(self, index):
        shadow_img = Image.open(self.birdy_shadoweds[index]).convert('RGB')
        deshadow_img = Image.open(self.birdy_deshadoweds[index]).convert('RGB')
        fg_instance = Image.open(self.birdy_fg_instances[index]).convert('L')
        fg_shadow = Image.open(self.birdy_fg_shadows[index]).convert('L')

        shadow_img = np.array(shadow_img)
        deshadow_img = np.array(deshadow_img)
        fg_instance = np.array(fg_instance)
        fg_shadow = np.array(fg_shadow)

        # if the shadow area is too small, let's not change anything:
        sparam = open(self.birdy_shadow_params[index])
        line = sparam.read()
        shadow_param = np.asarray([float(i) for i in line.split(" ") if i.strip()], dtype=np.float32)
        shadow_param = shadow_param[0:6]
        if np.sum(fg_shadow > 0) < 30:
            shadow_param[:] = 1

        return shadow_img, deshadow_img, fg_instance, fg_shadow, shadow_param

    def __len__(self):
        return self.data_size


def preprocess_img(shadow_img, deshadow_img, fg_instance, fg_shadow, shadow_param, img_size):
    shadow_img = cv2.resize(shadow_img, (img_size, img_size), interpolation=cv2.INTER_NEAREST)
    deshadow_img = cv2.resize(deshadow_img, (img_size, img_size), interpolation=cv2.INTER_NEAREST)
    fg_instance = cv2.resize(fg_instance, (img_size, img_size), interpolation=cv2.INTER_NEAREST)
    fg_shadow = cv2.resize(fg_shadow, (img_size, img_size), interpolation=cv2.INTER_NEAREST)

    shadow_img = (shadow_img.astype(np.float32) - 127.5) / 127.5   # [-1, 1]
    deshadow_img = (deshadow_img.astype(np.float32) - 127.5) / 127.5   # [-1, 1]
    fg_instance = (fg_instance.astype(np.float32) - 127.5) / 127.5   # [-1, 1]
    fg_shadow = (fg_shadow.astype(np.float32) - 127.5) / 127.5   # [-1, 1]

    shadow_img = shadow_img.transpose(2, 0, 1)
    deshadow_img = deshadow_img.transpose(2, 0, 1)
    fg_instance = np.expand_dims(fg_instance, axis=0)
    fg_shadow = np.expand_dims(fg_shadow, axis=0)
    return shadow_img, deshadow_img, fg_instance, fg_shadow, shadow_param


def build_dataset(image_set, args):
    root = Path(args.data_path)
    assert root.exists(), f'provided data path {root} does not exist'

    PATHS = {
        "train": (root / "train"),
        "bos": (root / "bos"),
        "bosfree": (root / "bosfree"),
    }
    img_folder = PATHS[image_set]
    dataset_generator = DESOBA_Dataset(root=img_folder)

    if image_set == 'train':
        shuffle = True
        is_train = True
    else:
        shuffle = False
        is_train = False

    dataset = ds.GeneratorDataset(dataset_generator, column_names=dataset_generator.column_names, shuffle=shuffle, num_parallel_workers=args.num_workers)
    compose_map_func = (lambda shadow_img, deshadow_img, fg_instance, fg_shadow, shadow_param:
                        preprocess_img(shadow_img, deshadow_img, fg_instance, fg_shadow, shadow_param, args.image_size))
    dataset = dataset.map(operations=compose_map_func, input_columns=dataset_generator.column_names,
                          output_columns=dataset_generator.column_names, column_order=dataset_generator.column_names,
                          num_parallel_workers=args.num_workers)
    dataset = dataset.batch(args.batch_size, drop_remainder=is_train, num_parallel_workers=args.num_workers)
    return dataset

if __name__ == '__main__':
    dataset_generator = DESOBA_Dataset(root='D:/Dataset/ShadowGenerate/DESOBA/train')
    dataset = ds.GeneratorDataset(dataset_generator, column_names=dataset_generator.column_names, shuffle=False)
    img_size = 256
    compose_map_func = (lambda shadow_img, deshadow_img, fg_instance, fg_shadow, shadow_param:
                        preprocess_img(shadow_img, deshadow_img, fg_instance, fg_shadow, shadow_param, img_size))
    dataset = dataset.map(operations=compose_map_func, input_columns=dataset_generator.column_names,
                          output_columns=dataset_generator.column_names, column_order=dataset_generator.column_names,
                          num_parallel_workers=1)
    dataset = dataset.batch(batch_size=10)

    for data in dataset.create_dict_iterator():
        shadow_img = data['shadow_img']
        deshadow_img = data['deshadow_img']
        fg_instance = data['fg_instance']
        fg_shadow = data['fg_shadow']
        shadow_param = data['shadow_param']

        print(type(shadow_img))
        print(shadow_img.shape, deshadow_img.shape, fg_instance.shape, fg_shadow.shape, shadow_param.shape)
        quit(0)
        shadow_img = shadow_img.asnumpy()
        shadow_img = (shadow_img + 1) * 127.5
        shadow_img = shadow_img.transpose(1, 2, 0)
        shadow_img = shadow_img.astype(np.uint8)
        shadow_img = shadow_img[:, :, [2, 1, 0]]

        deshadow_img = deshadow_img.asnumpy()
        deshadow_img = (deshadow_img + 1) * 127.5
        deshadow_img = deshadow_img.transpose(1, 2, 0)
        deshadow_img = deshadow_img.astype(np.uint8)
        deshadow_img = deshadow_img[:, :, [2, 1, 0]]

        fg_instance = fg_instance.asnumpy()
        fg_instance = (fg_instance + 1) * 127.5
        fg_instance = fg_instance.transpose(1, 2, 0)
        fg_instance = fg_instance.astype(np.uint8)

        fg_shadow = fg_shadow.asnumpy()
        fg_shadow = (fg_shadow + 1) * 127.5
        fg_shadow = fg_shadow.transpose(1, 2, 0)
        fg_shadow = fg_shadow.astype(np.uint8)


        cv2.imshow("shadow_img", shadow_img)
        cv2.imshow("deshadow_img", deshadow_img)
        cv2.imshow("fg_instance", fg_instance)
        cv2.imshow("fg_shadow", fg_shadow)
        cv2.waitKey(0)

        continue
