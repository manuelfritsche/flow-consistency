import os
import torch
import numpy as np
import scipy.misc as m
import re

from torch.utils import data

from semseg.utils import recursive_glob
from semseg.augmentations import *


class cityscapesLoader(data.Dataset):
    """cityscapesLoader
    https://www.cityscapes-dataset.com
    Data is derived from CityScapes, and can be downloaded from here:
    https://www.cityscapes-dataset.com/downloads/
    Many Thanks to @fvisin for the loader repo:
    https://github.com/fvisin/dataset_loaders/blob/master/dataset_loaders/images/cityscapes.py
    """

    colors = [  # [  0,   0,   0],
        [128, 64, 128],
        [244, 35, 232],
        [70, 70, 70],
        [102, 102, 156],
        [190, 153, 153],
        [153, 153, 153],
        [250, 170, 30],
        [220, 220, 0],
        [107, 142, 35],
        [152, 251, 152],
        [0, 130, 180],
        [220, 20, 60],
        [255, 0, 0],
        [0, 0, 142],
        [0, 0, 70],
        [0, 60, 100],
        [0, 80, 100],
        [0, 0, 230],
        [119, 11, 32],
    ]

    label_colours = dict(zip(range(19), colors))

    mean_rgb = {
        "pascal": [103.939, 116.779, 123.68],
        "cityscapes": [0.0, 0.0, 0.0],
    }  # pascal mean for PSPNet and ICNet pre-trained model

    def __init__(
        self,
        root,
        split="train",
        is_transform=False,
        img_size=(512, 1024),
        augmentations=None,
        img_norm=True,
        version="cityscapes",
        n_img_before=0,
        n_img_after=0,
        img_dist=1,
        frac_img=1.0,
        frac_lbl=1.0,
        get_flow=False,
        discard_flow_bottom=False,
    ):
        """__init__
        :param root:
        :param split:
        :param is_transform:
        :param img_size:
        :param augmentations
        """
        self.root = root
        self.split = split
        self.is_transform = is_transform
        self.augmentations = augmentations
        self.img_norm = img_norm
        self.n_classes = 19
        self.img_size = (
            img_size if isinstance(img_size, tuple) else (img_size, img_size)
        )
        self.mean = np.array(self.mean_rgb[version])
        self.files = {}

        self.images_base = os.path.join(self.root, "leftImg8bit", self.split)
        self.annotations_base = os.path.join(
            self.root, "gtFine", self.split
        )

        self.files[split] = recursive_glob(rootdir=self.images_base, suffix=".png")
        if frac_img < 1.0:
            n_img_to_keep = int(min(len(self.files[split]), np.ceil(frac_img * len(self.files[split]))))
            self.files[split] = self.files[split][:n_img_to_keep]
        if frac_lbl < 1.0:
            self.n_lbl_to_keep = int(min(len(self.files[split]), np.ceil(frac_lbl * len(self.files[split]))))
        else:
            self.n_lbl_to_keep = len(self.files[split])

        self.void_classes = [0, 1, 2, 3, 4, 5, 6, 9, 10, 14, 15, 16, 18, 29, 30, -1]
        self.valid_classes = [
            7,
            8,
            11,
            12,
            13,
            17,
            19,
            20,
            21,
            22,
            23,
            24,
            25,
            26,
            27,
            28,
            31,
            32,
            33,
        ]
        self.class_names = [
            "unlabelled",
            "road",
            "sidewalk",
            "building",
            "wall",
            "fence",
            "pole",
            "traffic_light",
            "traffic_sign",
            "vegetation",
            "terrain",
            "sky",
            "person",
            "rider",
            "car",
            "truck",
            "bus",
            "train",
            "motorcycle",
            "bicycle",
        ]

        self.ignore_index = 250
        self.class_map = dict(zip(self.valid_classes, range(19)))

        self.n_img_before = n_img_before
        self.n_img_after = n_img_after
        self.img_dist = img_dist
        self.get_flow = get_flow
        self.discard_flow_bottom = discard_flow_bottom

        if not self.files[split]:
            raise Exception(
                "No files for split=[%s] found in %s" % (split, self.images_base)
            )

        print("Found %d %s images" % (len(self.files[split]), split))

    def __len__(self):
        """__len__"""
        return len(self.files[self.split])

    def __getitem__(self, index):
        """__getitem__
        :param index:
        """
        img_path = self.files[self.split][index].rstrip()
        flow_path = re.sub('.png$', '.npy', img_path)
        flow_path = re.sub('leftImg8bit', 'flow', flow_path)
        img_path_seq = re.subn('leftImg8bit', 'leftImg8bit_sequence', img_path, count=1)[0]
        lbl_path = os.path.join(
            self.annotations_base,
            img_path.split(os.sep)[-2],
            os.path.basename(img_path)[:-15] + "gtFine_labelIds.png",
        )

        img = m.imread(img_path)
        img = np.array(img, dtype=np.uint8)

        lbl = m.imread(lbl_path)
        lbl = self.encode_segmap(np.array(lbl, dtype=np.uint8))
        if self.n_img_before > 0 or self.n_img_after > 0:
            original_lbl = np.copy(lbl)

        if self.augmentations is not None:
            img, lbl = self.augmentations(img, lbl)

        if self.is_transform:
            img, lbl = self.transform(img, lbl)

        # concatenate the images right before the current one
        if self.n_img_before > 0:
            for ind in range(self.n_img_before):
                img_path_add = self.neighbour_img(img_path_seq, -self.img_dist * (ind+1))
                add_img = m.imread(img_path_add)
                add_img = np.array(add_img, dtype=np.uint8)
                if self.augmentations is not None:
                    add_img, _ = self.augmentations(add_img, original_lbl)
                if self.is_transform:
                    add_img, _ = self.transform(add_img)
                img = np.concatenate((add_img, img), axis=0)

        # concatenate the images right after the current one
        if self.n_img_after > 0:
            for ind in range(self.n_img_after):
                img_path_add = self.neighbour_img(img_path_seq, self.img_dist * (ind+1))
                add_img = m.imread(img_path_add)
                add_img = np.array(add_img, dtype=np.uint8)
                if self.augmentations is not None:
                    add_img, _ = self.augmentations(add_img, original_lbl)
                if self.is_transform:
                    add_img, _ = self.transform(add_img)
                img = np.concatenate((img, add_img), axis=0)

        if self.get_flow:
            flow = np.load(flow_path)
            if self.discard_flow_bottom:
                height = np.shape(flow)[0]
                flow[int(height*0.75):, ...] = 0
            # make sure flow is in the same shape as img
            if np.shape(flow)[0] != self.img_size[0]:
                print("Warning flow was resized! ")
                h, w, c = np.shape(flow)
                flow_exp = np.zeros([h, w, c+1])
                flow_exp[..., 1:] = flow
                flow_exp = m.imresize(flow_exp, (self.img_size[0], self.img_size[1]))
                flow = flow_exp[..., 1:]
            flow = np.moveaxis(flow, 2, 0)
            if index >= self.n_lbl_to_keep:
                lbl = torch.zeros_like(lbl) + 255
            return img, lbl, flow.astype(dtype=np.float32)
        else:
            return img, lbl, torch.tensor(-1)

    def transform(self, img, lbl=None):
        """transform
        :param img:
        :param lbl:
        """
        img = m.imresize(
            img, (self.img_size[0], self.img_size[1])
        )  # uint8 with RGB mode
        img = img[:, :, ::-1]  # RGB -> BGR
        img = img.astype(np.float64)
        img -= self.mean
        if self.img_norm:
            # Resize scales images from 0 to 255, thus we need
            # to divide by 255.0
            img = img.astype(float) / 255.0
        # NHWC -> NCHW
        img = img.transpose(2, 0, 1)

        if lbl is not None:
            classes = np.unique(lbl)
            lbl = lbl.astype(float)
            lbl = m.imresize(lbl, (self.img_size[0], self.img_size[1]), "nearest", mode="F")
            lbl = lbl.astype(int)

            if not np.all(classes == np.unique(lbl)):
                print("WARN: resizing labels yielded fewer classes")

            if not np.all(np.unique(lbl[lbl != self.ignore_index]) < self.n_classes):
                print("after det", classes, np.unique(lbl))
                raise ValueError("Segmentation map contained invalid class values")

            lbl = torch.from_numpy(lbl).long()
        img = torch.from_numpy(img).float()
        return img, lbl

    def decode_segmap(self, temp):
        r = temp.copy()
        g = temp.copy()
        b = temp.copy()
        for l in range(0, self.n_classes):
            r[temp == l] = self.label_colours[l][0]
            g[temp == l] = self.label_colours[l][1]
            b[temp == l] = self.label_colours[l][2]

        rgb = np.zeros((temp.shape[0], temp.shape[1], 3))
        rgb[:, :, 0] = r / 255.0
        rgb[:, :, 1] = g / 255.0
        rgb[:, :, 2] = b / 255.0
        return rgb

    def encode_segmap(self, mask):
        # Put all void classes to zero
        for _voidc in self.void_classes:
            mask[mask == _voidc] = self.ignore_index
        for _validc in self.valid_classes:
            mask[mask == _validc] = self.class_map[_validc]
        return mask

    def neighbour_img(self, path, rel_location):
        # change number in image path
        number = re.search('_[0-9]+_([0-9]+)_', path).group(1)
        length = len(number)
        number = int(number) + rel_location
        number = str(number).zfill(length)
        return re.sub(r'([0-9]+)_[0-9]+_', r'\1_' + number + '_', path)

if __name__ == "__main__":
    import torchvision
    import matplotlib.pyplot as plt

    augmentations = Compose([Scale(2048), RandomRotate(10), RandomHorizontallyFlip(0.5)])

    local_path = "/datasets01/cityscapes/112817/"
    dst = cityscapesLoader(local_path, is_transform=True, augmentations=augmentations)
    bs = 4
    trainloader = data.DataLoader(dst, batch_size=bs, num_workers=0)
    for i, data in enumerate(trainloader):
        imgs, labels = data
        import pdb;pdb.set_trace()
        imgs = imgs.numpy()[:, ::-1, :, :]
        imgs = np.transpose(imgs, [0, 2, 3, 1])
        f, axarr = plt.subplots(bs, 2)
        for j in range(bs):
            axarr[j][0].imshow(imgs[j])
            axarr[j][1].imshow(dst.decode_segmap(labels.numpy()[j]))
        plt.show()
        a = raw_input()
        if a == "ex":
            break
        else:
            plt.close()