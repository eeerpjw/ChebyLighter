# -*- coding: utf-8 -*-
from torchvision import transforms
import torch
import torch.nn as nn
import numpy as np
import kornia.augmentation as aug

class DataAugmentation(nn.Module):
    def __init__(self, image_size):
        super(DataAugmentation, self).__init__()
        self.image_size = image_size
        # if you found unrecognized params p, please update your kornia to version 0.4.1
        self.crop = aug.RandomCrop(p=1.0, size=[image_size,image_size],same_on_batch=False)
        self.hflip = aug.RandomHorizontalFlip(p=1.0,same_on_batch=False)
        self.vflip = aug.RandomVerticalFlip(p=1.0,same_on_batch=False)
        self.rot90 = aug.RandomRotation(p=1.0, degrees=90.0,same_on_batch=False)
        '''
        self.rot180 = aug.RandomRotation(p=1.0, degrees=180.0,same_on_batch=False)
        self.rot270 = aug.RandomRotation(p=1.0, degrees=270.0,same_on_batch=False)
        '''
    def forward(self, sample):
        """
        :param sample: dict，BCHW ，torch.tensor
        :return: dict，BCHW
        """
        keys = ["h", "l"]
        out = sample

        # random crop
        crop_params = self.crop.generate_parameters(out["h"].shape)
        for key in keys:
            out[key] = self.crop(out[key], crop_params)

        # hflip
        if np.random.random() > 0.5:
            hflip_params = self.hflip.generate_parameters(out["h"].shape)
            for key in keys:
                out[key] = self.hflip(out[key], hflip_params)
        # vflip
        if np.random.random() > 0.5:
            vflip_params = self.vflip.generate_parameters(out["h"].shape)
            for key in keys:
                out[key] = self.vflip(out[key], vflip_params)

        # rotate 90 degree
        if np.random.random() > 0.5:
            rot90_params = self.rot90.generate_parameters(out["h"].shape)
            for key in keys:
                out[key] = self.rot90(out[key], rot90_params)
        return out