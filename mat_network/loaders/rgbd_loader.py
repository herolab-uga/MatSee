#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 8 20:45:25 2022

@author: sivakr
"""

import PIL.Image
import numpy as np
import augmentation
import torch, torchvision

class RGBDLoader:
    def __init__(self, mode='train'):
        self.mode = mode
        self.blur_rgb = augmentation.GaussianBlur(signma=1)
        self.drop_channel = augmentation.DropChannel([(0,1,2), 3], -1)
        self.transform_rgb = torchvision.transforms.Compose([
            augmentation.Brightness(minmax=(0, .2)),
            augmentation.GaussianNoise(),
            augmentation.Clamp((0.0, 1.0)),
            torchvision.transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                            std=[0.5, 0.5, 0.5])
        ])
        self.transform_d = torchvision.transforms.Compose([
            augmentation.Brightness(minmax=(0, .8)),
            augmentation.GaussianNoise(std=0.05),
            augmentation.Clamp((0.15, 1.0)),
            torchvision.transforms.Normalize(mean=[0.575], std=[0.425]),
            augmentation.DepthUniformNoise(p=0.01, minmax=-1),
        ])
        self.crop_resize = augmentation.CropAndResize((227, 227), scale=(0.4, 1.0))
        # if mode == 'train':
        #     self.crop_resize = augmentation.CropAndResize((227,227), scale=(0.4, 1.0))
        # else:
        #     self.crop_resize = augmentation.CenterCrop((227,227))

    def __call__(self, rgb_path, d_path):
        # resize the cropped images
        img_shape = (32, 32)

        # load rgb
        pil_image = PIL.Image.open(rgb_path)
        pil_image = pil_image.resize(img_shape)
        if pil_image.mode != 'RGB':
            pil_image = pil_image.convert('RGB')

        np_rgb = np.array(pil_image, np.float32) / 255.0

        # load depth        
        d_img = PIL.Image.open(d_path)
        d_img = d_img.resize(img_shape)
        #print(d_img.mode)

        #dImg = np.squeeze(d_img)
        #r_channel = (dImg[:,:]).astype(np.float32).tobytes()
        #np_d = np.frombuffer(r_channel, dtype=np.float32).reshape(d_img.size[0], d_img.size[1], 1)
        dImg = np.squeeze(d_img)
        r_channel = (dImg[:,:]).astype(np.float32).tobytes()
        np_d = np.frombuffer(r_channel, dtype=np.float32).reshape((d_img.size[0], d_img.size[1], 1))
        #print(np_d)

        # transform
        # crop saperately, so the depth and rgb does not match
        #np_rgb = self.crop_resize(np_rgb)
        #np_d = self.crop_resize(np_d)

        # blur
        np_rgb = self.blur_rgb(np_rgb)
        # to tensor
        t_rgb = torch.from_numpy(np_rgb).permute((2,0,1))
        t_d = torch.from_numpy(np_d).permute((2,0,1))
        # transform rgb
        t_rgb = self.transform_rgb(t_rgb)
        # transform d
        t_d = self.transform_d(t_d)

        # concat again
        t_rgbd = torch.cat((t_rgb, t_d), dim=0)
        # randomly turn off rgb or d channel
        t_rgbd = self.drop_channel(t_rgbd)

        return t_rgbd
