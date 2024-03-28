#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 8 16:58:18 2022

@author: sivakr
"""

import PIL.Image
import numpy as np
import torch

def from_image(path):
    pil_image = PIL.Image.open(path)
    if pil_image.mode != 'L':
        pil_image = pil_image.convert('L')

    return pil_image

def from_exr(path):
    d_image = PIL.Image.open(path)
    np_depth = np.array(d_image, np.float32) / 255.0
    return np_depth
