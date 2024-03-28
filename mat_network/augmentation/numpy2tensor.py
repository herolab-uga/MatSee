#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 10 10:25:25 2022

@author: sivakr
"""

import numpy as np
import torch

class Numpy2Tensor(object):
    def __init__(self):
        pass

    def __call__(self, np_arr : np.ndarray):
        if np_arr.dtype is np.uint8:
            tensor =  torch.tensor(np_arr, dtype=torch.float32) / 255.0
        else:
            tensor = torch.tensor(np_arr, dtype=torch.float32)

        if len(tensor.shape) == 3:
            tensor = tensor.permute((2, 0, 1))

        return tensor