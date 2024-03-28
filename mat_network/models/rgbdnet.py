#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 21 14:16:23 2022

@author: sivakr
"""

from .matnet import MatNet

class RGBDNet(MatNet):
    def __init__(self, cfg):
        cfg['in_channels'] = 4
        super(RGBDNet, self).__init__(cfg)