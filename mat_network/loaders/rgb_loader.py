#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 8 19:38:25 2022

@author: sivakr
"""

import PIL.Image

def from_image(path):
    pil_image = PIL.Image.open(path)
    if pil_image.mode != 'RGB':
        pil_image = pil_image.convert('RGB')

    return pil_image