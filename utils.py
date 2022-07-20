#
# MIT License
#
# Copyright (c) 2018 Matteo Poggi m.poggi@unibo.it
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import numpy as np
from matplotlib import cm
import cv2


# resize with cropping aspect ratio
def resize_with_aspect_ratio(img, width, height):
    original_height, original_width = img.shape[:2]
    if original_width / original_height < width / height:
        ratio = width / original_width
        new_w = width
        new_h = int(ratio * original_height)
        img = cv2.resize(img, (new_w, new_h)).astype(np.float32) / 255.
        img = img[(new_h - height) // 2:-(new_h - height) // 2, :]
    else:
        ratio = height / original_width
        new_h = height
        new_w = int(ratio * original_width)
        img = cv2.resize(img, (new_w, new_h)).astype(np.float32) / 255.
        img = img[:, (new_w - width) // 2:-(new_w - width) // 2]

    return img

# Colormap wrapper


def applyColorMap(img, cmap):
    colormap = cm.get_cmap(cmap)
    colored = colormap(img)
    return np.float32(cv2.cvtColor(
        np.uint8(colored * 255), cv2.COLOR_RGBA2BGR)) / 255.

# 2D convolution wrapper


def count_text_lines(file_path):
    f = open(file_path, 'r')
    lines = f.readlines()
    f.close()
    return len(lines)
