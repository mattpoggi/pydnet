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


import tensorflow as tf
import sys
import os
import argparse
import time
import datetime
from utils import *
from pydnet import *


parser = argparse.ArgumentParser(description='Argument parser')

""" Arguments related to network architecture"""
parser.add_argument(
    '--src',
    dest='src',
    type=str,
    default='0',
    help='choose input for videocapture')
parser.add_argument(
    '--save_ratio',
    dest='save_ratio',
    type=bool,
    default=False,
    help='crop center of image to save aspect ratio')
parser.add_argument(
    '--host',
    dest='host',
    type=str,
    choices=[
        'cpu',
        'gpu'],
    default='cpu',
    help='choose host')
parser.add_argument(
    '--model',
    dest='model',
    type=str,
    choices=[
        'pydnet',
        'pydnet2'],
    default='pydnet',
    help='choose model')
parser.add_argument(
    '--resolution',
    dest='resolution',
    type=int,
    default=1,
    help='resolution [1:H, 2:Q, 3:E]')
parser.add_argument(
    '--checkpoint_dir',
    dest='checkpoint_dir',
    type=str,
    default='checkpoint/IROS18/pydnet',
    help='checkpoint directory')

args = parser.parse_args()

if args.host == 'cpu':
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
elif args.host == 'gpu':
    pass


def main(_):

    with tf.Graph().as_default():
        if args.model == 'pydnet':
            height, width = 256, 640
        elif args.model == 'pydnet2':
            height, width = 192, 512

        placeholders = {
            'im0': tf.compat.v1.placeholder(
                tf.float32, [
                    None, None, None, 3], name='im0')}

        with tf.compat.v1.variable_scope("model") as scope:
            if args.model == 'pydnet':
                model = pydnet(placeholders)
            elif args.model == 'pydnet2':
                model = pydnet2(placeholders)

        init = tf.group(tf.compat.v1.global_variables_initializer(),
                        tf.compat.v1.local_variables_initializer())

        loader = tf.compat.v1.train.Saver()
        saver = tf.compat.v1.train.Saver()
        src = args.src
        try:
            src = int(args.src)
        except ValueError:
            pass
        cam = cv2.VideoCapture(src)

        with tf.compat.v1.Session() as sess:
            sess.run(init)
            if args.model == 'pydnet2':
                args.checkpoint_dir = 'checkpoint/ITS/pydnet2'

            loader.restore(sess, args.checkpoint_dir)
            while True:
                for i in range(4):
                    cam.grab()
                ret_val, img = cam.read()

                if args.save_ratio:
                    img = resize_with_aspect_ratio(img, width, height)
                else:
                    img = cv2.resize(
                        img, (width, height)).astype(
                        np.float32) / 255.
                img = np.expand_dims(img, 0)
                start = time.time()
                disp = sess.run(
                    model.results[args.resolution - 1], feed_dict={placeholders['im0']: img})
                end = time.time()

                color_scaling = 20
                if args.model == 'pydnet2':
                    color_scaling = 1 / 64.

                disp_color = applyColorMap(
                    disp[0, :, :, 0] * color_scaling, 'plasma')
                toShow = (
                    np.concatenate(
                        (img[0],
                         disp_color),
                        0) *
                    255.).astype(
                    np.uint8)
                toShow = cv2.resize(toShow, (width, height * 2))

                cv2.imshow(args.model, toShow)
                k = cv2.waitKey(1)
                if k == 1048603 or k == 27:
                    break  # esc to quit
                if k == ord('p'):
                    cv2.waitKey(0)  # 'p' to pause

                print("Time: " + str(end - start))
                del img
                del disp
                del toShow

            cam.release()


if __name__ == '__main__':
    tf.compat.v1.app.run()
