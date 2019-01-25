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

from __future__ import division

# only keep warnings and errors
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='1'

import numpy as np
import argparse
import re
import time
import tensorflow as tf
import tensorflow.contrib.slim as slim

from pydnet import *

parser = argparse.ArgumentParser(description='Argument parser')

parser.add_argument('--dataset',           type=str,   help='dataset to train on, kitti, or cityscapes', default='kitti')
parser.add_argument('--datapath',          type=str,   help='path to the data', required=True)
parser.add_argument('--filenames',         type=str,   help='path to the filenames text file', required=True)
parser.add_argument('--output_directory',  type=str,   help='output directory for test disparities, if empty outputs to checkpoint folder', default='.')
parser.add_argument('--checkpoint_dir',        type=str,   help='path to a specific checkpoint to load', default='checkpoint/IROS18/pydnet')
parser.add_argument('--resolution',        type=int, default=1, help='resolution [1:H, 2:Q, 3:E]')

args = parser.parse_args()

def read_image(image_path):
    image  = tf.image.decode_image(tf.read_file(args.datapath+'/'+image_path))
    image.set_shape( [None, None, 3])
    image  = tf.image.convert_image_dtype(image,  tf.float32)
    image  = tf.expand_dims(tf.image.resize_images(image,  [256, 512], tf.image.ResizeMethod.AREA), 0)

    return image

def test(params):

    input_queue = tf.train.string_input_producer([args.filenames], shuffle=False)
    line_reader = tf.TextLineReader()
    _, line = line_reader.read(input_queue)
    img_path = tf.string_split([line]).values[0]
    img = read_image(img_path)

    placeholders = {'im0':img}

    with tf.variable_scope("model") as scope:    
      model = pydnet(placeholders)

    # SESSION
    config = tf.ConfigProto(allow_soft_placement=True)
    sess = tf.Session(config=config)

    # SAVER
    train_saver = tf.train.Saver()

    # INIT
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())
    coordinator = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coordinator)

    # RESTORE
    train_saver.restore(sess, args.checkpoint_dir)

    # GET TEST IMAGES NAMES
    f = open(args.filenames, 'r')
    samples = len(f.readlines())
    f.close()

    print('Testing {} frames'.format(samples))
    disparities = np.zeros((samples, 256, 512), dtype=np.float32)
    for step in range(samples):
        print('Running %d out of %d'%(step, samples))

        # If you want to evaluate lower resolution results, just get results[1] or results[2]
        disp = sess.run(model.results[args.resolution-1])
        disparities[step] = disp[0,:,:,0].squeeze()

    print('Test done!')

    print('Saving disparities as .npy')
    if args.output_directory == '':
        output_directory = os.path.dirname(args.checkpoint_dir)
    else:
        output_directory = args.output_directory
    np.save(output_directory + '/disparities.npy', disparities)

    print('Disparities saved!')

def main(_):

    test(args)

if __name__ == '__main__':
    tf.app.run()
