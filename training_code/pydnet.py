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

from layers import *

class pydnet(object):

    def __init__(self, placeholders=None):
        self.model_collection = ['PyDnet']    
        self.placeholders = placeholders
        self.build_model()
        self.build_outputs()        
    
    def build_model(self):    
      with tf.variable_scope("pyramid") as scope:
        pyramid = self.build_pyramid(self.placeholders['im0'])
      # SCALE 6
      with tf.variable_scope("L6") as scope:
        with tf.variable_scope("estimator") as scope:
          conv6 = self.build_estimator(pyramid[6])
          self.disp7 = self.get_disp(conv6)
        with tf.variable_scope("upsampler") as scope:
          upconv6 = self.bilinear_upsampling_by_deconvolution(conv6)
      # SCALE 5
      with tf.variable_scope("L5") as scope:
        with tf.variable_scope("estimator") as scope:
          conv5 = self.build_estimator(pyramid[5], upconv6)
          self.disp6 = self.get_disp(conv5)
        with tf.variable_scope("upsampler") as scope:
          upconv5 = self.bilinear_upsampling_by_deconvolution(conv5)
      # SCALE 4
      with tf.variable_scope("L4") as scope:
        with tf.variable_scope("estimator") as scope:
          conv4 = self.build_estimator(pyramid[4], upconv5)
          self.disp5 = self.get_disp(conv4)
        with tf.variable_scope("upsampler") as scope:
          upconv4 = self.bilinear_upsampling_by_deconvolution(conv4)
      # SCALE 3
      with tf.variable_scope("L3") as scope:
        with tf.variable_scope("estimator") as scope:
          conv3 = self.build_estimator(pyramid[3], upconv4)
          self.disp4 = self.get_disp(conv3)
        with tf.variable_scope("upsampler") as scope:
          upconv3 = self.bilinear_upsampling_by_deconvolution(conv3)
      # SCALE 2
      with tf.variable_scope("L2") as scope:
        with tf.variable_scope("estimator") as scope:
          conv2 = self.build_estimator(pyramid[2], upconv3)
          self.disp3 = self.get_disp(conv2)
        with tf.variable_scope("upsampler") as scope:
          upconv2 = self.bilinear_upsampling_by_deconvolution(conv2)
      # SCALE 1
      with tf.variable_scope("L1") as scope:
        with tf.variable_scope("estimator") as scope:
          conv1 = self.build_estimator(pyramid[1], upconv2)
          self.disp2 = self.get_disp(conv1)

    # Pyramidal features extraction
    def build_pyramid(self, input_batch):
      features = []
      features.append(input_batch)
      with tf.variable_scope("conv1a"):
        conv1a = conv2d_leaky(input_batch, [3, 3, 3, 16], [16], 2, True)
      with tf.variable_scope("conv1b"):
        conv1b = conv2d_leaky(conv1a, [3, 3, 16, 16], [16], 1, True)
      features.append(conv1b)
      with tf.variable_scope("conv2a"):
        conv2a = conv2d_leaky(conv1b, [3, 3, 16, 32], [32], 2, True)
      with tf.variable_scope("conv2b"):
        conv2b = conv2d_leaky(conv2a, [3, 3, 32, 32], [32], 1, True)
      features.append(conv2b)
      with tf.variable_scope("conv3a"):
        conv3a = conv2d_leaky(conv2b, [3, 3, 32, 64], [64], 2, True)
      with tf.variable_scope("conv3b"):
        conv3b = conv2d_leaky(conv3a, [3, 3, 64, 64], [64], 1, True)
      features.append(conv3b)
      with tf.variable_scope("conv4a"):
        conv4a = conv2d_leaky(conv3b, [3, 3, 64, 96], [96], 2, True)
      with tf.variable_scope("conv4b"):
        conv4b = conv2d_leaky(conv4a, [3, 3, 96, 96], [96], 1, True)
      features.append(conv4b)
      with tf.variable_scope("conv5a"):
        conv5a = conv2d_leaky(conv4b, [3, 3, 96, 128], [128], 2, True)
      with tf.variable_scope("conv5b"):
        conv5b = conv2d_leaky(conv5a, [3, 3, 128, 128], [128], 1, True)
      features.append(conv5b)
      with tf.variable_scope("conv6a"):
        conv6a = conv2d_leaky(conv5b, [3, 3, 128, 192], [192], 2, True)
      with tf.variable_scope("conv6b"):
        conv6b = conv2d_leaky(conv6a, [3, 3, 192, 192], [192], 1, True)
      features.append(conv6b)
      return features	     

    # Single scale estimator
    def build_estimator(self, features, upsampled_disp=None):
      if upsampled_disp is not None:
        disp2 = tf.concat([features, upsampled_disp], -1)
      else:
        disp2 = features
      with tf.variable_scope("disp-3") as scope:
        disp3 = conv2d_leaky(disp2, [3, 3, disp2.shape[3], 96], [96], 1, True)
      with tf.variable_scope("disp-4") as scope:
        disp4 = conv2d_leaky(disp3, [3, 3, disp3.shape[3], 64], [64], 1, True)
      with tf.variable_scope("disp-5") as scope:
        disp5 = conv2d_leaky(disp4, [3, 3, disp4.shape[3], 32], [32], 1, True)
      with tf.variable_scope("disp-6") as scope:
        disp6 = conv2d_leaky(disp5, [3, 3, disp5.shape[3], 8], [8], 1, False) # 8 channels for compatibility with @other@ devices
      return disp6
      
    # Upsampling layer
    def bilinear_upsampling_by_deconvolution(self,x):
      f = x.get_shape().as_list()[-1]
      return deconv2d_leaky(x, [2, 2, f, f], f, 2, True)

    # Disparity prediction layer
    def get_disp(self, x):
      disp = 0.3 * tf.nn.sigmoid(tf.slice(x, [0,0,0,0], [-1,-1,-1,2]))
      return disp
      
    # Build multi-scale outputs
    def build_outputs(self):
      shape = tf.shape(self.placeholders['im0']) 
      size = [shape[1], shape[2]]          
      self.results = tf.image.resize_images(self.disp2, size), tf.image.resize_images(self.disp3, size), tf.image.resize_images(self.disp4, size)
