from __future__ import division, print_function, absolute_import
import matplotlib
matplotlib.use("AGG")
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from glob import glob
import helper_functions as hf
from time import time
import os
from sklearn.metrics import roc_curve
from sklearn.metrics import confusion_matrix
import sys
from copy import copy
import h5py


def AmpPhsFCN(x, reuse=None, mode_bn=True, d_out=0.):

    """                                                                                                                                        
    Amplitude only Fully Convolutional Neural Network Model. Uses 6 convolutional layers in the downsampling                                    
    stage, into 2 fully connected layers, and upsampled through 4 transpose convolutional layers.                                              
    """
    
    with tf.variable_scope('FCN', reuse=reuse):
        kt_phs = 3
        kt_amp = 3
        pad_size = 8
        sh = x.get_shape().as_list()
        s = 2
        activation = tf.nn.leaky_relu
        input_layer = x
        # Amplitude branch of the D-FCN
        slayer1 = hf.stacked_layer(input_layer[:,:,:,:1],8*s,kt_amp,kt_amp,activation,[2,2],[2,2],bnorm=True,mode=mode_bn)
        slayer2 = hf.stacked_layer(slayer1,s*16,kt_amp,kt_amp,activation,[2,2],[2,2],bnorm=True,mode=mode_bn)
        slayer3 = hf.stacked_layer(slayer2,s*32,kt_amp,kt_amp,activation,[2,2],[2,2],bnorm=True,mode=mode_bn,dropout=d_out)
        s3sh = slayer3.get_shape().as_list()
        slayer4 = tf.layers.dropout(hf.stacked_layer(slayer3,s*64,kt_amp,kt_amp,activation,[2,2],[2,2],bnorm=True,mode=mode_bn),rate=d_out)
        slayer5 = hf.stacked_layer(slayer4,s*128,kt_amp,kt_amp,activation,[2,2],[2,2],bnorm=True,mode=mode_bn,dropout=d_out)
        s5sh = slayer5.get_shape().as_list()
        slayer6 = hf.stacked_layer(slayer5,s*256,kt_amp,kt_amp,activation,[2,2],[2,2],bnorm=True,mode=mode_bn,maxpool=None)
        
        # Phase branch of the D-FCN
        slayer1b = hf.stacked_layer(input_layer[:,:,:,1:],8*s,kt_phs,kt_phs,activation,[2,2],[2,2],bnorm=True,mode=mode_bn,maxpool=False) 
        slayer2b = hf.stacked_layer(slayer1b,16*s,kt_phs,kt_phs,activation,[2,2],[2,2],bnorm=True,mode=mode_bn,maxpool=False)
        slayer3b = hf.stacked_layer(slayer2b,32*s,kt_phs,kt_phs,activation,[2,2],[2,2],bnorm=True,mode=mode_bn,maxpool=False,dropout=d_out)
        slayer4b = tf.layers.dropout(hf.stacked_layer(slayer3b,64*s,kt_phs,kt_phs,activation,[2,2],[2,2],bnorm=True,mode=mode_bn,maxpool=False),rate=d_out)
        slayer5b = hf.stacked_layer(slayer4b,128*s,kt_phs,kt_phs,activation,[2,2],[2,2],bnorm=True,mode=mode_bn,maxpool=False,dropout=d_out)
        slayer6b = hf.stacked_layer(slayer5b,256*s,kt_phs,kt_phs,activation,[2,2],[2,2],bnorm=True,mode=mode_bn,maxpool=None)    
        s3sh = slayer3b.get_shape().as_list()
        s6sh = slayer6b.get_shape().as_list()

        # Combine both amplitude and phase branches going into the fully connected-convolutional layers
        slayer6 = tf.concat([tf.layers.dropout(hf.tfnormalize(slayer6),rate=d_out),tf.layers.dropout(hf.tfnormalize(slayer6b),rate=d_out)],axis=-1)

        k1_x = 3
        k1_y = 3
        upsamp1 = tf.layers.conv2d_transpose(slayer6,filters=s*128,data_format='channels_last',strides=(1,1),kernel_size=[k1_x,k1_y],activation=activation,padding='same')

        upsamp1 = tf.layers.dropout(tf.layers.batch_normalization(upsamp1,scale=True,center=True,training=mode_bn,fused=True),rate=d_out)
        upsamp1 = tf.concat([tf.layers.dropout(hf.tfnormalize(upsamp1),rate=d_out),tf.layers.dropout(hf.tfnormalize(slayer5),rate=d_out),tf.layers.dropout(hf.tfnormalize(slayer5b),rate=d_out)],axis=-1)

        upsh1 = upsamp1.get_shape().as_list()
        k2_x = 3
        k2_y = 3
        upsamp2 = tf.layers.conv2d_transpose(upsamp1,filters=s*32,data_format='channels_last',strides=(4,4),kernel_size=[k2_x,k2_y],activation=activation,padding='same')
        upsamp2 = tf.layers.dropout(tf.layers.batch_normalization(upsamp2,scale=True,center=True,training=mode_bn,fused=True),rate=d_out)
        upsamp2 = tf.concat([
                tf.layers.dropout(hf.tfnormalize(upsamp2),rate=d_out),
                tf.layers.dropout(hf.tfnormalize(slayer3),rate=d_out),
                tf.layers.dropout(hf.tfnormalize(slayer3b),rate=d_out)],axis=-1)

        upsh2 = upsamp2.get_shape().as_list()
        k3_x = 3
        k3_y = 3
        upsamp3 = tf.layers.conv2d_transpose(upsamp2,filters=s*8,data_format='channels_last',strides=(4,4),kernel_size=[k3_x,k3_y],activation=activation,padding='same')
        upsh3 = upsamp3.get_shape().as_list()

        upsamp3 = tf.concat([
                tf.layers.dropout(hf.tfnormalize(upsamp3),rate=d_out),
                tf.layers.dropout(hf.tfnormalize(slayer1),rate=d_out),
                tf.layers.dropout(hf.tfnormalize(slayer1b),rate=d_out)],axis=-1)

        out_filter = 2
        upsh3 = upsamp3.get_shape().as_list()
        k3_x = 3
        k3_y = 3
        upsamp4 = tf.layers.conv2d_transpose(upsamp3,filters=out_filter-1,strides=(2,2),data_format='channels_last',kernel_size=[k3_x,k3_y],activation=activation,padding='same')
        upsamp4 = tf.concat([
                  tf.layers.dropout(hf.tfnormalize(upsamp4),rate=d_out),
                  tf.layers.dropout(hf.tfnormalize(input_layer[:,:,:,:1]),rate=d_out),
                  tf.layers.dropout(hf.tfnormalize(input_layer[:,:,:,1:]),rate=d_out)],axis=-1)

        upsamp4 = tf.layers.conv2d(inputs=upsamp4,
                                   filters=out_filter,
                                   kernel_size=[1,1],
                                   padding="same",
                                   activation=activation)

        final_conv = tf.reshape(upsamp4,[-1,tf.shape(x)[1]*tf.shape(x)[2],2])
    return final_conv
