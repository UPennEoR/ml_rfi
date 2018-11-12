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
from statsmodels import robust
import sys
from copy import copy
import h5py


def AmpPhsFCN(x, reuse=None, mode_bn=True, d_out=0.):
    with tf.variable_scope('FCN', reuse=reuse):
        kt_phs = 11
        kt_amp = 3
        pad_size = 68
        sh = x.get_shape().as_list()
        s = 1
        activation = tf.nn.leaky_relu
        input_layer = tf.cast(tf.reshape(x,[-1,sh[1],sh[2],sh[3]]),dtype=tf.float32)
        tf.summary.image('IP_Amp',tf.reshape(input_layer[0,:,:,0],[1,pad_size,pad_size,1]))

        # Amplitude branch of the D-FCN
        slayer1 = hf.stacked_layer(tf.reshape(input_layer[:,:,:,:1],[-1,sh[1],sh[2],1]),8*s,kt_amp,kt_amp,activation,[2,2],[2,2],bnorm=True,mode=mode_bn)
        slayer2 = hf.stacked_layer(slayer1,s*16,kt_amp,kt_amp,activation,[2,2],[2,2],bnorm=True,mode=mode_bn)
        slayer3 = hf.stacked_layer(slayer2,s*32,kt_amp,kt_amp,activation,[2,2],[2,2],bnorm=True,mode=mode_bn,dropout=0.)
        s3sh = slayer3.get_shape().as_list()
        slayer4 = tf.layers.dropout(hf.stacked_layer(slayer3,s*64,kt_amp,kt_amp,activation,[2,2],[2,2],bnorm=True,mode=mode_bn),rate=0.)
        slayer5 = hf.stacked_layer(slayer4,s*128,kt_amp,kt_amp,activation,[2,2],[2,2],bnorm=True,mode=mode_bn,dropout=0.)
        slayer6 = hf.stacked_layer(slayer5,s*256,kt_amp,kt_amp,activation,[2,2],[2,2],bnorm=True,mode=mode_bn)
        s6sh = slayer6.get_shape().as_list()
        slayer7 = hf.stacked_layer(slayer6,s*512,1,1,activation,[1,1],[1,1],bnorm=True,dropout=d_out,mode=mode_bn)
        
        # Phase branch of the D-FCN
        slayer1b = hf.stacked_layer(tf.reshape(input_layer[:,:,:,1],[-1,sh[1],sh[2],1]),8*s,kt_phs,kt_phs,activation,[2,2],[2,2],bnorm=True,mode=mode_bn,maxpool=False)
        slayer2b = hf.stacked_layer(slayer1b,16*s,kt_phs,kt_phs,activation,[2,2],[2,2],bnorm=True,mode=mode_bn,maxpool=False)
        slayer3b = hf.stacked_layer(slayer2b,32*s,kt_phs,kt_phs,activation,[2,2],[2,2],bnorm=True,mode=mode_bn,maxpool=False,dropout=0.)
        slayer4b = tf.layers.dropout(hf.stacked_layer(slayer3b,64*s,kt_phs,kt_phs,activation,[2,2],[2,2],bnorm=True,mode=mode_bn,maxpool=False),rate=0.)
        slayer5b = hf.stacked_layer(slayer4b,128*s,kt_phs,kt_phs,activation,[2,2],[2,2],bnorm=True,mode=mode_bn,maxpool=False,dropout=0.0)
        slayer6b = hf.stacked_layer(slayer5b,256*s,kt_phs,kt_phs,activation,[2,2],[2,2],bnorm=True,mode=mode_bn,maxpool=False)    
        s3sh = slayer3b.get_shape().as_list()
        s6sh = slayer6b.get_shape().as_list()
        slayer7b = hf.stacked_layer(slayer6b,s*512,1,1,activation,[1,1],[1,1],bnorm=True,dropout=0.0,mode=mode_bn)
        tf.summary.image('Amp_l1',tf.reshape(tf.reduce_max(slayer1[0,:,:,:],axis=-1),[1,int(pad_size/2),int(pad_size/2),1]))
        tf.summary.image('Phs_l1',tf.reshape(tf.reduce_max(slayer1b[0,:,:,:],axis=-1),[1,int(pad_size/2),int(pad_size/2),1]))

        # Combine both amplitude and phase branches going into the fully connected-convolutional layers
        slayer7 = tf.concat([tf.layers.dropout(hf.tfnormalize(slayer7),rate=0.7),tf.layers.dropout(hf.tfnormalize(slayer7b),rate=0.7)],axis=-1)
        slayer8 = hf.stacked_layer(slayer7,s*1024,1,1,activation,[1,1],[1,1],bnorm=True,dropout=d_out,mode=mode_bn)

        # Upsampling through transpose convolutional layers
        upsamp1 = tf.layers.conv2d_transpose(slayer8,filters=s*256,kernel_size=[s6sh[1],s6sh[1]],activation=activation)
        upsamp1 = tf.layers.dropout(tf.layers.batch_normalization(upsamp1,scale=True,center=True,training=mode_bn,fused=True),rate=0.7)
        upsamp1 = tf.concat([tf.layers.dropout(hf.tfnormalize(upsamp1),rate=0.7),tf.layers.dropout(hf.tfnormalize(slayer6),rate=0.7),tf.layers.dropout(hf.tfnormalize(slayer6b),rate=0.7)],axis=-1)
        upsamp2 = tf.layers.conv2d_transpose(upsamp1,filters=s*128,kernel_size=[s6sh[1]+1,s6sh[1]+1],activation=activation)
        upsamp2 = tf.layers.dropout(tf.layers.batch_normalization(upsamp2,scale=True,center=True,training=mode_bn,fused=True),rate=0.7)
        upsamp2 = tf.concat([
                tf.layers.dropout(hf.tfnormalize(upsamp2),rate=0.7),
                tf.layers.dropout(hf.tfnormalize(slayer5),rate=0.7),
                tf.layers.dropout(hf.tfnormalize(slayer5b),rate=0.7)],axis=-1)
        upsamp3 = tf.layers.conv2d_transpose(upsamp2,filters=s*32,kernel_size=[s3sh[1]-2*s6sh[1]+1,s3sh[1]-2*s6sh[1]+1],activation=activation)
        upsh3 = upsamp3.get_shape().as_list()
        upsamp3 = tf.layers.dropout(tf.layers.batch_normalization(upsamp3,scale=True,center=True,training=mode_bn,fused=True),rate=0.)
        upsamp3 = tf.concat([
                tf.layers.dropout(hf.tfnormalize(upsamp3),rate=0.7),
                tf.layers.dropout(hf.tfnormalize(slayer3),rate=0.7),
                tf.layers.dropout(hf.tfnormalize(slayer3b),rate=0.7)],axis=-1)
        out_filter = 2
        upsamp4 = tf.layers.conv2d_transpose(upsamp3,filters=out_filter,kernel_size=[int(sh[1] - upsh3[1]) + 1,int(sh[1] - upsh3[1]) + 1],activation=activation)
        upsamp4 = tf.layers.dropout(tf.layers.batch_normalization(upsamp4,scale=True,center=True,training=mode_bn,fused=True),rate=0.7)
        tf.summary.image('Flag Guess',tf.reshape(upsamp4[0,:,:,1],[1,pad_size,pad_size,1]))
        final_conv = tf.reshape(upsamp4,[-1,sh[1]*sh[1],2])
        tf.summary.image('ArgMax',tf.cast(tf.reshape(tf.argmax(upsamp4[0,:,:,:],axis=-1),[1,pad_size,pad_size,1]),dtype=tf.float32))
    return final_conv
