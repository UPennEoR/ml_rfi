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
    with tf.variable_scope('FCN', reuse=reuse):
        kt_phs = 3 #11
        kt_amp = 3
        pad_size = 68
        sh = x.get_shape().as_list()
        s = 1
        activation = tf.nn.leaky_relu
        input_layer = x#tf.cast(tf.reshape(x,[-1,sh[1],sh[2],sh[3]]),dtype=tf.float32)
#        tf.summary.image('IP_Amp',tf.reshape(input_layer[0,:,:,0],[1,pad_size,pad_size,1]))
        print('Input',sh)
        # Amplitude branch of the D-FCN
        #slayer1 = hf.stacked_layer(tf.reshape(input_layer[:,:,:,:1],[-1,sh[1],sh[2],1]),8*s,kt_amp,kt_amp,activation,[2,2],[2,2],bnorm=True,mode=mode_bn)
        slayer1 = hf.stacked_layer(input_layer[:,:,:,:1],8*s,kt_amp,kt_amp,activation,[2,2],[2,2],bnorm=True,mode=mode_bn)
        slayer2 = hf.stacked_layer(slayer1,s*16,kt_amp,kt_amp,activation,[2,2],[2,2],bnorm=True,mode=mode_bn)
        slayer3 = hf.stacked_layer(slayer2,s*32,kt_amp,kt_amp,activation,[2,2],[2,2],bnorm=True,mode=mode_bn,dropout=0.)
        s3sh = slayer3.get_shape().as_list()
        slayer4 = tf.layers.dropout(hf.stacked_layer(slayer3,s*64,kt_amp,kt_amp,activation,[2,2],[2,2],bnorm=True,mode=mode_bn),rate=0.)
        slayer5 = hf.stacked_layer(slayer4,s*128,kt_amp,kt_amp,activation,[2,2],[2,2],bnorm=True,mode=mode_bn,dropout=0.)
        s5sh = slayer5.get_shape().as_list()
        slayer6 = hf.stacked_layer(slayer5,s*256,kt_amp,kt_amp,activation,[2,2],[2,2],bnorm=True,mode=mode_bn,maxpool=None)
#        s6sh = slayer6.get_shape().as_list()
#        slayer7 = hf.stacked_layer(slayer6,s*512,1,1,activation,[1,1],[1,1],bnorm=True,dropout=d_out,mode=mode_bn,maxpool=None)
        
        # Phase branch of the D-FCN
        #slayer1b = hf.stacked_layer(tf.reshape(input_layer[:,:,:,1],[-1,sh[1],sh[2],1]),8*s,kt_phs,kt_phs,activation,[2,2],[2,2],bnorm=True,mode=mode_bn,maxpool=False)
        slayer1b = hf.stacked_layer(input_layer[:,:,:,1:],8*s,kt_phs,kt_phs,activation,[2,2],[2,2],bnorm=True,mode=mode_bn,maxpool=False) 
        slayer2b = hf.stacked_layer(slayer1b,16*s,kt_phs,kt_phs,activation,[2,2],[2,2],bnorm=True,mode=mode_bn,maxpool=False)
        slayer3b = hf.stacked_layer(slayer2b,32*s,kt_phs,kt_phs,activation,[2,2],[2,2],bnorm=True,mode=mode_bn,maxpool=False,dropout=0.)
        slayer4b = tf.layers.dropout(hf.stacked_layer(slayer3b,64*s,kt_phs,kt_phs,activation,[2,2],[2,2],bnorm=True,mode=mode_bn,maxpool=False),rate=0.)
        slayer5b = hf.stacked_layer(slayer4b,128*s,kt_phs,kt_phs,activation,[2,2],[2,2],bnorm=True,mode=mode_bn,maxpool=False,dropout=0.0)
        slayer6b = hf.stacked_layer(slayer5b,256*s,kt_phs,kt_phs,activation,[2,2],[2,2],bnorm=True,mode=mode_bn,maxpool=None)    
        s3sh = slayer3b.get_shape().as_list()
        s6sh = slayer6b.get_shape().as_list()
#        slayer7b = hf.stacked_layer(slayer6b,s*512,1,1,activation,[1,1],[1,1],bnorm=True,dropout=0.0,mode=mode_bn,maxpool=None)
#        tf.summary.image('Amp_l1',tf.reshape(tf.reduce_max(slayer1[0,:,:,:],axis=-1),[1,int(pad_size/2),int(pad_size/2),1]))
#        tf.summary.image('Phs_l1',tf.reshape(tf.reduce_max(slayer1b[0,:,:,:],axis=-1),[1,int(pad_size/2),int(pad_size/2),1]))

        # Combine both amplitude and phase branches going into the fully connected-convolutional layers
#        slayer7 = tf.concat([tf.layers.dropout(hf.tfnormalize(slayer7),rate=0.7),tf.layers.dropout(hf.tfnormalize(slayer7b),rate=0.7)],axis=-1)
        slayer6 = tf.concat([tf.layers.dropout(hf.tfnormalize(slayer6),rate=0.7),tf.layers.dropout(hf.tfnormalize(slayer6b),rate=0.7)],axis=-1)
#        slayer8 = hf.stacked_layer(slayer7,s*1024,1,1,activation,[1,1],[1,1],bnorm=True,dropout=d_out,mode=mode_bn,maxpool=None)
#        s8sh = slayer8.get_shape().as_list()
        # Upsampling through transpose convolutional layers
#        print('layer 6',s6sh)
#        print('layer 8',np.shape(slayer8))
        #upsamp1 = tf.layers.conv2d_transpose(slayer8,filters=s*256,strides=(2,2),kernel_size=[s6sh[1],s6sh[2]],activation=activation,padding='valid')
        k1_x = 7#int(s6sh[1]) + 1 - int(s8sh[1])
        k1_y = 7#int(s6sh[2]) + 1 - int(s8sh[2])
        upsamp1 = tf.layers.conv2d_transpose(slayer6,filters=s*128,data_format='channels_last',strides=(1,1),kernel_size=[k1_x,k1_y],activation=activation,padding='same')
        print('upsamp1',np.shape(upsamp1))
        print('slayer6',s6sh)
        print('slayer 3',s3sh)
        upsamp1 = tf.layers.dropout(tf.layers.batch_normalization(upsamp1,scale=True,center=True,training=mode_bn,fused=True),rate=0.7)
        upsamp1 = tf.concat([tf.layers.dropout(hf.tfnormalize(upsamp1),rate=0.7),tf.layers.dropout(hf.tfnormalize(slayer5),rate=0.7),tf.layers.dropout(hf.tfnormalize(slayer5b),rate=0.7)],axis=-1)
#        upsamp1 = tf.layers.conv2d(inputs=upsamp1,
#                                 filters=512,
#                                 kernel_size=[3,3],
#                                 padding="same",
#                                 activation=tf.nn.relu)
#        upsamp1 = tf.layers.batch_normalization(upsamp1,scale=True,center=True,training=mode_bn,fused=True)
        print(np.shape(upsamp1))
        print(s5sh)
        upsh1 = upsamp1.get_shape().as_list()
        k2_x = 7#int(s5sh[1]) + 1 - int(upsh1[1])
        k2_y = 7#int(s5sh[2]) + 1 - int(upsh1[2])
        upsamp2 = tf.layers.conv2d_transpose(upsamp1,filters=s*32,data_format='channels_last',strides=(4,4),kernel_size=[k2_x,k2_y],activation=activation,padding='same')
        upsamp2 = tf.layers.dropout(tf.layers.batch_normalization(upsamp2,scale=True,center=True,training=mode_bn,fused=True),rate=0.7)
        upsamp2 = tf.concat([
                tf.layers.dropout(hf.tfnormalize(upsamp2),rate=0.7),
                tf.layers.dropout(hf.tfnormalize(slayer3),rate=0.7),
                tf.layers.dropout(hf.tfnormalize(slayer3b),rate=0.7)],axis=-1)
#        upsamp2 = tf.layers.conv2d(inputs=upsamp2,
#                                   filters=256,
#                                   kernel_size=[3,3],
#                                   padding="same",
#                                   activation=tf.nn.relu)
        upsh2 = upsamp2.get_shape().as_list()
#        upsamp2 = tf.layers.batch_normalization(upsamp2,scale=True,center=True,training=mode_bn,fused=True)
        k3_x = 7#int(s3sh[1]) + 1 - int(upsh2[1])
        k3_y = 7#int(s3sh[2]) + 1 - int(upsh2[2]) 
        upsamp3 = tf.layers.conv2d_transpose(upsamp2,filters=s*8,data_format='channels_last',strides=(4,4),kernel_size=[k3_x,k3_y],activation=activation,padding='same')
        upsh3 = upsamp3.get_shape().as_list()
        print(upsh3)
#        upsamp3 = tf.layers.dropout(tf.layers.batch_normalization(upsamp3,scale=True,center=True,training=mode_bn,fused=True),rate=0.)
        upsamp3 = tf.concat([
                tf.layers.dropout(hf.tfnormalize(upsamp3),rate=0.7),
                tf.layers.dropout(hf.tfnormalize(slayer1),rate=0.7),
                tf.layers.dropout(hf.tfnormalize(slayer1b),rate=0.7)],axis=-1)
#        upsamp3 = tf.layers.conv2d(inputs=upsamp3,
#                                   filters=64,
#                                   kernel_size=[3,3],
#                                   padding="same",
#                                   activation=tf.nn.relu)
#        upsamp3 = tf.layers.batch_normalization(upsamp3,scale=True,center=True,training=mode_bn,fused=True)
        out_filter = 2
        upsh3 = upsamp3.get_shape().as_list()
        k3_x = 7#int(sh[1]) + 1 - int(upsh3[1])
        k3_y = 7#int(sh[2]) + 1 - int(upsh3[2])
        upsamp4 = tf.layers.conv2d_transpose(upsamp3,filters=out_filter-1,strides=(2,2),data_format='channels_last',kernel_size=[k3_x,k3_y],activation=activation,padding='same')
        upsamp4 = tf.concat([
                  tf.layers.dropout(hf.tfnormalize(upsamp4),rate=0.7),
                  tf.layers.dropout(hf.tfnormalize(input_layer[:,:,:,:1]),rate=0.7),
                  tf.layers.dropout(hf.tfnormalize(input_layer[:,:,:,1:]),rate=0.7)],axis=-1)
#        upsamp4 = tf.layers.dropout(tf.layers.batch_normalization(upsamp4,scale=True,center=True,training=mode_bn,fused=True),rate=0.7)
        upsamp4 = tf.layers.conv2d(inputs=upsamp4,
                                   filters=out_filter,
                                   kernel_size=[1,1],
                                   padding="same",
                                   activation=tf.nn.relu)


#        tf.summary.image('Flag Guess',tf.reshape(upsamp4[0,:,:,1],[1,pad_size,pad_size,1]))
        print(np.shape(upsamp4))
        final_conv = tf.reshape(upsamp4,[-1,tf.shape(x)[1]*tf.shape(x)[2],2])
        print(np.shape(final_conv))
#        tf.summary.image('ArgMax',tf.cast(tf.reshape(tf.argmax(upsamp4[0,:,:,:],axis=-1),[1,pad_size,pad_size,1]),dtype=tf.float32))
    return final_conv
