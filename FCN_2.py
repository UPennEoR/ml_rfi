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

# Run on a single GPU
os.environ["CUDA_VISIBLE_DEVICES"]="1"

# Training Params                                                                                                                                  
num_steps = 10000
batch_size = 8
pad_size = 68
model_dir = np.sort(glob("./FCN/model_*"))
try:
    model = model_dir[-1].split('/')[-1].split('.')[0]+'.ckpt'
    start_step = int(model.split('_')[1].split('.ckpt')[0])
    print(model)
except:
    start_step = 0
    print('Starting training at step %i' % start_step)
mode = ''

# Generator Network                                                                                                                               
# Input: Noise, Output: Image                                                                                                                     

def FCN(x, reuse=None):
    with tf.variable_scope('FCN', reuse=reuse):
        kt = 3
        kf = 3
        activation = tf.nn.relu
        sh = x.get_shape().as_list()
        input_layer = tf.cast(tf.reshape(x,[-1,sh[1],sh[2],2]),dtype=tf.float32)

        # Convolution / Downsampling layers
        slayer1 = hf.stacked_layer(input_layer,16,kt,kf,activation,[2,2],[2,2],bnorm=True)
        s1sh = slayer1.get_shape().as_list()
        slayer2 = hf.stacked_layer(slayer1,32,kt,kf,activation,[2,2],[2,2],bnorm=True)
        slayer3 = hf.stacked_layer(slayer2,64,kt,kf,activation,[2,2],[2,2],bnorm=True)
        slayer4 = hf.stacked_layer(slayer3,128,kt,kf,activation,[2,2],[2,2],bnorm=True)
        s4sh = slayer4.get_shape().as_list()
        slayer5 = tf.layers.dropout(hf.stacked_layer(slayer4,256,1,1,activation,[2,2],[2,2],bnorm=True),rate=0.5,training=True)
        slayer6 = hf.stacked_layer(slayer5,512,1,1,activation,[2,2],[2,2],bnorm=True)

        # Fully connected and convolutional layers
        slayer7 = tf.layers.dropout(hf.stacked_layer(slayer6,2048,1,1,activation,[1,1],[1,1],bnorm=True),rate=0.5,training=True)
        slayer8 = hf.stacked_layer(slayer7,2048,1,1,activation,[1,1],[1,1],bnorm=True)

        # Transpose convolution layers
        upsamp1 = tf.layers.conv2d_transpose(slayer8,filters=1,kernel_size=[s4sh[1],s4sh[1]],activation=activation)
        upsamp1 = tf.add(upsamp1,tf.reshape(tf.reduce_max(slayer4,axis=-1),[-1,s4sh[1],s4sh[1],1]))
        upsamp1 = tf.contrib.layers.batch_norm(upsamp1,scale=True)
        upsamp2 = tf.layers.conv2d_transpose(upsamp1,filters=1,kernel_size=[s4sh[1]+1,s4sh[1]+1],activation=activation)
        upsamp2 = tf.contrib.layers.batch_norm(upsamp2,scale=True)
        upsamp3 = tf.layers.conv2d_transpose(upsamp2,filters=1,kernel_size=[s1sh[1]-2*s4sh[1]+1,s1sh[1]-2*s4sh[1]+1],activation=activation)
        upsamp3 = tf.add(upsamp3,tf.reshape(tf.reduce_max(slayer1,axis=-1),[-1,s1sh[1],s1sh[1],1]))
        upsamp3 = tf.contrib.layers.batch_norm(upsamp3,scale=True)
        upsamp4 = tf.layers.conv2d_transpose(upsamp3,filters=2,kernel_size=[int(sh[1]/2) + 1,int(sh[1]/2) + 1],activation=activation)
        upsamp4 = tf.contrib.layers.batch_norm(upsamp4,scale=True)
        final_conv = tf.reshape(upsamp4,[-1,sh[1]*sh[1],2])
        
    return final_conv

# Build Networks                                                                                                                                  
# Network Inputs                                                                                                                                  
vis_input = tf.placeholder(tf.float32, shape=[None, pad_size, pad_size, 2]) #this is a waterfall amp/phs visibility                               

# Build Generator Network                                                                                                                         
RFI_guess = FCN(vis_input)
bsl_RFI = tf.summary.image(name='Baseline Metrics',tensor=RFI_guess)

summary = tf.summary.merge_all()

RFI_targets = tf.placeholder(tf.int32, shape=[None, pad_size*pad_size])

loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=RFI_guess, labels=RFI_targets)
optimizer_gen = tf.train.AdamOptimizer(learning_rate=0.0001)
fcn_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='FCN')
train_fcn = optimizer_gen.minimize(loss, var_list=fcn_vars)

# Initialize the variables (i.e. assign their default value)                                                                                      
init = tf.global_variables_initializer()
# save variables                                                                                                                                  
saver = tf.train.Saver()


# Load dataset
dset = hf.RFIDataset()
dset.load(batch_size)

with tf.Session() as sess:

    summary_writer = tf.summary.FileWriter('./FCN/',sess.graph)
    # Run the initializer                                                                                                                         
    sess.run(init)
    #check to see if model exists                                                                                                                 
    if len(model_dir) > 0:
        print('Model exists. Loading last save.')
        saver.restore(sess, './FCN/'+model)
        print('Model '+str(model) + ' loaded.')
    else:
        print('No Model Found.')
    if mode == 'train':
        for i in range(start_step, start_step+num_steps+1):
            # Prepare Input Data                                                                                                                  
            batch_x, batch_targets = dset.next_train()
            # Training                                                                                                                           
            feed_dict = {vis_input: batch_x, RFI_targets: batch_targets}
            _,loss_ = sess.run([train_fcn,loss],feed_dict=feed_dict)
            #summ1 = sess.run(rflagsdelay, feed_dict=feed_dict)
            #summ2 = sess.run(gflagsdelay, feed_dict=feed_dict)
            #summary_writer.add_summary(summ1,i)
            #summary_writer.add_summary(summ2,i)
            if i % 100 == 0 or i == 1:
                print('Step %i: RFI Crushinator Loss: %.9f' % (i, np.mean(loss_)))
            if i % 1000 == 0:
                print('Saving model...')
                summary_writer.flush()
                save_path = saver.save(sess,'./FCN/model_%i.ckpt' % i)

    elif mode == 'eval':
        for i in range(1, num_steps+1):
            batch_x, batch_targets = dset.next_eval()
            feed_dict = {vis_input: batch_x, RFI_targets: batch_targets}
            eval_class = sess.run(RFI_guess,feed_dict=feed_dict)
            if i % 10 == 0:
                if 'acc' not in globals():
                    acc = 0.
                acc = hf.batch_accuracy(batch_targets,tf.argmax(eval_class,axis=-1)).eval()
                print('Batch accuracy %f' % acc)
    else:
        time0 = time()
        batch_x, batch_targets = dset.random_test(16*450)
        g = sess.run(RFI_guess, feed_dict={vis_input: batch_x})
        print('N=%i number of baselines time: ' % 450,time() - time0)

        plt.subplot(311)
        plt.imshow(batch_x[0,:,:,0],aspect='auto')
        plt.subplot(312)
        plt.imshow(batch_targets[0,:].reshape(68,68),aspect='auto')
        plt.colorbar()
        plt.subplot(313)
        plt.imshow(tf.reshape(tf.argmax(g[0,:,:],axis=1),[68,68]).eval(),aspect='auto')
        plt.colorbar()
        plt.savefig('VisClassify.pdf')

