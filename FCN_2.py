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
model_name = 'AmpSimBSize8'
chtypes='Amp'
num_steps = 50001
batch_size = 8
pad_size = 68
ch_input = 1
f_factor = 16
mode = 'train'
model_dir = glob("./"+model_name+"/model_*")

try:
    models2sort = [int(model_dir[i].split('/')[2].split('.')[0].split('_')[1]) for i in range(len(model_dir))]
    model_ind = np.argmax(models2sort)
    model = 'model_'+str(models2sort[model_ind])+'.ckpt'#model_dir[np.max(model_ind)].split('/')[2]+'.ckpt'
    start_step = int(model.split('_')[1].split('.ckpt')[0])
    print(model)
except:
    start_step = 0

print('Starting training at step %i' % start_step)

# Generator Network                                                                                                                               
# Input: Noise, Output: Image                                                                                                                     

def FCN(x, reuse=None, mode=True):
    with tf.variable_scope('FCN', reuse=reuse):
        kt = 3
        kf = 3
        s = 1
        activation = tf.nn.leaky_relu
        sh = x.get_shape().as_list()
        input_layer = tf.cast(tf.reshape(x,[-1,sh[1],sh[2],sh[3]]),dtype=tf.float32)
        tf.summary.image('IP_Amp',tf.reshape(input_layer[0,:,:,0],[1,pad_size,pad_size,1]))
#        tf.summary.image('IP_Phs',tf.reshape(input_layer[0,:,:,1],[1,pad_size,pad_size,1]))
        # Convolution / Downsampling layers
        slayer1 = hf.stacked_layer(input_layer,s*16,kt,kf,activation,[2,2],[2,2],bnorm=True,mode=mode_bn)
        s1sh = slayer1.get_shape().as_list()
        print(s1sh)
        tf.summary.image('S1',tf.reshape(tf.reduce_sum(slayer1[0,:,:,:],axis=-1),[1,int(pad_size/2),int(pad_size/2),1]))

        slayer2 = hf.stacked_layer(slayer1,s*32,kt,kf,activation,[2,2],[2,2],bnorm=True,mode=mode_bn)
        slayer3 = hf.stacked_layer(slayer2,s*64,kt,kf,activation,[2,2],[2,2],bnorm=True,mode=mode_bn)
        s3sh = slayer3.get_shape().as_list()
        slayer4 = hf.stacked_layer(slayer3,s*128,kt,kf,activation,[2,2],[2,2],bnorm=True,mode=mode_bn)
        slayer5 = hf.stacked_layer(slayer4,s*512,kt,kf,activation,[2,2],[2,2],bnorm=True,mode=mode_bn)
        slayer6 = hf.stacked_layer(slayer5,s*1024,kt,kf,activation,[2,2],[2,2],bnorm=True,mode=mode_bn)
        
        # Fully connected and convolutional layers
        slayer7 = hf.stacked_layer(slayer6,s*2048,1,1,activation,[1,1],[1,1],bnorm=True,dropout=True,mode=mode_bn)
        slayer8 = hf.stacked_layer(slayer7,s*2048,1,1,activation,[1,1],[1,1],bnorm=True,dropout=True,mode=mode_bn)
        
        # Transpose convolution layers
        upsamp1 = tf.layers.conv2d_transpose(slayer8,filters=256,kernel_size=[s3sh[1],s3sh[1]],activation=activation)
        upsamp1 = tf.add(upsamp1,tf.reshape(tf.reduce_max(slayer3,axis=-1),[-1,s3sh[1],s3sh[1],1]))
        upsamp1 = tf.layers.batch_normalization(upsamp1,scale=True,center=True,training=mode,fused=True)#tf.contrib.layers.batch_norm(upsamp1,scale=True)

        upsamp2 = tf.layers.conv2d_transpose(upsamp1,filters=128,kernel_size=[s3sh[1]+1,s3sh[1]+1],activation=activation)
        upsamp2 = tf.layers.batch_normalization(upsamp2,scale=True,center=True,training=mode_bn,fused=True)#tf.contrib.layers.batch_norm(upsamp2,scale=True)
        upsamp3 = tf.layers.conv2d_transpose(upsamp2,filters=64,kernel_size=[s1sh[1]-2*s3sh[1]+1,s1sh[1]-2*s3sh[1]+1],activation=activation)
        upsamp3 = tf.add(upsamp3,tf.reshape(tf.reduce_max(slayer1,axis=-1),[-1,s1sh[1],s1sh[1],1]))
        upsh3 = upsamp3.get_shape().as_list()
        upsamp3 = tf.layers.batch_normalization(upsamp3,scale=True,center=True,training=mode_bn,fused=True)#tf.contrib.layers.batch_norm(upsamp3,scale=True)
        tf.summary.image('Upsamp_3',tf.reshape(upsamp3[0,:,:,0],[1,int(pad_size/2),int(pad_size/2),1]))
        upsamp4 = tf.layers.conv2d_transpose(upsamp3,filters=2,kernel_size=[int(sh[1] - upsh3[1]) + 1,int(sh[1] - upsh3[1]) + 1],activation=activation)
        upsamp4 = tf.layers.batch_normalization(upsamp4,scale=True,center=True,training=mode_bn,fused=True)#tf.contrib.layers.batch_norm(upsamp4,scale=True)
        tf.summary.image('Flag Guess',tf.reshape(upsamp4[0,:,:,1],[1,pad_size,pad_size,1]))
        final_conv = tf.reshape(upsamp4,[-1,sh[1]*sh[1],2])
        
    return final_conv

# Build Networks                                                                                                                                  
# Network Inputs                                                                                                                                  
vis_input = tf.placeholder(tf.float32, shape=[None, pad_size, pad_size, ch_input]) #this is a waterfall amp/phs/comp visibility                               
mode_bn = tf.placeholder(tf.bool)
# Build Generator Network                                                                                                                         
RFI_guess = FCN(vis_input,mode=mode_bn)
#bsl_RFI = tf.summary.image(name='Baseline Metrics',tensor=RFI_guess)


RFI_targets = tf.placeholder(tf.int32, shape=[None, pad_size*pad_size])
learn_rate = tf.placeholder(tf.float32, shape=[1])

#hthresh = hf.hard_thresh(RFI_guess)
argmax = tf.argmax(RFI_guess,axis=-1)

recall = tf.metrics.recall(labels=RFI_targets,predictions=argmax)
precision = tf.metrics.precision(labels=RFI_targets,predictions=argmax)
batch_accuracy = hf.batch_accuracy(RFI_targets,argmax)
#accuracy = tf.metrics.accuracy(labels=RFI_targets,predictions=tf.argmax(RFI_guess,axis=-1))
f1 = 2.*precision[0]*recall[0]/(precision[0]+recall[0])
f1 = tf.where(tf.is_nan(f1),tf.zeros_like(f1),f1)
#loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=RFI_targets,logits=RFI_guess)) + 100.*(1-f1)
loss = tf.losses.sparse_softmax_cross_entropy(labels=RFI_targets,logits=RFI_guess)#s,weights=1./batch_accuracy)
#loss = tf.losses.softmax_cross_entropy(onehot_labels=RFI_targets,logits=argmax,weights=1.-f1)

tf.summary.scalar('loss',loss)
tf.summary.scalar('recall',recall[0])
tf.summary.scalar('precision',precision[0])
tf.summary.scalar('F1',f1)
tf.summary.scalar('batch_accuracy',batch_accuracy)
summary = tf.summary.merge_all()
optimizer_gen = tf.train.AdamOptimizer(learning_rate=learn_rate[0])
#optimizer_gen = tf.train.MomentumOptimizer(learning_rate=0.01,momentum=0.01)
#optimizer_gen = tf.train.AdagradOptimizer(learning_rate=learn_rate[0])
fcn_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='FCN')
train_fcn = optimizer_gen.minimize(loss, var_list=fcn_vars)

# Initialize the variables (i.e. assign their default value)                                                                                      
init = tf.group(tf.global_variables_initializer(),tf.local_variables_initializer())
# save variables                                                                                                                                  
saver = tf.train.Saver()


# Load dataset
dset = hf.RFIDataset()
dset.load(batch_size,pad_size,hybrid=False,chtypes=chtypes,fold_factor=f_factor)

with tf.Session() as sess:

    
    # Run the initializer                                                                                                                         
    sess.run(init)
    #check to see if model exists                                                                                                                 
    if len(model_dir) > 0:
        print('Model exists. Loading last save.')
        saver.restore(sess, './'+model_name+'/'+model)
        print('Model '+'./'+model_name+'/'+model + ' loaded.')
    else:
        print('No Model Found.')
    if mode == 'train':
        train_writer = tf.summary.FileWriter('./'+model_name+'_train/',sess.graph)
        lr = np.array([0.003])
        for i in range(start_step, start_step+num_steps+1):
            # Prepare Input Data                                                                                                                  
            batch_x, batch_targets = dset.next_train()
            # Training                                                                                                                           
            feed_dict = {vis_input: batch_x, RFI_targets: batch_targets,
                         learn_rate: lr, mode_bn: True}
            _,loss_,s1,rec,pre,f1_,ba = sess.run([train_fcn,loss,summary,recall,precision,f1,batch_accuracy],feed_dict=feed_dict)
            #s1 = sess.run(s1image, feed_dict=feed_dict)
            #up3 = sess.run(up3image, feed_dict=feed_dict)
            if i % 20 == 0:
                train_writer.add_summary(s1,i)
                train_writer.flush()
                #lr *= .98
                #summary_writer.add_summary(up3,i)
            if i % 100 == 0 or i == 1:
                print('Step %i: RFI Crushinator Loss: %.9f' % (i, loss_))
                print('Recall : %.9f' % rec[0])
                print('Precision : %.9f' % pre[0])
                print('F1 : %.9f' % f1_)
                print('Batch Accuracy : %.4f' % ba)
            if i % 1000 == 0 and i != 0:
                print('Saving model...')
                save_path = saver.save(sess,'./'+model_name+'/model_%i.ckpt' % i)
#            if i % 10000 == 0:
#                lr *= 0.1
#                print('Learning rate decreased to %f.' % lr)
    elif mode == 'eval':
        eval_writer = tf.summary.FileWriter('./'+model_name+'_eval/',sess.graph)
        for i in range(1, num_steps+1):
            batch_x, batch_targets = dset.next_eval()
            feed_dict = {vis_input: batch_x, RFI_targets: batch_targets, mode_bn: True}
            eval_class,rec,pre,f1_,s1 = sess.run([RFI_guess,recall,precision,f1,summary],feed_dict=feed_dict)
            if i % 10 == 0:
                if 'acc' not in globals():
                    acc = 0.
                print('F1 %f' % f1_)
                acc = hf.batch_accuracy(batch_targets,tf.argmax(eval_class,axis=-1)).eval()
                print('Batch accuracy %f' % acc)
            if i % 20 == 0:
                eval_writer.add_summary(s1,i)
                eval_writer.flush()
            #if i % 1000 == 0:
            #    save_path = saver.save(sess,'./'+model_name+'_eval'+'/model_%i.ckpt' % i)
    else:
        time0 = time()
        ind = 0
        batch_x, batch_targets = dset.random_test(16)
        g = sess.run(RFI_guess, feed_dict={vis_input: batch_x, mode_bn: True})
        print('N=%i number of baselines time: ' % 1,time() - time0)
        for ind in range(16):
            acc = hf.accuracy(batch_targets[ind,:],tf.reshape(tf.argmax(g[ind,:,:],axis=-1),[-1]).eval())
            print('Accuracy compared to XRFI flags: %f'% acc)
            plt.figure()
            plt.subplot(321)
            plt.imshow(hf.unpad(batch_x[ind,:,:,0]),aspect='auto')
            plt.title('Amplitude')
            plt.colorbar()
            plt.subplot(322)
            plt.imshow(hf.unpad(batch_x[ind,:,:,0]),aspect='auto')
            plt.title('Phase')
            plt.colorbar()
            
            plt.subplot(323)
            plt.imshow(hf.unpad(batch_targets[ind,:].reshape(pad_size,pad_size)),aspect='auto')
            plt.title('Targets')
            plt.colorbar()
            plt.subplot(324)
            plt.imshow(hf.unpad(tf.reshape(tf.argmax(g[ind,:,:],axis=-1),[pad_size,pad_size]).eval()),aspect='auto')
            plt.text(512,50,s=str(acc))
            plt.title('Argmax')
            plt.colorbar()
            plt.subplot(313)
            plt.imshow(hf.unpad(tf.reshape(hf.hard_thresh(g[ind,:,:]),[pad_size,pad_size]).eval()),aspect='auto')
            plt.title('Hard Threshold')
            plt.tight_layout()
            plt.savefig(model_name+'_%i.pdf'%ind)
            plt.clf()
