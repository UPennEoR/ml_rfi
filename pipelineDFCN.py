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
from AmpModel import AmpFCN
from AmpPhsModel import AmpPhsFCN

# Run on a single GPU
os.environ["CUDA_VISIBLE_DEVICES"]="1"
args = sys.argv[1:]

chtypes = 'AmpPhs'
FCN_version = 'v100'
tdset_type = 'v00'
edset_type = 'uv'
tdset_version = 'v00'
mods = 'test'
pad_size = 68
model_name = chtypes+FCN_version+tdset_type+edset_type+tdset_version+'_'+'64'+'BSize'+mods
model_dir = glob("./"+model_name+"/model_*")

try:
    models2sort = [int(model_dir[i].split('/')[2].split('.')[0].split('_')[1]) for i in range(len(model_dir))]
    model_ind = np.argmax(models2sort)
    model = 'model_'+str(models2sort[model_ind])+'.ckpt'
    print(model)
except:
    print('Cannot find model.')

vis_input = tf.placeholder(tf.float32, shape=[None, pad_size, pad_size, ch_input]) #this is a waterfall amp/phs/comp visibility      
mode_bn = tf.placeholder(tf.bool)
d_out = tf.placeholder(tf.float32)
kernel_size = tf.placeholder(tf.int32)

# Initialize Network
if chtypes == 'Amp':
    RFI_guess = AmpFCN(vis_input,mode_bn=mode_bn,d_out=d_out)
elif chtypes == 'AmpPhs':
    RFI_guess = AmpPhsFCN(vis_input,mode_bn=mode_bn,d_out=d_out)

optimizer_gen = tf.train.AdamOptimizer(learning_rate=learn_rate[0])
fcn_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='FCN')

# Initialize the variables (i.e. assign their default value)                                                                                      
init = tf.group(tf.global_variables_initializer(),tf.local_variables_initializer())

# Load dataset
dset = hf.RFIDataset()
dset_start_time = time()
dset.load_pyuvdata()
#dset.load(tdset_version,vdset,batch_size,pad_size,chtypes=chtypes)
dset_load_time = (time() - dset_start_time)/dset.get_size() # per visibility

with tf.Session() as sess:    
    # Run the initializer                                                                                                                         
    sess.run(init)
    # Check to see if model exists                                                                                                                 
    if len(model_dir) > 0:
        print('Model exists. Loading last save.')
        saver.restore(sess, './'+model_name+'/'+model)
        print('Model '+'./'+model_name+'/'+model + ' loaded.')
    else:
        raise ValueError("No Model Found. Pipeline killed.")        

    time0 = time()
    ind = 0
    print('N=%i number of baselines time: ' % 1,time() - time0)
    # Cut off band edges, it's in factors of 64
    # Low: 1 and High: 1 is approx 13% of the band
    while ct < 10:
        batch_x = dset.predict_pyuvdata()
        pred_start = time()
        g = sess.run(RFI_guess, feed_dict={vis_input: batch_x, mode_bn: True})
        print('Current Visibility: {0}'.format(ct))            
        pred_unfold = hf.unfoldl(tf.reshape(tf.argmax(g,axis=-1),[16,68,68]).eval())
        #pred_unfold = hf.unfoldl(tf.reshape(g[:,:,1],[16,68,68]).eval())
        pred_time = time() - pred_start
        if chtypes == 'AmpPhs':
            thresh = 0.62 #0.329 real #0.08 sim 
        else:
            thresh = 0.385 #0.385 real #0.126 sim
        y_pred = pred_unfold[:,64*ci_1:1024-64*ci_2].reshape(-1)
#       y_pred = hf.hard_thresh(pred_unfold[:,64*ci_1:1024-64*ci_2],thresh=thresh).reshape(-1)

