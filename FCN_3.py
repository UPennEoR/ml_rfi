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
import sys

# Run on a single GPU
os.environ["CUDA_VISIBLE_DEVICES"]="1"
args = sys.argv[1:]
print(bool(args[5]),'Arg 5')
# Training Params
tdset_version = args[0]#'v4'      # which training dataset version to use
FCN_version = args[1]#'v6'        # which FCN version number
tdset_type = 'Sim'        # type of training dataset used
edset_type = 'Real'       # type of eval dataset used
mods = '_PostFCStitch_layerNorm' # additional modifiers that have been applied
num_steps = 10000
batch_size = int(args[2])#256
pad_size = 68
ch_input = int(args[3])#2
mode = args[4]#'eval'
hybrid=bool(args[5])#True
chtypes=args[6]#'AmpPhs'
dropout=np.float32(args[7])
model_name = chtypes+FCN_version+tdset_type+edset_type+tdset_version+'_'+str(batch_size)+'BSize'+mods+'DOUT'+str(dropout)
model_dir = glob("./"+model_name+"/model_*")
plot = False
ROC = True
if hybrid:
    cut = True
    f_factor = 12
else:
    cut = False
    f_factor = 16

try:
    models2sort = [int(model_dir[i].split('/')[2].split('.')[0].split('_')[1]) for i in range(len(model_dir))]
    model_ind = np.argmax(models2sort)
    model = 'model_'+str(models2sort[model_ind])+'.ckpt' #model_dir[np.max(model_ind)].split('/')[2]+'.ckpt'
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
        sh = x.get_shape().as_list()
        if sh[3] > 1.:
            s = 1 #4 for Amp, 2 for AmpPhs
                  # 2xDeep 8 for Amp, 4 for AmpPhs
        else:
            s = 1
        activation = tf.nn.leaky_relu
        input_layer = tf.cast(tf.reshape(x,[-1,sh[1],sh[2],sh[3]]),dtype=tf.float32)
        tf.summary.image('IP_Amp',tf.reshape(input_layer[0,:,:,0],[1,pad_size,pad_size,1]))
        if sh[3] > 1:
            slayer1 = hf.stacked_layer(tf.reshape(input_layer[:,:,:,0],[-1,sh[1],sh[2],1]),4*s,kt,kf,activation,[2,2],[2,2],bnorm=True,mode=mode_bn)
            #slayer2a = hf.stacked_layer(slayer1a,4*s,kt,kf,activation,[2,2],[2,2],bnorm=True,mode=mode_bn)
            slayer1b = hf.stacked_layer(tf.reshape(input_layer[:,:,:,1:],[-1,sh[1],sh[2],sh[3]-1]),4*s,kt,kf,activation,[2,2],[2,2],bnorm=True,mode=mode_bn,maxpool=True)
            slayer2b = hf.stacked_layer(slayer1b,8*s,kt,kf,activation,[2,2],[2,2],bnorm=True,mode=mode_bn,maxpool=True)
            slayer3b = hf.stacked_layer(slayer2b,16*s,kt,kf,activation,[2,2],[2,2],bnorm=True,mode=mode_bn,maxpool=True,dropout=d_out)
            slayer4b = tf.layers.dropout(hf.stacked_layer(slayer3b,32*s,kt,kf,activation,[2,2],[2,2],bnorm=True,mode=mode_bn,maxpool=True),rate=0.)
            slayer5b = hf.stacked_layer(slayer4b,64*s,kt,kf,activation,[2,2],[2,2],bnorm=True,mode=mode_bn,maxpool=True)
            slayer6b = hf.stacked_layer(slayer5b,128*s,kt,kf,activation,[2,2],[2,2],bnorm=True,mode=mode_bn,maxpool=True)
            s3sh = slayer3b.get_shape().as_list()
            s6sh = slayer6b.get_shape().as_list()
            slayer7b = hf.stacked_layer(slayer6b,s*512,1,1,activation,[1,1],[1,1],bnorm=True,dropout=d_out,mode=mode_bn)
            
#            upsamp1b = tf.layers.conv2d_transpose(slayer7b,filters=s*128,kernel_size=[s6sh[1],s6sh[1]],activation=activation)
            #upsamp1b = tf.add(upsamp1b,slayer6b)
#            upsamp1b = tf.layers.dropout(tf.layers.batch_normalization(upsamp1b,scale=True,center=True,training=mode,fused=True),rate=0.)
#            upsamp2b = tf.layers.conv2d_transpose(upsamp1b,filters=s*64,kernel_size=[s6sh[1]+1,s6sh[1]+1],activation=activation)
#            upsamp2b = tf.layers.dropout(tf.layers.batch_normalization(upsamp2b,scale=True,center=True,training=mode_bn,fused=True),rate=0.)
#            upsamp3b = tf.layers.conv2d_transpose(upsamp2b,filters=s*16,kernel_size=[s3sh[1]-2*s6sh[1]+1,s3sh[1]-2*s6sh[1]+1],activation=activation)
            #upsamp3b = tf.add(upsamp3b,slayer3b)
#            upsh3 = upsamp3b.get_shape().as_list()
#            upsamp4b = tf.layers.conv2d_transpose(upsamp3b,filters=2,kernel_size=[int(sh[1] - upsh3[1]) + 1,int(sh[1] - upsh3[1]) + 1],activation=activation)
#            upsamp4b = tf.layers.dropout(tf.layers.batch_normalization(upsamp4b,scale=True,center=True,training=mode_bn,fused=True),rate=0.)
#            final_convb = tf.reshape(upsamp4b,[-1,sh[1]*sh[1],1])
#            s1sh = slayer1.get_shape().as_list()
            tf.summary.image('Amp_l1',tf.reshape(tf.reduce_max(slayer1[0,:,:,:],axis=-1),[1,int(pad_size/2),int(pad_size/2),1]))
            tf.summary.image('Phs_l1',tf.reshape(tf.reduce_max(slayer1b[0,:,:,:],axis=-1),[1,int(pad_size/2),int(pad_size/2),1]))
            
        else:
            slayer1 = hf.stacked_layer(input_layer,s*4,kt,kf,activation,[2,2],[2,2],bnorm=True,mode=mode_bn)
            s1sh = slayer1.get_shape().as_list()
            tf.summary.image('S1',tf.reshape(tf.reduce_max(slayer1[0,:,:,:],axis=-1),[1,int(pad_size/2),int(pad_size/2),1]))
        
        slayer2 = hf.stacked_layer(slayer1,s*8,kt,kf,activation,[2,2],[2,2],bnorm=True,mode=mode_bn)
        
        slayer3 = hf.stacked_layer(slayer2,s*16,kt,kf,activation,[2,2],[2,2],bnorm=True,mode=mode_bn,dropout=d_out)

        s3sh = slayer3.get_shape().as_list()
        slayer4 = tf.layers.dropout(hf.stacked_layer(slayer3,s*32,kt,kf,activation,[2,2],[2,2],bnorm=True,mode=mode_bn),rate=0.)
        
        slayer5 = hf.stacked_layer(slayer4,s*64,kt,kf,activation,[2,2],[2,2],bnorm=True,mode=mode_bn)
        
        slayer6 = hf.stacked_layer(slayer5,s*128,kt,kf,activation,[2,2],[2,2],bnorm=True,mode=mode_bn)
        s6sh = slayer6.get_shape().as_list()
        # Fully connected and convolutional layers
#        if sh[3] > 1.:
#            slayer6 = tf.layers.dropout(tf.reduce_mean([slayer6,slayer6b],axis=0),rate=0.5)

        slayer7 = hf.stacked_layer(slayer6,s*512,1,1,activation,[1,1],[1,1],bnorm=True,dropout=d_out,mode=mode_bn)
        
        # Transpose convolution layers
        if sh[3] > 1.:
            slayer7 = tf.concat([
                hf.tfnormalize(slayer7),hf.tfnormalize(slayer7b)],axis=-1)
            
        upsamp1 = tf.layers.conv2d_transpose(slayer7,filters=128,kernel_size=[s6sh[1],s6sh[1]],activation=activation)
#        upsamp1 = tf.layers.batch_normalization(upsamp1,scale=True,center=True,training=mode,fused=True)       
        #upsamp1 = tf.layers.dropout(tf.add(slayer6b,slayer6),rate=0.)
        upsamp1 = tf.layers.dropout(tf.layers.batch_normalization(upsamp1,scale=True,center=True,training=mode,fused=True),rate=0.)#tf.contrib.layers.batch_norm(upsamp1,scale=True)
#        upsamp1 = tf.layers.dropout(upsamp1, rate=.8)
        if sh[3] > 1.:
            upsamp1 = tf.concat([hf.tfnormalize(upsamp1),hf.tfnormalize(slayer6),hf.tfnormalize(slayer6b)],axis=-1)#tf.layers.dropout(tf.add(upsamp1,slayer),rate=0.)
        else:
            upsamp1 = tf.concat([hf.tfnormalize(upsamp1),hf.tfnormalize(slayer6)],axis=-1)
        upsamp2 = tf.layers.conv2d_transpose(upsamp1,filters=64,kernel_size=[s6sh[1]+1,s6sh[1]+1],activation=activation)
        upsamp2 = tf.layers.dropout(tf.layers.batch_normalization(upsamp2,scale=True,center=True,training=mode_bn,fused=True),rate=0.)#tf.contrib.layers.batch_norm(upsamp2,scale=True)
#        upsamp2 = tf.contrib.layers.layer_norm(upsamp2)
        if sh[3] > 1.:
            upsamp2 = tf.concat([
                hf.tfnormalize(upsamp2),
                hf.tfnormalize(slayer5),
                hf.tfnormalize(slayer5b)],axis=-1)#tf.layers.dropout(tf.add(upsamp2,upsamp2b),rate=0.)
        else:
            upsamp2 = tf.concat([hf.tfnormalize(upsamp2),hf.tfnormalize(slayer5)],axis=-1)
        upsamp3 = tf.layers.conv2d_transpose(upsamp2,filters=16,kernel_size=[s3sh[1]-2*s6sh[1]+1,s3sh[1]-2*s6sh[1]+1],activation=activation)
#        upsamp3 = tf.layers.batch_normalization(upsamp3,scale=True,center=True,training=mode,fused=True)
#        if sh[3] > 1.:
#            upsamp3 = tf.reduce_mean([tf.add(upsamp3,slayer3),slayer3b],axis=0)
#        else:
        #upsamp3 = tf.layers.dropout(tf.add(upsamp3,slayer3),rate=0.5)
        upsh3 = upsamp3.get_shape().as_list()
        upsamp3 = tf.layers.dropout(tf.layers.batch_normalization(upsamp3,scale=True,center=True,training=mode_bn,fused=True),rate=0.)#tf.contrib.layers.batch_norm(upsamp3,scale=True)
        if sh[3] > 1.:
            upsamp3 = tf.concat([
                hf.tfnormalize(upsamp3),
                hf.tfnormalize(slayer3),
                hf.tfnormalize(slayer3b)],axis=-1)#tf.layers.dropout(tf.add(upsamp3,upsamp3b),rate=0.)
        else:
            upsamp3 = tf.concat([hf.tfnormalize(upsamp3),hf.tfnormalize(slayer3)],axis=-1)
        #        upsamp3 = tf.layers.dropout(upsamp3, rate=.8)
        #tf.summary.image('Upsamp_3',tf.reshape(upsamp3[5,:,:,0],[1,int(pad_size/2),int(pad_size/2),1]))
        if sh[3] == 1:
            out_filter = 2
        else:
            out_filter = 2
        upsamp4 = tf.layers.conv2d_transpose(upsamp3,filters=out_filter,kernel_size=[int(sh[1] - upsh3[1]) + 1,int(sh[1] - upsh3[1]) + 1],activation=activation)
        upsamp4 = tf.layers.dropout(tf.layers.batch_normalization(upsamp4,scale=True,center=True,training=mode_bn,fused=True),rate=0.)#tf.contrib.layers.batch_norm(upsamp4,scale=True)
#        upsamp4 = tf.nn.l2_normalize(upsamp4,axis=[1,2,3])
        #upsamp4 = tf.layers.dropout(tf.add(upsamp4,upsamp4b),rate=0.)
        tf.summary.image('Flag Guess',tf.reshape(upsamp4[0,:,:,1],[1,pad_size,pad_size,1]))
#        if sh[3] == 1:
        final_conv = tf.reshape(upsamp4,[-1,sh[1]*sh[1],2])
#        else:
#            final_conv = tf.reshape(upsamp4,[-1,sh[1]*sh[1],1])
#            final_conv = tf.concat([final_conv,final_convb],axis=-1)
        print(np.shape(final_conv))
    return final_conv

# Build Networks                                                                                                                                  
# Network Inputs                                                                                                                                  
vis_input = tf.placeholder(tf.float32, shape=[None, pad_size, pad_size, ch_input]) #this is a waterfall amp/phs/comp visibility                               
mode_bn = tf.placeholder(tf.bool)
d_out = tf.placeholder(tf.float32)
# Build Generator Network                                                                                                                         
RFI_guess = FCN(vis_input,mode=mode_bn)
#bsl_RFI = tf.summary.image(name='Baseline Metrics',tensor=RFI_guess)


RFI_targets = tf.placeholder(tf.int32, shape=[None, pad_size*pad_size])
learn_rate = tf.placeholder(tf.float32, shape=[1])

#hthresh = hf.hard_thresh(RFI_guess)
argmax = tf.argmax(RFI_guess,axis=-1)

recall = tf.metrics.recall(labels=RFI_targets,predictions=argmax) #aka True Pos. Rate
precision = tf.metrics.precision(labels=RFI_targets,predictions=argmax)
batch_accuracy = hf.batch_accuracy(RFI_targets,argmax)
#fpr = tf.metrics.false_positives(RFI_targets,argmax)
#accuracy = tf.metrics.accuracy(labels=RFI_targets,predictions=tf.argmax(RFI_guess,axis=-1))
f1 = 2.*precision[0]*recall[0]/(precision[0]+recall[0])
f1 = tf.where(tf.is_nan(f1),tf.zeros_like(f1),f1)
#loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=RFI_targets,logits=RFI_guess)) + 100.*(1-f1)
loss = tf.losses.sparse_softmax_cross_entropy(labels=RFI_targets,logits=RFI_guess)#s,weights=1./batch_accuracy)
#loss = tf.losses.softmax_cross_entropy(onehot_labels=RFI_targets,logits=argmax,weights=1.-f1)

tf.summary.scalar('loss',loss)
#tf.summary.scalar('False Positive Rate',fpr[0])
tf.summary.scalar('recall',recall[0])
tf.summary.scalar('precision',precision[0])
tf.summary.scalar('F1',f1)
tf.summary.scalar('batch_accuracy',batch_accuracy)
#tf.summary.scalar('AUC',auc[0])
summary = tf.summary.merge_all()
optimizer_gen = tf.train.AdamOptimizer(learning_rate=learn_rate[0])
fcn_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='FCN')
train_fcn = optimizer_gen.minimize(loss, var_list=fcn_vars)

# Initialize the variables (i.e. assign their default value)                                                                                      
init = tf.group(tf.global_variables_initializer(),tf.local_variables_initializer())
# save variables                                                                                                                                  
saver = tf.train.Saver()


# Load dataset
dset = hf.RFIDataset()
dset.load(tdset_version,batch_size,pad_size,hybrid=hybrid,chtypes=chtypes,fold_factor=f_factor,cut=cut)

#fpr_arr = []
#tpr_arr = []

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
                         learn_rate: lr, mode_bn: True, d_out: dropout}
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
                #lr *= .98
                #print('Learning rate decreased to %f.' % lr)
                print('Saving model...')
                save_path = saver.save(sess,'./'+model_name+'/model_%i.ckpt' % i)
            #if i % 10000 == 0:
                #lr *= 0.1
                #print('Learning rate decreased to %f.' % lr)
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
#                print('False Positive Rate %f' % fpr_[0])
                acc = hf.batch_accuracy(batch_targets,tf.argmax(eval_class,axis=-1)).eval()
                print('Batch accuracy %f' % acc)
            if i % 20 == 0:
                eval_writer.add_summary(s1,i)
                eval_writer.flush()
            #if i % 1000 == 0:
            #    save_path = saver.save(sess,'./'+model_name+'_eval'+'/model_%i.ckpt' % i)
    elif mode == 'traineval':
        lr = np.array([0.003])
        for i in range(start_step, start_step+num_steps+1):
            batch_x_train, batch_targets_train = dset.next_train()
            feed_dict_train = {vis_input: batch_x_train, RFI_targets: batch_targets_train,
                                                  learn_rate: lr, mode_bn: True}
            _,loss_,s1,rec,pre,f1_train,ba = sess.run([train_fcn,loss,summary,recall,precision,f1,batch_accuracy],feed_dict=feed_dict_train)

            batch_x_eval, batch_targets_eval = dset.next_eval()
            feed_dict_eval = {vis_input: batch_x_eval, RFI_targets: batch_targets_eval, mode_bn: True}
            eval_class,rec,pre,f1_eval,s1 = sess.run([RFI_guess,recall,precision,f1,summary],feed_dict=feed_dict_eval)

            if i % 100 == 0 or i == 1:
                print('Train F1 : %.9f' % f1_train)
                print('Eval F1 : %.9f' % f1_eval)
    else:
        time0 = time()
        ind = 0
        batch_x, batch_targets = dset.random_test(800)
        g = sess.run(RFI_guess, feed_dict={vis_input: batch_x, mode_bn: True})
        print('N=%i number of baselines time: ' % 1,time() - time0)
        ct = 0
        ind = 0
        while ct != 100:
            print(ct)
#        for ind in range(1000):
            acc = hf.accuracy(batch_targets[ind,:],tf.reshape(tf.argmax(g[ind,:,:],axis=-1),[-1]).eval())
            print('Accuracy compared to XRFI flags: %f'% acc)
            if plot:
                plt.figure()
                plt.subplot(321)
                plt.imshow(hf.unpad(batch_x[ind,:,:,0]),aspect='auto')
                plt.title('Amplitude')
                plt.colorbar()
                plt.subplot(322)
                if chtypes == 'AmpPhs':
                    plt.imshow(hf.unpad(batch_x[ind,:,:,1]),aspect='auto')
                else:
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
            
#        for ind in range(200):
            y_true = hf.unpad(batch_targets[ind,:].reshape(pad_size,pad_size)).reshape(-1)
            y_pred = hf.unpad(g[ind,:,1].reshape(pad_size,pad_size)).reshape(-1)
            
            fpr,tpr,_ = roc_curve(y_true,y_pred,drop_intermediate=False)
            print(np.shape(fpr))
            if ct==0 and np.size(fpr) == 60*64:
                fpr_arr = np.array(fpr).reshape(-1)
                tpr_arr = np.array(tpr).reshape(-1)
                ct+=1
            elif np.size(fpr) == 60*64:
                fpr_arr = np.vstack((fpr_arr,np.array(fpr).reshape(-1)))
                tpr_arr = np.vstack((tpr_arr,np.array(tpr).reshape(-1)))
                ct+=1
            ind+=1
        print(np.nanmean(fpr_arr,0))
        print(np.nanmean(tpr_arr,0))

        if ROC:
            import h5py
            #        try:
            f = h5py.File('ROC_curves.h5','a')
            try:
                mname = f.create_group(model_name)
                mname.create_dataset('FPR',data=np.nanmean(fpr_arr,0))
                mname.create_dataset('TPR',data=np.nanmean(tpr_arr,0))
                mname.create_dataset('FPRstd',data=np.nanstd(fpr_arr,0))
                mname.create_dataset('TPRstd',data=np.nanstd(tpr_arr,0))
                f.close()
            except:
                print('Data group already exists.')
                mname = f.require_group(model_name)
                mname['FPR'][...] = np.nanmean(fpr_arr,0)
                mname['TPR'][...] = np.nanmean(tpr_arr,0)
                mname['FPRstd'][...] = np.nanstd(fpr_arr,0)
                mname['TPRstd'][...] = np.nanstd(tpr_arr,0)
                f.close()
