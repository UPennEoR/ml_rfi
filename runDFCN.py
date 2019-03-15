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
os.environ["CUDA_VISIBLE_DEVICES"]="0"
config = tf.ConfigProto()
config.gpu_options.allow_growth = True

args = sys.argv[1:]

stats = False
waterfall_sample = False
ROC = False

# Training Params
dropout=np.float32(args[7])
global ksize
ksize=int(args[8])           # kernel size is for PHASE C-layers only !!!
tdset_version = args[0]      # which training dataset version to use
FCN_version = args[1]        # which FCN version number
try:
    vdset = args[10]
except:
    vdset = ''
tdset_type = 'Sim'        # type of training dataset used
edset_type = 'Real'       # type of eval dataset used
#mods = 'New'
#mods = '_ExpandedDataset_Softmax_1x_DOUT'+str(dropout)+'_Converge_teval'
mods = 'DynamicVis'
num_steps = int(args[9])
batch_size = int(args[2])
pad_size = 16 #68
ch_input = int(args[3])
mode = args[4]
expand = True
patchwise_train = False #np.logical_not(bool(args[5]))
hybrid=bool(args[5])
chtypes=args[6]
model_name = chtypes+FCN_version+tdset_type+edset_type+tdset_version+'_'+'64'+'BSize'+mods
model_dir = glob("./"+model_name+"/model_*")
if hybrid:
    cut = False
    f_factor = 16
else:
    cut = False
    f_factor = 16#16

try:
    models2sort = [int(model_dir[i].split('/')[2].split('.')[0].split('_')[1]) for i in range(len(model_dir))]
    model_ind = np.argmax(models2sort)
    model = 'model_'+str(models2sort[model_ind])+'.ckpt'
    start_step = int(model.split('_')[1].split('.ckpt')[0])
    print(model)
except:
    start_step = 0

print('Starting training at step %i' % start_step)

vis_input = tf.placeholder(tf.float32, shape=[None,None,None,ch_input])#shape=[None, 2*(pad_size+2)+60, 2*pad_size+1024/f_factor, ch_input]) #this is a waterfall amp/phs/comp visibility
mode_bn = tf.placeholder(tf.bool)
d_out = tf.placeholder(tf.float32)
kernel_size = tf.placeholder(tf.int32)

# Initialize Network
if chtypes == 'Amp':
    RFI_guess = AmpFCN(vis_input,mode_bn=mode_bn,d_out=d_out)
elif chtypes == 'AmpPhs':
    RFI_guess = AmpPhsFCN(vis_input,mode_bn=mode_bn,d_out=d_out)
RFI_targets = tf.placeholder(tf.int32, shape=[None,None])#(2*(pad_size+2)+60)*(2*pad_size+1024/f_factor)])
learn_rate = tf.placeholder(tf.float32, shape=[1])

# Output statistics and metrics
argmax = tf.argmax(RFI_guess,axis=-1)
recall = tf.metrics.recall(labels=RFI_targets,predictions=argmax)
precision = tf.metrics.precision(labels=RFI_targets,predictions=argmax)
batch_accuracy = hf.batch_accuracy(RFI_targets,argmax)
f1 = 2.*precision[0]*recall[0]/(precision[0]+recall[0])
f1 = tf.where(tf.is_nan(f1),tf.zeros_like(f1),f1)
loss = tf.losses.sparse_softmax_cross_entropy(labels=RFI_targets,logits=RFI_guess)

# Add metrics to summary
tf.summary.scalar('loss',loss)
tf.summary.scalar('recall',recall[0])
tf.summary.scalar('precision',precision[0])
tf.summary.scalar('F1',f1)
tf.summary.scalar('batch_accuracy',batch_accuracy)
summary = tf.summary.merge_all()

optimizer_gen = tf.train.AdamOptimizer(learning_rate=learn_rate[0])
fcn_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='FCN')
train_fcn = optimizer_gen.minimize(loss, var_list=fcn_vars)

# Initialize the variables (i.e. assign their default value)                                                                                      
init = tf.group(tf.global_variables_initializer(),tf.local_variables_initializer())
# Save variables                                                                                                                                  
saver = tf.train.Saver()

# Load dataset
dset = hf.RFIDataset()
dset_start_time = time()
dset.load(tdset_version,vdset,batch_size,pad_size,hybrid=hybrid,chtypes=chtypes,fold_factor=f_factor,cut=cut,patchwise_train=patchwise_train,expand=expand)
dset_load_time = (time() - dset_start_time)/dset.get_size() # per visibility

with tf.Session(config=config) as sess:    
    # Run the initializer                                                                                                                         
    sess.run(init)
    # Check to see if model exists                                                                                                                 
    if len(model_dir) > 0:
        print('Model exists. Loading last save.')
        saver.restore(sess, './'+model_name+'/'+model)
        print('Model '+'./'+model_name+'/'+model + ' loaded.')
    else:
        print('No Model Found.')
    if mode == 'train':
        # Run training only session
        train_writer = tf.summary.FileWriter('./'+model_name+'_train/',sess.graph)
        lr = np.array([0.003])
        for i in range(start_step, start_step+num_steps+1):
            # Prepare Input Data                                                                                                                  
            batch_x, batch_targets = dset.next_train()
            # Training                                                                                                                           
            feed_dict = {vis_input: batch_x, RFI_targets: batch_targets,
                         learn_rate: lr, mode_bn: True, d_out: dropout}
            _,loss_,s1,rec,pre,f1_,ba = sess.run([train_fcn,loss,summary,recall,precision,f1,batch_accuracy],feed_dict=feed_dict)
            if i % 20 == 0:
                # Save metrics every 20 steps
                train_writer.add_summary(s1,i)
                train_writer.flush()
            if i % 100 == 0 or i == 1:
                # Output to terminal every 100 steps
                print('Step {0}: Loss: {1}'.format(i, loss_))
                print('Recall: {0}'.format(rec[0]))
                print('Precision: {0}'.format(pre[0]))
                print('F1: {0}'.format(f1_))
                print('RFI Class Accuracy: {0}'.format(ba))
            if i % 1000 == 0 and i != 0:
                # Save model every 1000 steps
                print('Saving model...')
                save_path = saver.save(sess,'./'+model_name+'/model_%i.ckpt' % i)
    elif mode == 'eval':
        # Run evaluation only session
        eval_writer = tf.summary.FileWriter('./'+model_name+'_eval_'+vdset+'/',sess.graph)
        for i in range(start_step, start_step+num_steps+1):
            batch_x, batch_targets = dset.next_eval()
            feed_dict = {vis_input: batch_x, RFI_targets: batch_targets, mode_bn: True}
            eval_class,rec,pre,f1_,s1,loss_ = sess.run([RFI_guess,recall,precision,f1,summary,loss],feed_dict=feed_dict)
            print('recall: {0} precision: {1} f1: {2} '.format(rec,pre,f1_))
            if i % 10 == 0:
                # Output to terminal every 10 steps
                if 'acc' not in globals():
                    acc = 0.
                print('F1: {0}'.format(f1_))
                print('Recall: {0}'.format(rec))
                print('Precision: {0}'.format(pre))
                acc = hf.batch_accuracy(batch_targets,tf.argmax(eval_class,axis=-1)).eval()
                print('RFI Class Accuracy: {0}'.format(acc))
            if i % 20 == 0:
                # Add metrics to summary and flush
                eval_writer.add_summary(s1,i)
                eval_writer.flush()
    elif mode == 'traineval':
        # Preferred mode, where training and evaluation happens concurrently, saving
        # summary statistics for both training and evaluation modes
        batch_init = np.copy(batch_size)
        lr = np.array([0.0003])
        train_writer = tf.summary.FileWriter('./'+model_name+'_train/',sess.graph)
        eval_writer = tf.summary.FileWriter('./'+model_name+'_eval_'+vdset+'/',sess.graph)
        for i in range(start_step, start_step+num_steps+1):
            batch_x_train, batch_targets_train = dset.next_train()
            feed_dict_train = {vis_input: batch_x_train, RFI_targets: batch_targets_train,
                                                  learn_rate: lr, mode_bn: True}
            _,loss_,strain,rec,pre,f1_train,ba = sess.run([train_fcn,loss,summary,recall,precision,f1,batch_accuracy],feed_dict=feed_dict_train)
            if i % 20 == 0:
                # Add training stats to summary and then roll into evaluation and do the same
                train_writer.add_summary(strain,i)
                train_writer.flush()

                batch_x_eval, batch_targets_eval = dset.next_eval()
                feed_dict_eval = {vis_input: batch_x_eval, RFI_targets: batch_targets_eval, mode_bn: True}
                eval_class,rec,pre,f1_eval,seval = sess.run([RFI_guess,recall,precision,f1,summary],feed_dict=feed_dict_eval)
                eval_writer.add_summary(seval,i)
                eval_writer.flush()
                
            if i % 100 == 0 or i == 1:
                # Output training and evaluation F1 scores to terminal
                print('Train F1 : %.9f' % f1_train)
                print('Eval F1 : %.9f' % f1_eval)
                print('Recall: {0}'.format(rec[0]))
                print('Precision: {0}'.format(pre[0]))
            if i % 5000 == 0 and i != 0:
                # Save model every 1000 steps
                print('Saving model...')
                save_path = saver.save(sess,'./'+model_name+'/model_%i.ckpt' % i)
                psize = np.random.choice([16,32])
                fold_factor = np.random.choice([8,16,32])                
                print('Using a fold factor of {0} and padding size of {1}'.format(fold_factor,psize))
                print('Changing input dimensions to {0} x {1}'.format(64+2*psize,2*psize + 1024/fold_factor))
                print('Subsampling time but padding back to 60 time ints.')
                dset.reload(fold_factor,psize,time_subsample=True)
            if i % 5000 == 0 and i != 0:
                # Optional increasing/decreasing batch size, preferred over decreasing learning rate
                # See https://arxiv.org/abs/1711.00489
                batch_init = int(batch_init*2)
                dset.change_batch_size(new_bs=batch_init)
                print('Decreasing batch size to {0}'.format(batch_init))
    else:
        # If no mode is specified it jumps into ensemble stats mode which is for understanding
        # how well the network performs on non-simulated data

        # Count how many trainable parameters we have
        total_parameters = 0
        for variable in tf.trainable_variables():
            shape = variable.get_shape()
            variable_parameters = 1
            for dim in shape:
                print(dim)
                variable_parameters *= dim.value
            print(variable_parameters)
            total_parameters += variable_parameters
        print('Number of trainable params: {0}'.format(total_parameters))
        from matplotlib import rc
        rc('text', usetex=True)
        plt.figure(num=None, figsize=(8, 6), dpi=300)
        time0 = time()
        ind = 0
        print('N=%i number of baselines time: ' % 1,time() - time0)
        # Cut off band edges, it's in factors of 64
        # Low: 1 and High: 1 is approx 13% of the band
        ci_1 = 1
        ci_2 = 1
        ct = 0
        ind = 0
        # Init all arrays for output stats
        fpr_arr = []
        tpr_arr = []
        mcc_arr = []
        npv_arr = []
        acc_arr = []
        f2_arr = []
        ident_flux = []
        missed_flux = []
        snrtpr_arr = []
        _FPR_ARR = []
        _TPR_ARR = []
        _MCC_ARR = []
        _F2_ARR = []
        best_thresh_arr = []
        ps = 96
        while ct != 18:
            data_, batch_x, batch_targets = dset.next_predict()
            pred_start = time()
            g = sess.run(RFI_guess, feed_dict={vis_input: batch_x, mode_bn: True})
            #print('Current Visibility: {0}'.format(ct))            
            pred_unfold = hf.unfoldl(tf.reshape(tf.argmax(g,axis=-1),[16,ps,ps]).eval(),padding=16)
            #pred_unfold = hf.unfoldl(tf.reshape(g[:,:,1],[16,68,68]).eval())
            pred_unfold_0 = hf.unfoldl(tf.reshape(g[:,:,0],[16,ps,ps]).eval(),padding=16)
            pred_unfold_1 = hf.unfoldl(tf.reshape(g[:,:,1],[16,ps,ps]).eval(),padding=16)
            pred_unfold_ = np.stack((pred_unfold_0,pred_unfold_1),axis=-1)
            pred_time = time() - pred_start
            target_unfold = hf.unfoldl(batch_targets.reshape(16,ps,ps),padding=16)
            if chtypes == 'AmpPhs':
                thresh = 0.329#0.62 #0.329 real #0.08 sim 
            else:
                thresh = 0.452#0.385 #0.385 real #0.126 sim
            y_true = target_unfold[:,64*ci_1:1024-64*ci_2].reshape(-1)
            y_pred = pred_unfold[:,64*ci_1:1024-64*ci_2].reshape(-1)
#            y_pred = hf.hard_thresh(pred_unfold[:,64*ci_1:1024-64*ci_2],thresh=thresh).reshape(-1)

            try:
                # Build confusion matrix
                confusion_pred = confusion_matrix(y_true,y_pred.astype(int))
                if np.shape(confusion_pred) == 1:
                    tn = confusion_pred[0][0]
                    fn = 1e-10
                    tp = 1e-10
                    fp = 1e-10
                else:
                    tn, fp, fn, tp = confusion_pred.ravel()
            except:
                ind+=1
                continue
            data_flux = np.abs(data_[:,64*ci_1:1024-64*ci_2])
            targets_ = target_unfold.reshape(-1,1024)[:,64*ci_1:1024-64*ci_2]
            #predicts_ = hf.unfoldl(tf.reshape(g[:,:,1],[16,68,68]).eval()).reshape(-1,1024)[:,64*ci_1:1024-64*ci_2]
            predicts_0 = hf.unfoldl(tf.reshape(g[:,:,0],[16,ps,ps]).eval(),padding=16).reshape(-1,1024)[:,64*ci_1:1024-64*ci_2]
            predicts_1 = hf.unfoldl(tf.reshape(g[:,:,1],[16,ps,ps]).eval(),padding=16).reshape(-1,1024)[:,64*ci_1:1024-64*ci_2]
            predicts_ = predicts_1-predicts_0#np.where(predicts_1>predicts_0,predicts_1,0.)
            tp_sum = 1.#np.sum(np.where(targets_+predicts_ == 2,data_flux,np.zeros_like(data_flux)))
            fn_sum = 1.#np.sum(np.where(targets_-predicts_ == 1,data_flux,np.zeros_like(data_flux)))
            tpr = tp/(1.*(tp+fn)) #recall
            fpr = tp/(1.*(tp+fp)) #precision/pos predictive value
            npv = tn/(1.*(tn+fn)) #neg predictive value
            mcc = hf.MCC(tp,tn,fp,fn)
            acc = (tp+tn)/(1.*(tp+tn+fp+fn))
            print('tp: {0} tn: {1} fp: {2} fn: {3}'.format(tp,tn,fp,fn))
            print('MCC: {}'.format(hf.MCC(tp,tn,fp,fn)))
            ident_flux.append(tp_sum)
            missed_flux.append(fn_sum)
            fpr_arr.append(fpr)
            tpr_arr.append(tpr)
            npv_arr.append(npv)
            mcc_arr.append(mcc)
            acc_arr.append(acc)
            f2_arr.append(5.*tpr*fpr/(4.*fpr + tpr))
            # Save individual visibility samples that have been predicted on
            if waterfall_sample:
                np.savez('Real_{0}_SamplePredict_{1}.npz'.format(chtypes,ct),data=data_,target=target_unfold,prediction=pred_unfold_,f2=5.*tpr*fpr/(4.*fpr + tpr),recall=tpr,precision=fpr)
#            np.savez('{0}_SamplePredict.npz'.format(chtypes),data=data_,target=target_unfold,prediction=hf.hard_thresh(pred_unfold,thresh=thresh),f2=5.*tpr*fpr/(4.*fpr + tpr),recall=tpr,precision=fpr)

            ind+=1
            ct+=1
            if stats:
                FPR_,TPR_,MCC_,F2_,best_thresh = hf.ROC_stats(targets_,predicts_.reshape(1,-1))
                _FPR_ARR.append(FPR_)
                _TPR_ARR.append(TPR_)
                _MCC_ARR.append(MCC_)
                _F2_ARR.append(F2_)
                best_thresh_arr.append(best_thresh)
                np.savez('ROC_curves_newSim_{0}.npz'.format(chtypes),TPR=_TPR_ARR,FPR=_FPR_ARR,MCC=_MCC_ARR,F2=_F2_ARR,best_thresh=np.nanmedian(best_thresh_arr))

        print('Accuracy: {0}'.format(np.nanmean(acc_arr)))
        print('Precision: {0}'.format(np.nanmean(fpr_arr)))
        print('Recall: {0}'.format(np.nanmean(tpr_arr)))
        print('F2: {0}'.format(np.nanmean(f2_arr)))
        
        if ROC:
            #        try:
            f = h5py.File('KernelSize_TPFPrates_AllData'+vdset+'.h5','a')
            try:
                mname = f.create_group(model_name+vdset)
                mname.create_dataset('FPR',data=fpr_arr)
                mname.create_dataset('TPR',data=tpr_arr)
                mname.create_dataset('NPV', data=npv_arr)
                mname.create_dataset('MCC', data=mcc_arr)
                mname.create_dataset('ACC', data=acc_arr)
                mname.create_dataset('Identified Flux',data=ident_flux)
                mname.create_dataset('Missed Flux',data=missed_flux)
                f.close()
            except:
                print('Data group already exists.')
                mname = f.require_group(model_name+vdset)
                mname['FPR'][...] = fpr_arr
                mname['TPR'][...] = tpr_arr
                mname['NPV'][...] = npv_arr
                mname['MCC'][...] = mcc_arr
                mname['ACC'][...] = acc_arr
                mname['Identified Flux'][...] = ident_flux
                mname['Missed Flux'][...] = missed_flux
                f.close()
        print('Prediction pipeline time per waterfall visibility: {0}'.format(pred_time+dset_load_time))
