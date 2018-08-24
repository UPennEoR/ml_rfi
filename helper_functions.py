import numpy as np
import tensorflow as tf
import h5py
from time import time
import os
import random
import pylab as plt

np.random.seed(555)

def transpose(X):
    """
    Transpose for use in the map functions.
    """
    return X.T

def normalize(X):
    """
    Normalization for the log amplitude required in the folding process.
    """
    sh = np.shape(X)
    LOGabsX = np.nan_to_num(np.log10(np.abs(X+(1e-8)*np.random.rand(sh[0],sh[1])))).real
    return (LOGabsX-np.nanmean(LOGabsX))/np.nanstd(np.abs(LOGabsX))

def normphs(X):
    """                                                                                                                                                                            
    Normalization for the log amplitude required in the folding process.                                                                                                            
    """
    sh = np.shape(X)
    phsX = np.angle(X)
    phsX_ = np.mod(phsX,np.pi)
    return (phsX_ - np.nanmean(phsX_))#/np.nanstd(np.abs(phsX_))
    
def noisy_relu(x):
    try:
        mean,var = tf.nn.moments(x, axes=[1,2,3])
    except:
        mean,var = tf.nn.moments(x, axes=[1,2])
    return tf.nn.relu(x+tf.random_normal(tf.shape(x),stddev=tf.multiply(var,var)))

def leaky_relu(x,alpha=.8):
    return tf.nn.relu(x) - alpha * tf.nn.relu(-x)

def foldl(data,ch_fold=16,padding=2):
    """
    Folding function for carving up a waterfall visibility flags for prediction in the FCN.
    """
    sh = np.shape(data)
    _data = data.T.reshape(ch_fold,sh[1]/ch_fold,-1)[2:14]
    _DATA = np.array(map(transpose,_data))
    _DATApad = np.array(map(pad,_DATA))
    return _DATApad

def pad(data,padding=2):
    #for 68 pad with 2
    #    100        18
    sh = np.shape(data)
    t_pad = (sh[1] - sh[0])/2
    data_pad = np.pad(data,pad_width=((t_pad+padding,t_pad+padding),(padding,padding)),mode='constant',constant_values=0)
#    psh = np.shape(data_pad)
#    data_pad = np.where(data_pad == 0.,np.max(data_pad)*np.random.rand(psh[0],psh[1]),data_pad)
    return data_pad

def unpad(data,diff=4,padding=2):
    sh = np.shape(data)
    t_unpad = sh[0]
    # time axis isnt unpadding correctly
    return data[padding/2+diff/2:,padding:][:-padding/2-diff/2,:-padding][padding/2:,:][:-padding/2,:]
                      
def fold(data,ch_fold=16,padding=2):
    """
    Folding function for carving waterfall visibilities with additional normalized log 
    and phase channels.
    """
    sh = np.shape(data)
    _data = data.T.reshape(ch_fold,sh[1]/ch_fold,-1)[2:14]
    _DATA = np.array(map(transpose,_data))
    _DATApad = np.array(map(pad,_DATA))
    DATA = np.stack((np.array(map(normalize,_DATApad)),np.mod(np.array(map(normphs,_DATApad)),np.pi),np.mod(np.array(map(normphs,_DATApad)),np.pi)),axis=-1)
    return DATA

def unfoldl(data_fold,nchans=1024,ch_fold=16,padding=2):
    """
    Unfolding function for recombining the carved label (flag) frequency windows back into a complete 
    waterfall visibility.
    """
    data_unpad = np.array(map(unpad,data_fold))
    ch_fold,ntimes,dfreqs = np.shape(data_unpad)
    data_ = np.array(map(transpose,data_unpad))
    _data = data_.reshape(ch_fold*dfreqs,ntimes).T
    return _data

def stacked_layer(input_layer,num_filter_layers,kt,kf,activation,stride,pool,bnorm=True,name='None',dropout=False,mode=True):
    """
    Creates a 3x stacked layer of convolutional layers. Each layer uses the same kernel size.
    Batch normalized output is default and recommended for faster convergence, although
    not every may require it (???).
    """
    conva = tf.layers.conv2d(inputs=input_layer,
                             filters=num_filter_layers,
                             kernel_size=[kt,kf],
                             padding="same",
                             activation=activation)
                            
    if not dropout:
        convb = tf.layers.conv2d(inputs=conva,
                             filters=num_filter_layers,
                             kernel_size=[kt,kf],
                             padding="same",
                             activation=activation)
                            
    else:
        convb = tf.layers.dropout(tf.layers.conv2d(inputs=conva,
                             filters=num_filter_layers,
                             kernel_size=[kt,kf],
                             padding="same",
                                                   activation=activation),rate=.5)
                             
        
    convc = tf.layers.conv2d(inputs=convb,
                             filters=num_filter_layers,
                             kernel_size=[1,1],
                             padding="same",
                             activation=activation)
                             
    if bnorm:
    	#bnorm_conv = tf.contrib.layers.batch_norm(convc,scale=True)
        bnorm_conv = tf.layers.batch_normalization(convc,scale=True,center=True,training=mode,fused=True)
    else:
    	bnorm_conv = convc
    pool = tf.layers.max_pooling2d(inputs=bnorm_conv,
                                    pool_size=pool,
                                    strides=stride)
    return pool

def batch_accuracy(labels,predictions):
    labels = tf.cast(labels,dtype=tf.int64)
    predictions = tf.cast(predictions,dtype=tf.int64)
    correct = tf.reduce_sum(tf.cast(tf.equal(tf.add(labels,predictions),2),dtype=tf.int64))
    total = tf.reduce_sum(labels)
    return tf.divide(correct,total)

def accuracy(labels,predictions):
    correct = 1.*np.sum((labels+predictions)==2)
    total = 1.*np.sum(labels==1)
    print('correct',correct)
    print('total',total)
    try:
        return correct/total
    except:
        return 0.
    
def delay_transform(data,flags):
    sh = data.get_shape().as_list()
    data = tf.reshape(data,[-1,sh[1],sh[1],1])
    flags = tf.cast(tf.reshape(flags,[-1,sh[1],sh[1],1]),dtype=tf.complex64)
    flags_comp = tf.add(flags,1j*flags)
    #flags = tf.cast(flags_comp,dtype=tf.complex64)
    #flags_ = tf.cast(tf.logical_not(tf.cast(flags,dtype=tf.bool)),dtype=tf.complex64)
    data_noflags = tf.transpose(data,perm=[2,1,0,3])
    DATA_noflags_ = tf.abs(tf.fft(data_noflags))
    DATA_noflags = tf.transpose(DATA_noflags_,perm=[2,1,0,3])
    data_ = tf.transpose(tf.multiply(data,flags_comp),perm=[2,1,0,3])
    DATA_ = tf.abs(tf.fft(data_))
    DATA = tf.transpose(DATA_,perm=[2,1,0,3])
    return tf.reduce_mean(tf.divide(DATA,DATA_noflags)[:,:,:,:])


def hard_thresh(layer,thresh=1e-5):
    try:
        layer = layer[:,:,1]
    except:
        layer = layer[:,1]
    return tf.where(layer > thresh,tf.ones_like(layer),tf.zeros_like(layer))

def ROC_stats(ground_truth,softmax_logits):
    ground_truth = tf.cast(ground_truth,dtype=tf.bool)
    softmax_logits = tf.cast(softmax_logits,dtype=tf.float32)
    #for i in linspace(.0,1.,100):
    thresholds = np.linspace(.0,1.,100,dtype=np.float32)
    FPR = tf.metrics.false_positives_at_thresholds(ground_truth,softmax_logits,thresholds)[0]
    TPR = tf.metrics.true_positives_at_thresholds(ground_truth,softmax_logits,thresholds)[0]
    return FPR,TPR

def plot_ROC(FPR,TPR,fname):
    fig = plt.figure()
    plt.plot(FPR,TPR)
    plt.plot(np.linspace(0.,1.,100),np.linspace(0.,1.,100),label='Random Choice')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.savefig(fname+'.pdf')
    
#def load_data():

class RFIDataset():
    def __init__(self):
        print('Welcome to the HERA RFI training and evaluation dataset.')

    def load(self,batch_size,psize,hybrid=False,chtypes='AmpPhs',fold_factor=16):
        # load data
        self.chtypes = chtypes
        self.batch_size = batch_size
        print('A batch size of %i has been set.' % self.batch_size)
        f1 = h5py.File('RealVisRFI_v3.h5') # 'RealVisRFI_v3.h5','r') # Load in a real dataset 
        f2 = h5py.File('SimVis_v6_10000.h5','r') # Load in simulated data
        #    f2 = h5py.File('RealVisRFI_v3.h5','r')

        self.psize = psize # Pixel pad size for individual carved bands
        #fold_factor = 16 # Carving narrowband factor

        # We want to augment our training dataset with the entirety of the simulated data
        # but with only half of the real data. The remaining real data half will become
        # the evaluation dataset
    
        f1_r = 900#np.shape(f1['data'])[0]
        f2_s = np.shape(f2['data'])[0]

        print 'Size of real dataset: ',f1_r
        print ''
        # Cut up real dataset and labels
        f1_r = 900#np.shape(f1['data'])[0] # Everything after 900 doesn't look good
        samples = range(f1_r)
        rnd_ind = np.random.randint(0,f1_r)
        #else:
        #    rnd_ind = 100#np.random.randint(0,f1_r)
        #    samples = [rnd_ind]
        time0 = time()

        if chtypes ==  'AmpPhs':
            f_real = (np.array(map(fold,f1['data'][:f1_r,:,:]))[:,:,:,:,:2]).reshape(-1,self.psize,self.psize,2)
            f_real_labels = np.array(map(foldl,f1['flag'][:f1_r,:,:])).reshape(-1,self.psize,self.psize)
            # Cut up sim dataset and labels
            f_sim = (np.array(map(fold,f2['data'][:f2_s,:,:]))[:,:,:,:,:2]).reshape(-1,self.psize,self.psize,2)
            f_sim_labels = np.array(map(foldl,f2['flag'][:f2_s,:,:])).reshape(-1,self.psize,self.psize)
        elif chtypes == 'AmpPhs2' :
            f_real = np.array(map(fold,f1['data'][:f1_r,:,:])).reshape(-1,self.psize,self.psize,3)
            f_real_labels = np.array(map(foldl,f1['flag'][:f1_r,:,:])).reshape(-1,self.psize,self.psize)
            # Cut up sim dataset and labels
            f_sim = np.array(map(fold,f2['data'][:f2_s,:,:])).reshape(-1,self.psize,self.psize,3)
            f_sim_labels = np.array(map(foldl,f2['flag'][:f2_s,:,:])).reshape(-1,self.psize,self.psize)
        elif chtypes == 'Amp':
            f_real = (np.array(map(fold,f1['data'][:f1_r,:,:]))[:,:,:,:,0]).reshape(-1,self.psize,self.psize,1)
            print('f_real: ',np.shape(f_real))
            f_real_labels = np.array(map(foldl,f1['flag'][:f1_r,:,:])).reshape(-1,self.psize,self.psize)
            f_sim = (np.array(map(fold,f2['data'][:f2_s,:,:]))[:,:,:,:,0]).reshape(-1,self.psize,self.psize,1)
            f_sim_labels = np.array(map(foldl,f2['flag'][:f2_s,:,:])).reshape(-1,self.psize,self.psize)
        elif chtypes == 'Phs':
            f_real = (np.array(map(fold,f1['data'][:f1_r,:,:])).reshape(-1,self.psize,self.psize,1))
            f_real_labels = np.array(map(foldl,f1['flag'][:f1_r,:,:])).reshape(-1,self.psize,self.psize)
            f_sim = (np.array(map(fold,f2['data'][:f2_s,:,:])).reshape(-1,self.psize,self.psize,1))
            f_sim_labels = np.array(map(foldl,f2['flag'][:f2_s,:,:])).reshape(-1,self.psize,self.psize)
            
        print 'Training dataset loaded.'
        print 'Training dataset size: ',np.shape(f_real)

        print 'Simulated training dataset loaded.'
        print 'Training dataset size: ',np.shape(f_sim)
        
        real_sh = np.shape(f_real)

        if chtypes == 'AmpPhsCmp':
            d_type = np.complex64
        else:
            d_type = np.float64
        real_len = np.shape(f_real)[0]        
        if hybrid:
            print('Hybrid training dataset selected.')
            # We want to mix the real and simulated datasets
            # and then keep some real datasets for evaluation
            real_len = np.shape(f_real)[0]
            self.eval_data = np.asarray(f_real[:int(real_len/2),:,:,:],dtype=d_type)
            self.eval_labels = np.asarray(f_real_labels[:int(real_len/2),:,:],dtype=np.int32).reshape(-1,real_sh[1]*real_sh[2])
            
            train_data = np.vstack((f_real[int(real_len/2):,:,:,:],f_sim))
            train_labels = np.vstack((f_real_labels[int(real_len/2):,:,:],f_sim_labels))
            hybrid_len = np.shape(train_data)[0]
            mix_ind = np.random.permutation(hybrid_len)

            self.train_data = train_data[mix_ind,:,:,:]
            self.train_labels = train_labels[mix_ind,:,:].reshape(-1,real_sh[1]*real_sh[2])
            self.eval_len = np.shape(self.eval_data)[0]
            self.train_len = np.shape(self.train_data)[0]
        else:
            # Format evaluation dataset
            sim_len = np.shape(f_sim)[0]
            self.eval_data = np.asarray(f_sim[int(sim_len/3):,:,:,:],dtype=d_type)
            self.eval_labels = np.asarray(f_sim_labels[int(sim_len/3):,:,:],dtype=np.int32).reshape(-1,real_sh[1]*real_sh[2])
            eval1 = np.shape(self.eval_data)[0]

            # Format training dataset
            #self.train_data = np.asarray(f_real[:700,:,:,:],dtype=np.float64)
            #self.train_labels = np.asarray(f_real_labels[:700,:,:],dtype=np.int32).reshape(-1,real_sh[1]*real_sh[2])
            self.train_data = np.asarray(f_sim[:int(1-sim_len/3),:,:,:],dtype=d_type)#np.asarray(np.vstack((f_sim,f_real[:real_sh[0]/2,:,:,:])),dtype=np.float32)
            self.train_labels = np.asarray(f_sim_labels[:(1-sim_len/3),:,:],dtype=np.int32).reshape(-1,real_sh[1]*real_sh[2])#np.asarray(np.vstack((f_sim_labels,f_real_labels[:real_sh[0]/2,:,:])),dtype=np.int32).reshape(-1,real_sh[1]*real_sh[2])

            train0 = np.shape(self.train_data)[0]
            self.test_data = np.asarray(fold(f1['data'][rnd_ind,:,:]), dtype=d_type) # Random real visibility for testing
            self.test_labels = np.asarray(foldl(f1['flag'][rnd_ind,:,:]), dtype=np.int32).reshape(-1,real_sh[1]*real_sh[2])
            self.eval_len = np.shape(self.eval_data)[0]
            self.train_len = np.shape(self.train_data)[0]

    def next_train(self):
#        self.train_data = np.roll(self.train_data,self.batch_size,axis=0)
#        self.train_labels = np.roll(self.train_labels,self.batch_size,axis=0)
        rand_batch = random.sample(range(self.train_len),self.batch_size)#np.random.randint(0,self.train_len,size=self.batch_size)
        return self.train_data[rand_batch,:,:,:],self.train_labels[rand_batch,:]

    def next_eval(self):
#        self.eval_data = np.roll(self.eval_data,self.batch_size,axis=0)
#        self.eval_labels = np.roll(self.eval_labels,self.batch_size,axis=0)
        rand_batch = random.sample(range(self.eval_len),self.batch_size)#np.random.randint(0,self.eval_len,size=self.batch_size)
        return self.eval_data[rand_batch,:,:,:],self.eval_labels[rand_batch,:]

    def random_test(self,samples):
        ind = random.sample(range(np.shape(self.eval_data)[0]),samples)#np.random.randint(0,np.shape(self.eval_data)[0],size=samples)
        if self.chtypes == 'Amp':
            ch = 1
        elif self.chtypes == 'AmpPhs':
            ch = 2
        return self.eval_data[ind,:,:,:].reshape(samples,self.psize,self.psize,ch),self.eval_labels[ind,:].reshape(samples,self.psize*self.psize)

    
