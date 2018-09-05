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
    phsX_ = np.sin(np.angle(X))#np.random.randn(*sh)
    #phsX_ -= np.mean(phsX_)
    return phsX_#/np.nanmax(np.abs(phsX_))

def tfnormalize(X):
    sh = np.shape(X)
    mu,var = tf.nn.moments(X,axes=[1,2])
    mu = tf.reshape(mu,[-1,1,1,sh[3]])
#    var = tf.reshape(var,[-1,1,1,sh[3]])
#    var = tf.where(tf.is_nan(var),tf.ones_like(var),var)
#    var = tf.where(tf.is_inf(var),tf.ones_like(var),var)
#    var = tf.where(var == 0.,tf.ones_like(var),var)
    mu = tf.where(mu == np.nan,tf.zeros_like(mu),mu)
    mu = tf.where(mu == np.inf,tf.zeros_like(mu),mu)
    X_norm = X-mu
#    X_norm /= tf.sqrt(tf.reduce_sum(X_norm**2))
    X_norm = tf.where(tf.is_nan(X_norm),tf.zeros_like(X_norm),X_norm)
    return X_norm

def noisy_relu(x):
    try:
        mean,var = tf.nn.moments(x, axes=[1,2,3])
    except:
        mean,var = tf.nn.moments(x, axes=[1,2])
    return tf.nn.relu(x+tf.random_normal(tf.shape(x),stddev=tf.multiply(var,var)))

def leaky_relu(x,alpha=.8):
    return tf.nn.relu(x) - alpha * tf.nn.relu(-x)

def foldl(data,ch_fold,padding):
    """
    Folding function for carving up a waterfall visibility flags for prediction in the FCN.
    """
    sh = np.shape(data)
    _data = data.T.reshape(ch_fold,sh[1]/ch_fold,-1)#[2:14]
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
                      
def fold(data,ch_fold,padding):
    """
    Folding function for carving waterfall visibilities with additional normalized log 
    and phase channels.
    """
    sh = np.shape(data)
    _data = data.T.reshape(ch_fold,sh[1]/ch_fold,-1)
    _DATA = np.array(map(transpose,_data))
    _DATApad = np.array(map(pad,_DATA))
    DATA = np.stack((np.array(map(normalize,_DATApad)),np.array(map(normphs,_DATApad)),np.mod(np.array(map(normphs,_DATApad)),np.pi)),axis=-1)
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

def stacked_layer(input_layer,num_filter_layers,kt,kf,activation,stride,pool,bnorm=True,name='None',dropout=None,maxpool=True,mode=True):
    """
    Creates a 3x stacked layer of convolutional layers. Each layer uses the same kernel size.
    Batch normalized output is default and recommended for faster convergence, although
    not every may require it (???).
    """
#    try:
#        dropout = dropout.eval()
#        print('Dropout is set to '+str(dropout))
#    except:
#        dropout = 0.0
    conva = tf.layers.conv2d(inputs=input_layer,
                             filters=num_filter_layers,
                             kernel_size=[kt,kf],
                             padding="same",
                             activation=activation)
                            
    if dropout is not None:
        convb = tf.layers.dropout(tf.layers.conv2d(inputs=conva,
                             filters=num_filter_layers,
                             kernel_size=[kt,kf],
                             padding="same",
                                                   activation=activation), rate=dropout)                         
    else:
        convb = tf.layers.conv2d(inputs=conva,
                             filters=num_filter_layers,
                             kernel_size=[kt,kf],
                             padding="same",
                                                   activation=activation)
    shb = convb.get_shape().as_list()

    convc = tf.layers.conv2d(inputs=convb,
                             filters=num_filter_layers,
                             kernel_size=[1,1],
                             padding="same",
                             activation=activation)
#    convc = tf.layers.dense(convc,units=num_filter_layers,activation=activation)

    if bnorm:
    	#bnorm_conv = tf.contrib.layers.batch_norm(convc,scale=True)
        bnorm_conv = tf.layers.batch_normalization(convc,scale=True,center=True,training=mode,fused=True)
    else:
    	bnorm_conv = convc
    if maxpool:
        pool = tf.layers.max_pooling2d(inputs=bnorm_conv,
                                    pool_size=pool,
                                       strides=stride)
    else:
        pool = tf.layers.average_pooling2d(inputs=bnorm_conv,
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
        return 1.
    
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

    def load(self,tdset,batch_size,psize,hybrid=False,chtypes='AmpPhs',fold_factor=16,cut=False):
        # load data
        self.chtypes = chtypes
        self.batch_size = batch_size
        print('A batch size of %i has been set.' % self.batch_size)
        f1 = h5py.File('RealVisRFI_v3.h5') # 'RealVisRFI_v3.h5','r') # Load in a real dataset
        #f1 = h5py.File('IDR21TrainingData.h5','r')
        if tdset == 'v5':
            f2 = h5py.File('SimVis_v5.h5','r') # Load in simulated data
        elif tdset == 'v6':
            f2 = h5py.File('SimVis_v6_10000.h5','r')
        #    f2 = h5py.File('RealVisRFI_v3.h5','r')
        elif tdset == 'v4':
            f2 = h5py.File('SimVisRFI_15_120_v4.h5','r')
        
        self.psize = psize # Pixel pad size for individual carved bands
        #fold_factor = 16 # Carving narrowband factor

        # We want to augment our training dataset with the entirety of the simulated data
        # but with only half of the real data. The remaining real data half will become
        # the evaluation dataset
    
        f1_r = 900#np.shape(f1['data'])[0]
        f2_s = np.shape(f2['data'])[0]

        f_factor_r = f1_r*[fold_factor]
        pad_r = f1_r*[2]
        f_factor_s = f2_s*[fold_factor]
        pad_s = f2_s*[fold_factor]
        
        print 'Size of real dataset: ',f1_r
        print ''
        # Cut up real dataset and labels
        #f1_r = np.shape(f1['data'])[0] # Everything after 900 doesn't look good
        samples = range(f1_r)
        rnd_ind = np.random.randint(0,f1_r)
        if cut:
            data_real = f1['data'][:f1_r,:,2*64:14*64]
            labels_real = f1['flag'][:f1_r,:,2*64:14*64]
            data_sim = f2['data'][:f2_s,:,2*64:14*64]
            labels_sim = f2['flag'][:f2_s,:,2*64:14*64]            
        else:
            data_real = f1['data'][:f1_r,:,:]
            labels_real = f1['flag'][:f1_r,:,:]
            data_sim = f2['data'][:f2_s,:,:]
            labels_sim = f2['flag'][:f2_s,:,:]
        #else:
        #    rnd_ind = 100#np.random.randint(0,f1_r)
        #    samples = [rnd_ind]
        time0 = time()

        if chtypes ==  'AmpPhs':
            f_real = (np.array(map(fold,data_real,f_factor_r,pad_r))[:,:,:,:,:2]).reshape(-1,self.psize,self.psize,2)
            f_real_labels = np.array(map(foldl,labels_real,f_factor_r,pad_r)).reshape(-1,self.psize,self.psize)
            # Cut up sim dataset and labels
            f_sim = (np.array(map(fold,data_sim,f_factor_s,pad_s))[:,:,:,:,:2]).reshape(-1,self.psize,self.psize,2)
            f_sim_labels = np.array(map(foldl,labels_sim,f_factor_s,pad_s)).reshape(-1,self.psize,self.psize)
        elif chtypes == 'AmpPhs2' :
            f_real = np.array(map(fold,data_real,f_factor_r,pad_r)).reshape(-1,self.psize,self.psize,3)
            f_real_labels = np.array(map(foldl,labels_real,f_factor_r,pad_r)).reshape(-1,self.psize,self.psize)
            # Cut up sim dataset and labels
            f_sim = np.array(map(fold,data_sim,f_factor_s,pad_s)).reshape(-1,self.psize,self.psize,3)
            f_sim_labels = np.array(map(foldl,labels_sim,f_factor_s,pad_s)).reshape(-1,self.psize,self.psize)
        elif chtypes == 'Amp':
            f_real = (np.array(map(fold,data_real,f_factor_r,pad_r))[:,:,:,:,0]).reshape(-1,self.psize,self.psize,1)
            print('f_real: ',np.shape(f_real))
            f_real_labels = np.array(map(foldl,labels_real,f_factor_r,pad_r)).reshape(-1,self.psize,self.psize)
            f_sim = (np.array(map(fold,data_sim,f_factor_s,pad_s))[:,:,:,:,0]).reshape(-1,self.psize,self.psize,1)
            f_sim_labels = np.array(map(foldl,labels_sim,f_factor_s,pad_s)).reshape(-1,self.psize,self.psize)
        elif chtypes == 'Phs':
            f_real = (np.array(map(fold,data_real,f_factor_r,pad_r)).reshape(-1,self.psize,self.psize,1))
            f_real_labels = np.array(map(foldl,labels_real,f_factor_r,pad_r)).reshape(-1,self.psize,self.psize)
            f_sim = (np.array(map(fold,data_sim,f_factor_s,pad_s)).reshape(-1,self.psize,self.psize,1))
            f_sim_labels = np.array(map(foldl,labels_sim,f_factor_s,pad_s)).reshape(-1,self.psize,self.psize)
            
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
            self.test_data = self.eval_data[rnd_ind,:,:,:].reshape(1,real_sh[1],real_sh[2],real_sh[3])#np.asarray(fold(data_real[rnd_ind,:,:],12,2), dtype=d_type) # Random real visibility for testing
            self.test_labels = self.eval_labels[rnd_ind,:].reshape(1,real_sh[1]*real_sh[2]) #np.asarray(foldl(labels_real[rnd_ind,:,:],12,2), dtype=np.int32).reshape(-1,real_sh[1]*real_sh[2])
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
        rand_batch = random.sample(range(self.eval_len),self.batch_size) #np.random.randint(0,self.eval_len,size=self.batch_size)
        return self.eval_data[rand_batch,:,:,:],self.eval_labels[rand_batch,:]

    def random_test(self,samples):
        ind = random.sample(range(np.shape(self.eval_data)[0]),samples)#np.random.randint(0,np.shape(self.eval_data)[0],size=samples)
        if self.chtypes == 'Amp':
            ch = 1
        elif self.chtypes == 'AmpPhs':
            ch = 2
        elif self.chtypes == 'AmpPhs2':
            ch = 3
        return self.eval_data[ind,:,:,:].reshape(samples,self.psize,self.psize,ch),self.eval_labels[ind,:].reshape(samples,self.psize*self.psize)

    
