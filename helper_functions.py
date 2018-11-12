import numpy as np
import tensorflow as tf
import h5py
from time import time
import os
import random
import pylab as plt
from sklearn.metrics import confusion_matrix
from scipy import ndimage
import aipy
import noise
#np.random.seed()

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
    absX = np.abs(X)
    absX = np.where(absX <= 0., (1e-8)*np.random.randn(sh[0],sh[1]),absX)
    LOGabsX = np.nan_to_num(np.log10(absX))
    return np.nan_to_num((LOGabsX-np.nanmean(LOGabsX))/np.nanstd(np.abs(LOGabsX)))

def normphs(X):
    """                                                                                                                                           
    Normalization for the log amplitude required in the folding process.                                                                          
    """
    sh = np.shape(X)
#    phsX_ = np.angle(X)
    #phsX_ = np.sin(np.angle(X))
    ### Try a differencing
    diff = [np.sin(np.angle(X[:,i+1])) - np.sin(np.angle(X[:,i])) for i in range(sh[1]-1)]
    diff.append(np.sin(np.angle(X[:,-1])) - np.sin(np.angle(X[:,-2])))
    return np.array(diff).T
#    phsX_ -= np.nanmean(phsX_)
#    return phsX_/np.pi#np.ones_like(X).real

def tfnormalize(X):
    sh = np.shape(X)
    X_norm = tf.contrib.layers.layer_norm(X,trainable=False)
    return X

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
    _data = data.T.reshape(ch_fold,sh[1]/ch_fold,-1)#[2:14]
    _DATA = np.array(map(transpose,_data))
    _DATApad = np.array(map(pad,_DATA))
    return _DATApad

def pad(data,padding=2):
    #for 68 pad with 2
    #    100        18
    sh = np.shape(data)
    t_pad = (sh[1] - sh[0])/2
    data_pad = np.pad(data,pad_width=((t_pad+padding,t_pad+padding),(padding,padding)),mode='reflect')
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
                             kernel_size=[kt,kt],
                             padding="same",
                             activation=activation)
                            
    if dropout is not None:
        convb = tf.layers.dropout(tf.layers.conv2d(inputs=conva,
                             filters=num_filter_layers,
                             kernel_size=[kt,kt],
                             padding="same",
                                                   activation=activation), rate=dropout)                         
    else:
        convb = tf.layers.conv2d(inputs=conva,
                             filters=num_filter_layers,
                             kernel_size=[kt,kt],
                             padding="same",
                                                   activation=activation)
    shb = convb.get_shape().as_list()

    convc = tf.layers.conv2d(inputs=convb,
                             filters=num_filter_layers,
                             kernel_size=(1,1),
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

def MCC(tp,tn,fp,fn):
    if tp==0 and fn ==0:
        return (tp*tn - fp*fn)
    else:
        return (tp*tn - fp*fn)/np.sqrt((1.*(tp+fp)*(tp+fn)*(tn+fp)*(tn+fn)))

def f1(tp,tn,fp,fn):
    precision = tp/(1.*(tp+fp))
    recall = tp/(1.*(tp+fn))
    return 2.*precision*recall/(precision+recall)


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

def SNRvsTPR(data,true_flags,flags):
    SNR = np.linspace(0.,4.,30)
    snr_tprs = []
    data_ = np.copy(data)
    flags_ = np.copy(flags)
    true_flags_ = np.copy(true_flags)
    for snr_ in SNR:
       # print(snr_)
        snr_map = np.log10(data_*flags_/np.std(data_*np.logical_not(true_flags)))
        snr_inds = snr_map < snr_
        #print('snr sum',np.sum(snr_inds))
        confuse_mat = confusion_matrix(true_flags_[snr_inds].astype(int).reshape(-1),flags_[snr_inds].astype(int).reshape(-1))
        if np.size(confuse_mat) == 1:
            tp = 1.
            tn = confuse_mat[0][0]
            fp = 0.
            fn = 0.
        else:
            try:
                tn, fp, fn, tp = confuse_mat.ravel()
            except:
                tp = np.nan
                fn = np.nan
        snr_tprs.append(MCC(tp,tn,fp,fn))
        data_[snr_inds] = 0.
    return snr_tprs

def hard_thresh(layer,thresh=0.5):
#    try:
#        layer = layer[:,:,1]
#    except:
#        layer = layer[:,1]
    layer_sigmoid = tf.nn.sigmoid(layer)
    return tf.where(layer_sigmoid > thresh,tf.ones_like(layer),tf.zeros_like(layer))

def hard_thresh2(layer,thresh=0.5):
    layer_sigmoid = 1./(1. + np.exp(-layer))
    return np.where(layer_sigmoid > thresh, np.ones_like(layer),np.zeros_like(layer))

def softmax(X):
    return np.exp(X)/np.sum(np.exp(X),axis=-1)

def ROC_stats2(ground_truth,logits):
    ground_truth = np.reshape(ground_truth,[-1])
    thresholds = np.logspace(-2,2,30)
    FPR = []
    TPR = []
    MCC_arr = []
    F2 = []
    for thresh in thresholds:
        pred_ = hard_thresh2(logits,thresh=thresh).reshape(-1)
        tn, fp, fn, tp = confusion_matrix(ground_truth,pred_).ravel()
        recall = tp/(1.*(tp+fn))
        precision = tp/(1.*(tp+fp))
        TPR.append(tp/(1.*(tp+fn)))
        FPR.append(fp/(1.*(fp+tn)))
        MCC_arr.append(MCC(tp,tn,fp,fn))
        F2.append(5.*recall*precision/(4.*precision + recall))
    #print(MCC_arr)
    best_thresh = thresholds[np.nanargmax(F2)]
    return FPR,TPR,MCC_arr,F2,best_thresh
        
def ROC_stats(ground_truth,softmax_logits):
    ground_truth = tf.reshape(tf.cast(ground_truth,dtype=tf.float32),[-1])
    softmax_logits = tf.reshape(tf.cast(tf.nn.softmax(softmax_logits),dtype=tf.float32),[-1])
    thresholds = np.logspace(-20,1,100,dtype=np.float32)#np.linspace(.0,1.,100,dtype=np.float32)
    FPR = tf.metrics.false_positives_at_thresholds(ground_truth,softmax_logits,thresholds)[0]
    TPR = tf.metrics.true_positives_at_thresholds(ground_truth,softmax_logits,thresholds)[0]
    return FPR,TPR,thresholds

def plot_ROC(FPR,TPR,fname):
    fig = plt.figure()
    plt.plot(np.asarray(FPR).reshape(-1),np.asarray(TPR).reshape(-1))
    plt.plot(np.linspace(0.,1.,1000),np.linspace(0.,1.,1000),label='Random Choice')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.savefig(fname+'.pdf')

def load_pipeline_dset(stage_type):
    f = h5py.File('IDR21TrainingData_Raw_vX.h5','r')
    #f = h5py.File('IDR21InitialFlags_v2.h5','r') 
    #f = h5py.File('IDR21TrainingData.h5','r')
    #f = h5py.File('RealVisRFI_v5.h5','r')
    #f = h5py.File('RawRealVis_v1.h5','r')
    #f = h5py.File('SimVis_Blips_100.h5','r')
    #f = h5py.File('SimVis_1000_v9.h5','r')
    #uvO', u'uvOC', u'uvOCRS', u'uvOCRSD'
    try:
        if stage_type == 'uv':
            return f['uv']
        elif stage_type == 'uvO':
            return f['uvO']
        elif stage_type == 'uvOC':
            return f['uvOC']
        elif stage_type == 'uvOCRS':
            return f['uvOCRS']
        elif stage_type == 'uvOCRSD':
            return f['uvOCRSD']
    except:
        return f

def stride(input_data,input_labels):
    """
    Takes an input waterfall visibility with labels and strides across frequency,
    producing (Nchan - 64)/S new waterfalls to be folded.
    """
    spw_hw = 32 #spectral window half width
    nchans = 1024
    fold = nchans/(2*spw_hw)
    sample_spws = random.sample(range(0,60),fold)
    
    x = np.array([input_data[:,i-spw_hw:i+spw_hw] for i in range(spw_hw,1024-spw_hw,(nchans-2*spw_hw)/60)])
    x_labels = np.array([input_labels[:,i-spw_hw:i+spw_hw] for i in range(spw_hw,1024-spw_hw,(nchans-2*spw_hw)/60)])
    X = np.array([x[i].T for i in sample_spws])
    X_labels = np.array([x_labels[i].T for i in sample_spws]) 
    X_ = X.reshape(-1,60).T
    X_labels = X_labels.reshape(-1,60).T
    return X_,X_labels
    
def patchwise(data,labels):
    strided_dp = np.array(map(stride,data,labels))
    data_strided = np.copy(strided_dp[:,0,:,:])
    labels_strided = np.copy(strided_dp[:,1,:,:].astype(int))
    return data_strided,labels_strided

def expand_dataset(data,labels):
    bloat = 2
    sh = np.shape(data)
    out_data = []
    out_labels = []
    for i in range(bloat*sh[0]):
        rnd_num = np.random.rand()
        rnd_data_ind = np.random.randint(0,sh[0])
        order_ind = np.random.randint(0,10)
        order = np.logspace(-4,-1,10)[order_ind]
        noise = np.random.randn(sh[1],sh[2])+1j*np.random.randn(sh[1],sh[2])
        noise_data = np.copy(data[rnd_data_ind])
        noise_labels = np.copy(labels[rnd_data_ind])
        noise_data[:,:,0] += order*np.abs(noise)
        if sh[3] > 1:
            noise_data[:,:,1] += order*np.angle(noise)
        blur_sigma = np.random.uniform(0.,0.5)
        noise_data = ndimage.gaussian_filter(noise_data,sigma=blur_sigma)
        labels_blur = ndimage.gaussian_filter(noise_labels,sigma=blur_sigma)
        noise_labels = np.where(labels_blur > .1,np.ones_like(labels_blur),np.zeros_like(labels_blur))
        if rnd_num < .3:
            out_data.append(noise_data[::-1,:,:])
            out_labels.append(noise_labels[::-1,:])
        elif rnd_num >= .3 and rnd_num < .6:
            out_data.append(noise_data[:,::-1,:])
            out_labels.append(noise_labels[:,::-1])
        elif rnd_num >= .6:
            out_data.append(noise_data[::-1,::-1,:])
            out_labels.append(noise_labels[::-1,::-1])
    return np.array(out_data),np.array(out_labels)


def gen_bandpass(freqs, ants, gain_spread=.3):
    # add in some bandpass jiggle
    HERA_NRAO_BANDPASS = np.array([-2.04689451e+06, 1.90683718e+06,
                                       -7.41348361e+05, 1.53930807e+05, -1.79976473e+04, 1.12270390e+03,
                                       -2.91166102e+01])
    HERA_BPASS_Mod = HERA_NRAO_BANDPASS*(1. + 0.00001*np.random.randn(7))
    bp_base = np.polyval(HERA_BPASS_Mod, freqs)
    window = aipy.dsp.gen_window(freqs.size, 'blackman-harris')
    _modes = np.abs(np.fft.fft(window*bp_base))
    g = {}
    for ai in ants:
        delta_bp = np.fft.ifft(noise.white_noise(freqs.size) * _modes * gain_spread)
        g[ai] = bp_base + delta_bp
    return g

def gen_delay_phs(freqs, ants, dly_rng=(-40,40)):
    phs = {}
    for ai in ants:
        dly = np.random.uniform(dly_rng[0], dly_rng[1])
        phs[ai] = np.exp(2j*np.pi*dly*freqs)
    return phs

def gen_gains(freqs, ants, gain_spread=.01, dly_rng=(-40,40)):
    bp = gen_bandpass(freqs, ants, gain_spread)
    phs = gen_delay_phs(freqs, ants, dly_rng)
    return {ai: bp[ai]*phs[ai] for ai in ants}

def apply_gains(vis, gains, bl):
    gij = gains[bl[0]] * gains[bl[1]].conj()
    gij.shape = (1,-1)
    return vis * gij

def expand_validation_dataset(data,labels):
    bloat = 5
    sh = np.shape(data)
    out_data = []
    out_labels = []
    for i in range(bloat*sh[0]):
        rnd_data_ind = np.random.randint(0,sh[0])
        spi = np.random.uniform(-2.7,-.1)
        nos_jy = np.random.rand(sh[1],sh[2])+1j*np.random.rand(sh[1],sh[2])
        nos_jy *= ((np.linspace(0.1,0.2,1024)/.1)**(spi))
        #nos_jy /= np.nanmax(np.abs(nos_jy))
        nos_jy *= random.sample(np.logspace(-3,-1),1)[0]*np.nanmean(np.abs(data[rnd_data_ind]))
        #order = np.max(np.abs(data[rnd_data_ind]))*random.sample(np.logspace(-4,-1,10),1)
        #new_gains = gen_gains(np.linspace(.1,.2,1024),[0,1])
        #new_vis = apply_gains(data[rnd_data_ind], new_gains, [0,1])
        data_ = np.copy(data[rnd_data_ind]) + nos_jy
        labels_ = np.copy(labels[rnd_data_ind])
        if np.random.rand() > .5:
            data_ = data_[::-1,:]
            labels_ = labels_[::-1,:]
        if np.random.rand() > .5:
            data_ = data_[:,::-1]
            labels_ = labels_[:,::-1]
        if np.random.rand() > .5:
            #print(np.shape([data_]))
            data_,labels_ = patchwise([data_],[labels_])
        out_data.append(data_.reshape(-1,1024))
        out_labels.append(labels_.reshape(-1,1024))
    return out_data,out_labels
        
#def load_data():

class RFIDataset():
    def __init__(self):
        print('Welcome to the HERA RFI training and evaluation dataset.')

    def load(self,tdset,vdset,batch_size,psize,hybrid=False,chtypes='AmpPhs',fold_factor=16,cut=False,patchwise_train=False):
        # load data
        if cut:
            self.cut = 14
        else:
            self.cut = 16
        self.chtypes = chtypes
        self.batch_size = batch_size
        self.pred_ct = 0 #257 isnt bad
        print('A batch size of %i has been set.' % self.batch_size)

        if vdset == 'vanilla':
            f1 = h5py.File('SimVis_2000_v911.h5','r') # 'RealVisRFI_v3.h5','r') # Load in a real dataset
        elif vdset == '':
            f1 = h5py.File('SimVis_2000_v911.h5','r')#RealVisRFI_v3.h5')
        else:
            f1 = load_pipeline_dset(vdset)
            
        #f1 = h5py.File('IDR21TrainingData.h5','r')
        if tdset == 'v5':
            f2 = h5py.File('SimVis_v5.h5','r') # Load in simulated data
        elif tdset == 'v11':
            f2 = h5py.File('SimVis_1000_v11.h5','r')
        elif tdset == 'v7':
            f2 = h5py.File('SimVis_2000_v7.h5','r')
        elif tdset == 'v8':
            f2 = h5py.File('SimVis_2000_v8.h5','r')
        elif tdset == 'v9':
            f2 = h5py.File('SimVis_1000_v9.h5','r')
        elif tdset == 'v911':
            #f2 = h5py.File('SimVis_FineTune_100.h5','r')
            f2 = h5py.File('SimVis_2000_v911.h5','r')
            #    f2 = h5py.File('RealVisRFI_v3.h5','r')
        elif tdset == 'v12':
            f2 = h5py.File('SimVis_2000_v911.h5','r')
            #f2 = h5py.File('SimVis_FineTune_100.h5','r')#SimVis_Base_1000.h5','r')
        elif tdset == 'v13':
            # This is v9 + v11 + FineTune
            f2 = h5py.File('SimVis_3000_v13.h5','r')
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
        self.dset_size = np.copy(f1_r)+np.copy(f2_s)
        
        print 'Size of real dataset: ',f1_r
        print ''
        # Cut up real dataset and labels
        #f1_r = np.shape(f1['data'])[0] # Everything after 900 doesn't look good
        samples = range(f1_r)
        rnd_ind = np.random.randint(0,f1_r)
        if cut:
            #data_real,labels_real = expand_validation_dataset(f1['data'][:f1_r,:,:],f1['flag'][:f1_r,:,:])
            data_real = f1['data'][:f1_r,:,:]
            labels_real = f1['flag'][:f1_r,:,:]
            data_sim = f2['data'][:f2_s,:,:]
            labels_sim = f2['flag'][:f2_s,:,:]
            self.data_real = np.copy(data_real)
            self.labels_real = np.copy(labels_real)
        else:
            data_real,labels_real = expand_validation_dataset(f1['data'][:f1_r,:,:],f1['flag'][:f1_r,:,:])
            #data_real = f1['data'][:f1_r,:,:]
            #labels_real = f1['flag'][:f1_r,:,:]
            self.data_real = np.copy(data_real)
            self.labels_real = np.copy(labels_real)
            data_sim = f2['data'][:f2_s,:,:]
            labels_sim = f2['flag'][:f2_s,:,:]
        #else:
        #    rnd_ind = 100#np.random.randint(0,f1_r)
        #    samples = [rnd_ind]
        time0 = time()

        if chtypes ==  'AmpPhs':
            f_real = (np.array(map(fold,data_real))[:,:,:,:,:2]).reshape(-1,self.psize,self.psize,2)
            f_real_labels = np.array(map(foldl,labels_real)).reshape(-1,self.psize,self.psize)
            # Cut up sim dataset and labels
            if patchwise_train:
                data_sim_patch,labels_sim_patch = patchwise(data_sim,labels_sim)
                data_sim = np.array(np.vstack((data_sim,data_sim_patch)))
                labels_sim = np.array(np.vstack((labels_sim,labels_sim_patch)))
                print('data_sim size: {0}'.format(np.shape(data_sim)))
                f_sim = (np.array(map(fold,data_sim))[:,:,:,:,:2]).reshape(-1,self.psize,self.psize,2)
                f_sim_labels = np.array(map(foldl,labels_sim)).reshape(-1,self.psize,self.psize)
                f_sim,f_sim_labels = expand_dataset(f_sim,f_sim_labels)
                print('Expanded training dataset size: {0}'.format(np.shape(f_sim)))
            else:
                f_sim = (np.array(map(fold,data_sim,f_factor_s,pad_s))[:,:,:,:,:2]).reshape(-1,self.psize,self.psize,2)
                f_sim_labels = np.array(map(foldl,labels_sim,f_factor_s,pad_s)).reshape(-1,self.psize,self.psize)
        elif chtypes == 'AmpPhs2' :
            f_real = np.array(map(fold,data_real,f_factor_r,pad_r)).reshape(-1,self.psize,self.psize,3)
            f_real_labels = np.array(map(foldl,labels_real,f_factor_r,pad_r)).reshape(-1,self.psize,self.psize)
            # Cut up sim dataset and labels
            f_sim = np.array(map(fold,data_sim,f_factor_s,pad_s)).reshape(-1,self.psize,self.psize,3)
            f_sim_labels = np.array(map(foldl,labels_sim,f_factor_s,pad_s)).reshape(-1,self.psize,self.psize)
        elif chtypes == 'Amp':
            f_real = (np.array(map(fold,data_real))[:,:,:,:,0]).reshape(-1,self.psize,self.psize,1)
            print('f_real: ',np.shape(f_real))
            f_real_labels = np.array(map(foldl,labels_real)).reshape(-1,self.psize,self.psize)
            if patchwise_train:
                data_sim_patch,labels_sim_patch = patchwise(data_sim,labels_sim)
                data_sim = np.array(np.vstack((data_sim,data_sim_patch)))
                labels_sim = np.array(np.vstack((labels_sim,labels_sim_patch)))
                f_sim = (np.array(map(fold,data_sim))[:,:,:,:,0]).reshape(-1,self.psize,self.psize,1)
                f_sim_labels = np.array(map(foldl,labels_sim)).reshape(-1,self.psize,self.psize)
                f_sim,f_sim_labels = expand_dataset(f_sim,f_sim_labels)
            else:
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
#        elif hybrid == 'predict':
#            print('Prediction and testing mode.')
#            self.test_data = np.asarray(f_real,dtype=d_type)
#            self.test_labels = np.asarray(f_real_labels,dtype=np.int32).reshape(-1,real_sh[1]*real_sh[2])
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
        rand_batch = random.sample(range(self.train_len),self.batch_size)
        return self.train_data[rand_batch,:,:,:],self.train_labels[rand_batch,:]

    def change_batch_size(self,new_bs):
        self.batch_size = new_bs
    
    def next_eval(self):
#        self.eval_data = np.roll(self.eval_data,self.batch_size,axis=0)
#        self.eval_labels = np.roll(self.eval_labels,self.batch_size,axis=0)
        rand_batch = random.sample(range(self.eval_len),self.batch_size) #np.random.randint(0,self.eval_len,size=self.batch_size)
        return self.eval_data[rand_batch,:,:,:],self.eval_labels[rand_batch,:]

    def next_predict(self):
        if self.chtypes == 'AmpPhs':
            f_real = (np.array(fold(self.data_real[self.pred_ct,:,:],self.cut,2))[:,:,:,:2]).reshape(-1,self.psize,self.psize,2)
            f_real_labels = np.array(foldl(self.labels_real[self.pred_ct,:,:],self.cut,2)).reshape(-1,self.psize,self.psize)
            #f_real += np.random.randn(16,68,68,2)*(1e-2)
            #f_real = ndimage.gaussian_filter(f_real,sigma=np.random.uniform(0.,.5))
        elif self.chtypes == 'Amp':
            f_real = (np.array(fold(self.data_real[self.pred_ct,:,:],self.cut,2))[:,:,:,0]).reshape(-1,self.psize,self.psize,1)
            f_real_labels = np.array(foldl(self.labels_real[self.pred_ct,:,:],self.cut,2)).reshape(-1,self.psize,self.psize)
            #f_real += np.random.randn(16,68,68,1)*(1e-2)
            #f_real = ndimage.gaussian_filter(f_real,sigma=np.random.uniform(0.,.5))
        data_return = self.data_real[self.pred_ct,:,:]

        self.pred_ct += 1#np.random.randint(np.shape(self.data_real)[0])# += 1
        return data_return,f_real,f_real_labels
    
    def random_test(self,samples):
        ind = random.sample(range(np.shape(self.eval_data)[0]),samples)#np.random.randint(0,np.shape(self.eval_data)[0],size=samples)
        if self.chtypes == 'Amp':
            ch = 1
        elif self.chtypes == 'AmpPhs':
            ch = 2
        elif self.chtypes == 'AmpPhs2':
            ch = 3
        return self.eval_data[ind,:,:,:].reshape(samples,self.psize,self.psize,ch),self.eval_labels[ind,:].reshape(samples,self.psize*self.psize)

    def get_size(self):
        return self.dset_size
