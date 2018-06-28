import numpy as np
import tensorflow as tf
import h5py
import matplotlib
matplotlib.use("AGG")
import matplotlib.pyplot as plt
from xrfi import xrfi_simple
tf.logging.set_verbosity(tf.logging.INFO)
from time import time


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
    LOGabsX = np.nan_to_num(np.log10(np.abs(X+(1e-5)*np.random.rand(sh[0],sh[1])))).real
    return (LOGabsX-np.nanmean(LOGabsX))/np.nanmax(np.abs(LOGabsX))

def foldl(data,ch_fold=16):
    """
    Folding function for carving up a waterfall visibility flags for prediction in the FCN.
    """
    sh = np.shape(data)
    _data = data.T.reshape(ch_fold,sh[1]/ch_fold,-1)
    _DATA = np.array(map(transpose,_data))
    _DATApad = np.array(map(pad,_DATA))
    return _DATApad

def pad(data):
    sh = np.shape(data)
    t_pad = (sh[1] - sh[0])
    data_pad = np.pad(data,pad_width=((t_pad,t_pad),(2,2)),mode='constant')
    return data_pad

def unpad(data):
    sh = np.shape(data)
    t_unpad = sh[0]
    return data[2:,2:][:-2,:-2][2:,:][:-2,:]
                      
def fold(data,ch_fold=16):
    """
    Folding function for carving waterfall visibilities with additional normalized log 
    and phase channels.
    """
    sh = np.shape(data)
    _data = data.T.reshape(ch_fold,sh[1]/ch_fold,-1)
    _DATA = np.array(map(transpose,_data))
    _DATApad = np.array(map(pad,_DATA))
    DATA = np.stack((np.array(map(normalize,_DATApad)),np.angle(_DATApad)),axis=-1)
    return DATA

def unfoldl(data_fold,nchans=1024):
    """
    Unfolding function for recombining the carved label (flag) frequency windows back into a complete 
    waterfall visibility.
    """
    data_unpad = np.array(map(unpad,data_fold))
    ch_fold,ntimes,dfreqs = np.shape(data_unpad)
    data_ = np.array(map(transpose,data_unpad))
    _data = data_.reshape(ch_fold*dfreqs,ntimes).T
    return _data
    
def stacked_layer(input_layer,num_filter_layers,kt,kf,activation,stride,pool,bnorm=True):
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
    
    convb = tf.layers.conv2d(inputs=conva,
                             filters=num_filter_layers,
                             kernel_size=[kt,kf],
                             padding="same",
                             activation=activation)

    convc = tf.layers.conv2d(inputs=convb,
                             filters=num_filter_layers,
                             kernel_size=[kt,kf],
                             padding="same",
                             activation=activation)
    if bnorm:
    	bnorm_conv = tf.contrib.layers.batch_norm(convc,scale=True)
    else:
    	bnorm_conv = convc

    pool = tf.layers.max_pooling2d(inputs=bnorm_conv,
                                    pool_size=pool,
                                    strides=stride)
    return pool

def fcn(features,labels,mode):
    """
    Model for Deep Fully Convolutional Net RFI Flagger

    features: Visibility Array (batch#, time, freq, channel)
    labels: RFI flag array (batch#, time, freq)
    mode: used by tensorflow to distinguish training, evaluation, and testing
    """ 

    activation=tf.nn.relu # rectified exponential linear activation unit
    # kernel size
    kt = 3 
    kf = 3 

    # 4D tensor: batch size, height (ntimes), width (nfreq), channels (norm. log. amp, phase)
    input_layer = tf.reshape(features["x"],[-1,68,68,2]) # this can be made size indep.

    # 3x stacked layers similar to VGG
    #in: 68,68,2
    slayer1 = stacked_layer(input_layer,68,kt,kf,activation,[2,2],[2,2],bnorm=True)

    #1: 34,34,68
    slayer2 = stacked_layer(slayer1,2*68,kt,kf,activation,[2,2],[2,2],bnorm=True)

    #2: 17,17,136
    slayer3 = stacked_layer(slayer2,4*68,kt,kf,activation,[2,2],[2,2],bnorm=True) 

    #3: 8,8,272
    slayer4 = stacked_layer(slayer3,8*68,kt,kf,activation,[2,2],[2,2],bnorm=True)    

    #4 8,8,544
    slayer5 = stacked_layer(slayer4,16*68,1,1,activation,[1,1],[1,1],bnorm=True)

    #5 8,8,1088
    # Transpose convolution (deconvolve)
    upsamp = tf.layers.conv2d_transpose(slayer5,filters=2,kernel_size=[65,65],activation=activation)
    print 'Shape of upsamp: ',np.shape(upsamp)
    final_conv = tf.reshape(upsamp,[-1,68*68,2])

    # Grab some output weight info for tensorboard
    #tf.summary.image('FullyConnected_stacked_layer5',tf.reshape(final_conv[0,:,:], [-1,60,64,2]))

    predictions = {
        "classes": tf.argmax(input=final_conv, axis=2),
        "probabilities": tf.nn.softmax(final_conv,name="softmax_tensor")
    }

    try:    
        loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=final_conv)
    except:
        pass

    if mode == tf.estimator.ModeKeys.TRAIN:
        print 'Mode is train.'
        optimizer = tf.train.AdamOptimizer(learning_rate=.0001) #tf.train.GradientDescentOptimizer(learning_rate=.1)
        train_op = optimizer.minimize(loss=loss,global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode=mode,loss=loss,train_op=train_op)

    if mode == tf.estimator.ModeKeys.PREDICT:
        print 'Mode is predict.'
        return tf.estimator.EstimatorSpec(mode=mode,predictions=predictions)

    eval_metric_ops = {
        "accuracy": tf.metrics.accuracy(labels=labels,predictions=predictions['classes']),
        "F1_score": tf.metrics.recall(labels=labels,predictions=predictions['classes'])
}
    return tf.estimator.EstimatorSpec(mode=mode,loss=loss,eval_metric_ops=eval_metric_ops)

def main(args):
    # load data
    f1 = h5py.File('RealVisRFI_v3.h5','r') # Load in a real dataset 
    f2 = h5py.File('SimVisRFI_15_120_v3.h5','r') # Load in simulated data

    train = True
    evaluate = True
    test = True

    # We want to augment our training dataset with the entirety of the simulated data
    # but with only half of the real data. The remaining real data half will become
    # the evaluation dataset
    
    f1_r = 900 #np.shape(f1['data'])[0]
    f2_s = np.shape(f2['data'])[0]

    print 'Size of real dataset: ',f1_r
    print ''
    # Cut up real dataset and labels
    if train|evaluate:
        f1_r = 900 #np.shape(f1['data'])[0] # Everything after 900 doesn't look good
        samples = range(f1_r)
        rnd_ind = np.random.randint(0,f1_r)
    else:
        rnd_ind = np.random.randint(0,f1_r)
        samples = [rnd_ind]
    time0 = time()
    f_real = np.array(map(fold,f1['data'][:f1_r,:,:])).reshape(-1,68,68,2)
    f_real_labels = np.array(map(foldl,f1['flag'][:f1_r,:,:])).reshape(-1,68,68)
    print 'Training dataset loaded.'

    # Cut up sim dataset and labels
    if train|evaluate:
        f_sim = np.array(map(fold,f2['data'][:f2_s,:,:])).reshape(-1,68,68,2)
        f_sim_labels = np.array(map(foldl,f2['flag'][:f2_s,:,:])).reshape(-1,68,68)
        print 'Simulated training dataset loaded.'

    real_sh = np.shape(f_real)

    # Format evaluation dataset
    if evaluate:
        eval_data = np.asarray(f_real[:,:,:,:],dtype=np.float32)
        eval_labels = np.asarray(f_real_labels[:,:,:],dtype=np.int32).reshape(-1,real_sh[1]*real_sh[2])

    # Format training dataset
    if train|evaluate:
        train_data = np.asarray(f_sim,dtype=np.float32)#np.asarray(np.vstack((f_sim,f_real[:real_sh[0]/2,:,:,:])),dtype=np.float32)
        train_labels = np.asarray(f_sim_labels,dtype=np.int32).reshape(-1,real_sh[1]*real_sh[2])#np.asarray(np.vstack((f_sim_labels,f_real_labels[:real_sh[0]/2,:,:])),dtype=np.int32).reshape(-1,real_sh[1]*real_sh[2])

        train0 = np.shape(train_data)[0]
        eval1 = np.shape(eval_data)[0]
        steps = 100*train0

    # Format a single test dataset
    if test:
        test_data = np.asarray(fold(f1['data'][rnd_ind,:,:],16), dtype=np.float32) # Random real visibility for testing
        test_labels = np.asarray(foldl(f1['flag'][rnd_ind,:,:],16), dtype=np.int32).reshape(-1,real_sh[1]*real_sh[2])

    # create Estimator
    rfiFCN = tf.estimator.Estimator(model_fn=fcn,model_dir='./checkpoint_Patch4_SimTrain/')

    if train:
        train_input_fn = tf.estimator.inputs.numpy_input_fn(
            x={"x":train_data},
            y=train_labels,
            batch_size=5,
            num_epochs=1000,
            shuffle=True,
        )

    if evaluate:
        eval_input_fn = tf.estimator.inputs.numpy_input_fn(
            x={"x":eval_data},
            y=eval_labels,
            num_epochs=100,
            shuffle=False
        )	

    if test:
        test_input_fn = tf.estimator.inputs.numpy_input_fn(
            x={"x":test_data},
            shuffle=False
        )
    
    if train:
        rfiFCN.train(input_fn=train_input_fn, steps=steps)

    if evaluate:
        eval_results = rfiFCN.evaluate(input_fn=eval_input_fn)
        try:
            output = open('eval_results.txt','w')
            output.write(eval_results)
        except:
            print 'No eval results saved.'
        print(eval_results)

    if test:
        rfiPredict = rfiFCN.predict(input_fn=test_input_fn)

    # Predict on the test dataset where labels are hidden
    #print 'Prediction dataset is size: ',np.shape(train_data)[0]
    obs_flags = np.zeros((16,68,68))#np.shape(train_data)[0],60,64))
    probs = np.zeros((16,68,68))
    for i,predicts in enumerate(rfiPredict):
        print i,np.shape(predicts['probabilities'])
        obs_flags[i,:,:] = predicts['classes'].reshape(68,68)
        probs[i,:,:] = predicts['probabilities'][:,1].reshape(68,68)

    obs_flags = obs_flags.reshape(16,68,68)
    probs = probs.reshape(16,68,68)
    cnn_flags = unfoldl(obs_flags)
    probs = unfoldl(probs)

    print time() - time0
    print 'Shape of CNN flags: ',np.shape(cnn_flags)
    print 'Shape of Test flags: ',np.shape(test_labels)
    test_labels = unfoldl(test_labels.reshape(-1,68,68),1024)
    plt.subplot(411)
    plt.imshow(cnn_flags,aspect='auto')
    plt.title('Predicted Flags')
    plt.colorbar()

    plt.subplot(412)
    plt.imshow(test_labels.reshape(-1,1024),aspect='auto')
    plt.title('XRFI Flags')
    plt.colorbar()

    plt.subplot(413)
    plt.imshow(np.log10(np.abs(f1['data'][rnd_ind,:,:])),aspect='auto')
    plt.colorbar()
    plt.title('Vis. Log Normalized Amp.')

    plt.subplot(414)
    plt.imshow(np.angle(f1['data'][rnd_ind,:,:]),aspect='auto')
    plt.colorbar()
    plt.title('Vis. Phs.')
    plt.tight_layout()
    plt.savefig('RealData.pdf')


    plt.subplot(311)
    plt.imshow(probs,aspect='auto')
    plt.title('Probability of RFI')
    plt.colorbar()

    plt.subplot(312)
    plt.imshow(np.log10(np.abs(f1['data'][rnd_ind,:,:])),aspect='auto')
    plt.colorbar()
    plt.title('Vis. Log10 Amp.')

    plt.subplot(313)
    plt.imshow(np.angle(f1['data'][rnd_ind,:,:]),aspect='auto')
    plt.colorbar()
    plt.title('Vis. Phs.')
    plt.tight_layout()
    plt.savefig('ProbVSinputs.pdf')

    
    cnn_flags = np.logical_not(cnn_flags)
    xrfi_flags = np.logical_not(test_labels.reshape(-1,1024))
    plt.subplot(211)
    plt.imshow(np.log10(np.abs(f1['data'][rnd_ind,:,:]*cnn_flags)),aspect='auto')
    plt.title('Predicted Flags Applied')
        
    plt.subplot(212)
    plt.imshow(np.log10(np.abs(f1['data'][rnd_ind,:,:]*xrfi_flags)),aspect='auto')
    plt.title('XRFI Flags Applied')

    plt.savefig('VisApplied.pdf')

if __name__ == "__main__":
    tf.app.run()
