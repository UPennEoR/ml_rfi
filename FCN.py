import numpy as np
import tensorflow as tf
import h5py
import matplotlib
matplotlib.use("AGG")
import matplotlib.pyplot as plt
#from pyuvdata import UVData
from xrfi import xrfi_simple
tf.logging.set_verbosity(tf.logging.INFO)

def import_test_data(filename,bl_tup=(9,89),rescale=1.0):
    uvd = UVData()
    uvd.read_miriad(filename)
    a1,a2 = bl_tup
    data = np.nan_to_num(np.copy(uvd.get_data(a1,a2)))
    data*=rescale
    return data

def fold(data,ch_fold,labels=False):
    # We want to fold over in frequency
    # this will be done for both waterfalls and labels
    # data should be in (times,freqs) format
    ntimes,nfreqs = np.shape(data)
    dfreqs = int(nfreqs/ch_fold)
    if labels:
        data_fold = np.zeros((ntimes,nfreqs)).reshape(ch_fold,ntimes,dfreqs)
    else:
        data_fold = np.zeros((ntimes,nfreqs,2)).reshape(ch_fold,ntimes,dfreqs,2)
    for i in range(ch_fold):
        if labels:
            data_fold[i,:,:] = data[:,i*dfreqs:(i+1)*dfreqs]
        else:
            hold = np.nan_to_num(np.log10(np.abs(data[:,i*dfreqs:(i+1)*dfreqs]+np.random.rand(ntimes,dfreqs)))).real
            data_fold[i,:,:,0] = (hold - np.nanmean(hold))/np.nanmax(np.abs(hold)) #theres a better way to do this
            data_fold[i,:,:,1] = np.angle(data[:,i*dfreqs:(i+1)*dfreqs])
    return data_fold.real

def unfold(data_fold,nchans):
    ch_fold,ntimes,dfreqs = np.shape(data_fold)
    data = np.zeros_like(data_fold).reshape(60,1024)
    for i in range(ch_fold):
        data[:,i*dfreqs:(i+1)*dfreqs] = data_fold[i,:,:]
    return data

def stacked_layer(input_layer,num_filter_layers,kt,kf,activation,stride,pool,bnorm=True):
    """
    Creates a 3x stacked layer of convolutional layers ###
    """
    conva = tf.layers.conv2d(inputs=input_layer,
                             filters=num_filter_layers,
                             kernel_size=[kt,kf],
                             padding="same",
                             activation=activation)
    #sh = input_layer.get_shape().as_list()
    #conva_reshape = tf.reshape(conva, [-1,sh[1]*sh[2]*sh[3]])
    #fc = tf.layers.dense(conva_reshape, units=sh[1]*sh[2]*sh[3], activation=tf.nn.elu)
    #fc_reshape = tf.reshape(fc, tf.shape(conva))
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

def t_layer(input_layer,num_filter_layers,kt,kf,activation):
    conv = tf.layers.conv2d(inputs=input_layer,
                                 filters=num_filter_layers,
                                 kernel_size=[kt,kf],
                                 padding="same",
                                 activation=activation)
    return conv
        
def upsample(input_layer,out_size):
    """
    Creates an upsampling layer which passes an input layer through two fully connected 
    layers and then into a convolutional layer that expands the filter dimension for
    reshaping into an upsampled output
    """
    sh = input_layer.get_shape().as_list()
    #print sh,out_size
    f_layers = int((1.*out_size[0]*out_size[1]*out_size[2])/(1.*sh[2]*sh[1]))
    #print 'f_kayers: ',f_layers
    layer_reshape = tf.reshape(input_layer, [-1,sh[1]*sh[2]*sh[3]])
    fc_layer_reshape = tf.reshape(layer_reshape, [-1,sh[1],sh[2],sh[3]])
    upsamp = tf.layers.conv2d(inputs=fc_layer_reshape,
                             filters=f_layers,
                             kernel_size=[3,3],
                             padding="same",
                             activation=tf.nn.relu)    
    upsamp_reshape = tf.reshape(upsamp, [-1,out_size[0],out_size[1],out_size[2]])
    return tf.contrib.layers.batch_norm(upsamp_reshape,scale=True)

def dense(input_layer,out_size):
    """
    Combines 4 fully connected layers for a dense output after the conv. stacked
    and upsampling layers
    """
    sh = input_layer.get_shape().as_list()
    try:
        scale = (out_size[0]*out_size[1]*out_size[2])#/(sh[1]*sh[2]*sh[3])
    except:
        scale = out_size[0]*out_size[1]
    #print 'scale: ',scale
    try:
        input_layer_reshape = tf.reshape(input_layer, [-1,sh[1]*sh[2]*sh[3]])
    except:
        input_layer_reshape = tf.reshape(input_layer, [-1,sh[1]*sh[2]])

    fc3 = tf.layers.dense(input_layer_reshape, units=scale, activation=tf.nn.elu)
    try:
        fc3_reshape = tf.reshape(fc3, [-1,out_size[0],out_size[1],out_size[2]])
    except:
        fc3_reshape = tf.reshape(fc3, [-1,out_size[0],out_size[1]])
    return fc3_reshape

def dense2(input_layer, out_size):
    """                                                                                                                                              
    Combines 4 fully connected layers for a dense output after the conv. stacked                                                                     
    and upsampling layers                                                                                                                            
    """
    sh = input_layer.get_shape().as_list()
    try:
        scale = out_size[0]*out_size[1]
    except:
        scale = out_size[0]
    #print 'scale: ',scale
    try:
        input_layer_reshape = tf.reshape(input_layer, [-1,sh[1]*sh[2]])
    except:
        input_layer_reshape = tf.reshape(input_layer, [-1,sh[1]])

    fc3 = tf.layers.dense(input_layer_reshape, units=scale, activation=tf.nn.elu)
    try:
        fc3_reshape = tf.reshape(fc3, [-1,out_size[0],out_size[1]])
    except:
        fc3_reshape = tf.reshape(fc3, [-1,out_size[0]])
    return fc3_reshape

def batch_norm(layer, out_size):
 #   layer_reshape = tf.reshape(layer, [-1,out_size[0],out_size[1],out_size[2]])
    layer_norm = tf.contrib.layers.batch_norm(layer,scale=True)
 #   layer_out = tf.reshape(layer_norm, [-1,out_size[0],out_size[1]])
    return layer_norm

def cnn(features,labels,mode):
    """
    Model for CNN

    features: visibility array
    labels: RFI flag array
    mode: used by tensorflow to distinguish training and testing
    """ 

    activation=tf.nn.relu # exponential linear unit
    # kernel size
    kt = 3 #
    kf = 3 #

    # 4D tensor: batch size, height (ntimes), width (nfreq), channels (1)
    input_layer = tf.reshape(features["x"],[-1,60,64,2])

    # 3x stacked layers similar to VGG
    #in: 60,64,1
    slayer1 = stacked_layer(input_layer,64,kt,kf,activation,[2,2],[2,2],bnorm=True)
#    slayer1_ = t_layer(slayer1,4,1,1,activation)
    #1: 30,32,2
    slayer2 = stacked_layer(slayer1,128,kt,kf,activation,[2,2],[2,2],bnorm=True)
#    slayer2_ = t_layer(slayer2,16,1,1,activation)
    #2: 15,16,4
    slayer3 = stacked_layer(slayer2,4*192,kt,kf,activation,[3,2],[3,2],bnorm=True) 
#    slayer3_ = t_layer(slayer3,192,1,1,activation)
    #3: 5,8,16
    slayer4 = stacked_layer(slayer3,4*384,3,3,activation,[1,1],[1,1],bnorm=True)    
    #    slayer4_ = t_layer(slayer4,384,1,1,activation)
    #4 6,32,16
    slayer5 = stacked_layer(slayer4,1920,1,1,activation,[1,1],[1,1],bnorm=True)
    #5 5,16,256
    # Upsampleeeeeeee

#    tf.summary.image('Output_Layer',tf.reshape(final_conv[:,:,:,0], [-1,60,1024,1]))
#    upsamp2_final = tf.reshape(upsample(slayer2, [60,64,2]), [-1,60*64,2])
    upsamp = tf.layers.conv2d(inputs=slayer5,
                              filters=192,
                              kernel_size=[1,1],
                              padding="same",
                              activation=tf.nn.relu)

    #upsamp = batch_norm(upsamp,[5,8,192])
    final_conv = tf.reshape(upsamp,[-1,60*64,2])#,upsamp2_final)
    #final_conv = batch_norm(final_conv, [60*64,2])
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
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=.01)
        train_op = optimizer.minimize(loss=loss,global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode=mode,loss=loss,train_op=train_op)

    if mode == tf.estimator.ModeKeys.PREDICT:
        print 'Mode is predict.'
        return tf.estimator.EstimatorSpec(mode=mode,predictions=predictions)


    eval_metric_ops = {
        "accuracy": tf.metrics.accuracy(labels=labels,predictions=predictions['classes'])        
}
    return tf.estimator.EstimatorSpec(mode=mode,loss=loss,eval_metric_ops=eval_metric_ops)

def timediff(data):
    tdiff = np.copy(data)
    times,freqs= np.shape(data)
    for i in range(times-1):
        tdiff[i,:] = data[i+1,:] - data[i,:]
    tdiff /= np.nanmax(np.abs(tdiff)[~np.isinf(np.abs(tdiff))])
    return tdiff

def freqdiff(data):
    fdiff = np.copy(data)
    times,freqs= np.shape(data)
    for i in range(freqs-1):
        fdiff[:,i] = data[:,i+1] - data[:,i]
    fdiff /= np.nanmax(np.abs(fdiff)[~np.isinf(np.abs(fdiff))])
    return fdiff

def preprocess(data):
   # data array size should be (batch,time,freq)                                                                                   
   data_a = np.nan_to_num(np.copy(data))
   batch,t_num,f_num = np.shape(data)
   # initialize output array                                                                                                   
   data_out = np.zeros((batch,t_num,f_num,1))
   for b in range(batch):
       data_ = np.copy(data_a[b,:,:])
 #      data_ = np.nan_to_num(np.log10(np.abs(data_)))
#       data_ /= np.nanmax(np.abs(data_))
#       data_ -= np.nanmean(data_)
#       data_ = np.log10(np.abs(data_))
#       data_out[b,:,:,0] = data_
       data_angle = np.angle(np.copy(data_a[b,:,:]))
#       data_angle -= np.mean(data_angle)
       data_out[b,:,:,0] = data_angle
#       data_out[b,:,:,1] = xrfi_simple(np.abs(data_))
   return np.nan_to_num(data_out)


def main(args):
#    tset_size = 300#3584
#    trainlen = 200#2900
    # load data
    f1 = h5py.File('RealVisRFI_v3.h5','r')
    f2 = h5py.File('SimVisRFI_15_120_v3.h5','r')
    ## Pull eval set out of real data before mixing
    f1_0 = np.shape(f1['data'])[0]
    f1_eval = f1['data'][:int(f1_0*.2),:,:] # save 20% for eval
    f1_labels_eval = f1['flag'][:int(f1_0*.2),:,:]
    eval0 = np.shape(f1_eval)[0]

    f1_train = f1['data'][int(f1_0*.2):,:,:] # save 80% for training
    f1_labels_train = f1['flag'][int(f1_0*.2):,:,:]

    f_data = np.vstack((f1['data'][300:,:,:],f2['data']))
    f_labels = np.vstack((f1['flag'][300:,:,:],f2['flag']))
    train0 = np.shape(f_data)[0]

    f_data_c = fold(f_data[100,:,:],16)
    f_labels_c = fold(f_labels[100,:,:],16,labels=True)

    f1_eval_c = fold(f1['data'][0,:,:],16)#f_data[199,:,:],16)
    f1_eval_labels_c = fold(f1['flag'][0,:,:],16,labels=True)#f_labels[199,:,:],16,labels=True)
    #### Carve up 50 different waterfall visibilities                                                                                                                       
    for c in range(300):
        f_data_c = np.vstack((f_data_c,fold(f_data[c,:,:],16)))
        f_labels_c = np.vstack((f_labels_c,fold(f_labels[c,:,:],16,labels=True)))
        #if c > 100:
        f1_eval_c = np.vstack((f1_eval_c,fold(f1['data'][c+1,:,:],16)))
        f1_eval_labels_c = np.vstack((f1_eval_labels_c,fold(f1['flag'][c+1,:,:],16,labels=True)))

#    print 'Shape of folded dataset: ',np.shape(ct)
    train0 = np.shape(f_data_c)[0]
    eval1 = np.shape(f1_eval_c)[0]
    steps = 100*train0    

    # We want to add real data in between sim data w/ xrfi flags

    all_data = f_data_c#preprocess(f_data)
    train_data = np.asarray(all_data)
    train_data = np.asarray(train_data, dtype=np.float32)
    train_labels = np.reshape(np.asarray(f_labels_c), (train0, 64*60))
    train_labels = np.asarray(train_labels, dtype=np.int32)

    eval_data = f1_eval_c#np.asarray(preprocess(f1_eval))
    eval_data = np.asarray(eval_data,dtype=np.float32)
    eval_labels = np.asarray(f1_eval_labels_c,dtype=np.int32)
    eval_labels = np.reshape(eval_labels, (eval1, 64*60))

    real_data = fold(f1['data'][213,:,:],16) #eval_data[:16,:,:] #Just pull a real visibility from the eval set
    real_data = np.asarray(real_data, dtype=np.float32)
    real_labels = fold(f1['flag'][213,:,:],16,labels=True)
    real_labels = np.asarray(real_labels, dtype=np.int32)

    # create Estimator
    rfiCNN = tf.estimator.Estimator(model_fn=cnn,model_dir='./checkpoint_Patch3/')

    train_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x":train_data},
        y=train_labels,
        batch_size=10,
        num_epochs=1000,
        shuffle=True,
    )

    rfiCNN.train(input_fn=train_input_fn, steps=steps)

    eval_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x":eval_data},
        y=eval_labels,
        num_epochs=100,
        shuffle=False)	


    test_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x":real_data},
        shuffle=False
    )
# Eval mode is turned off for now, it requests a substantial amount of memory for
# the current version of this CNN

    eval_results = rfiCNN.evaluate(input_fn=eval_input_fn)
    print(eval_results)

    rfiPredict = rfiCNN.predict(input_fn=test_input_fn)
    #cnn_flags = np.zeros((60,1024))
    for i,predicts in enumerate(rfiPredict):
        print np.shape(i),np.shape(predicts['probabilities'])
        if i == 0:
            cnn_flags = predicts['classes'].reshape(1,60,64)
        else:
            cnn_flags = np.vstack((cnn_flags,predicts['classes'].reshape(1,60,64)))
    print 'Shape of CNN flags: ',np.shape(cnn_flags)
    print 'Shape of Real flags: ',np.shape(real_labels)
    cnn_flags = unfold(cnn_flags,1024)
    real_labels = real_labels.reshape(16,60,64)
    real_labels = unfold(real_labels,1024)
#        idxs = np.where(train_labels[1,:] == 1)
#        print predicts['probabilities'][idxs]

    plt.subplot(411)
    plt.imshow(cnn_flags,aspect='auto')
    plt.title('Predicted Flags')
    plt.colorbar()

    plt.subplot(412)
    plt.imshow(real_labels.reshape(-1,1024),aspect='auto')
    plt.title('XRFI Flags')
    plt.colorbar()

    plt.subplot(413)
    plt.imshow(np.log10(np.abs(f1['data'][213,:,:])),aspect='auto')
    plt.colorbar()
    plt.title('Vis. Log Normalized Amp.')

    plt.subplot(414)
    plt.imshow(np.angle(f1['data'][213,:,:]),aspect='auto')
    plt.colorbar()
    plt.title('Vis. Phs.')
        
    plt.savefig('RealData.png')
        
    cnn_flags = np.logical_not(cnn_flags)#np.logical_not(predicts['classes'].reshape(-1,1024))
    xrfi_flags = np.logical_not(real_labels.reshape(-1,1024))
    plt.subplot(211)
    plt.imshow(np.log10(np.abs(f1['data'][213,:,:]*cnn_flags)),aspect='auto')
    plt.title('Predicted Flags Applied')
        
    plt.subplot(212)
    plt.imshow(np.log10(np.abs(f1['data'][213,:,:]*xrfi_flags)),aspect='auto')
    plt.title('XRFI Flags Applied')

    plt.savefig('VisApplied.png')

if __name__ == "__main__":
    tf.app.run()
