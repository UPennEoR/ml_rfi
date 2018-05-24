import numpy as np
import tensorflow as tf
import h5py
import matplotlib
matplotlib.use("AGG")
import matplotlib.pyplot as plt
#from pyuvdata import UVData
#from hera_qm import xrfi
tf.logging.set_verbosity(tf.logging.INFO)

def import_test_data(filename,bl_tup=(9,89),rescale=1.0):
    uvd = UVData()
    uvd.read_miriad(filename)
    a1,a2 = bl_tup
    data = np.nan_to_num(np.copy(uvd.get_data(a1,a2)))
    data*=rescale
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

def upsample(input_layer,out_size):
    """
    Creates an upsampling layer which passes an input layer through two fully connected 
    layers and then into a convolutional layer that expands the filter dimension for
    reshaping into an upsampled output
    """
    sh = input_layer.get_shape().as_list()
    print sh,out_size
    f_layers = int((1.*out_size[0]*out_size[1]*out_size[2])/(1.*sh[2]*sh[1]))
    print 'f_kayers: ',f_layers
    layer_reshape = tf.reshape(input_layer, [-1,sh[1]*sh[2]*sh[3]])
    fc_layer_reshape = tf.reshape(layer_reshape, [-1,sh[1],sh[2],sh[3]])
    upsamp = tf.layers.conv2d(inputs=fc_layer_reshape,
                             filters=f_layers,
                             kernel_size=[3,3],
                             padding="same",
                             activation=tf.nn.elu)    
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
    print 'scale: ',scale
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

def cnn(features,labels,mode):
    """
    Model for CNN

    features: visibility array
    labels: RFI flag array
    mode: used by tensorflow to distinguish training and testing
    """ 

    activation=tf.nn.elu # exponential linear unit
    # kernel size
    kt = 2 #
    kf = 2 #

    # 4D tensor: batch size, height (ntimes), width (nfreq), channels (1)
    input_layer = tf.reshape(features["x"],[-1,60,1024,2])

    # 3x stacked layers similar to VGG
    #in: 60,1024,2
    slayer1 = stacked_layer(input_layer,8,kt,kf,activation,[5,4],[5,4],bnorm=True)
    #1: 12,256,2
    slayer2 = stacked_layer(slayer1,16,3,3,activation,[2,4],[2,4],bnorm=True)
    #2: 6,64,4
    slayer3 = stacked_layer(slayer2,32,3,3,activation,[1,2],[1,2],bnorm=True) 
    #3: 6,32,8
    slayer4 = stacked_layer(slayer3,64,3,3,activation,[1,1],[1,1],bnorm=True)    
    #4 6,32,16
#    slayer5 = stacked_layer(slayer4,32,1,1,activation,[1,1],[1,1],bnorm=True)
    #5 5,16,256

    tf.summary.image('Slayer4',tf.reshape(slayer4[:,:,:,0], [-1,6,32,1]))

    # FULLY CONNECT@!#!@!
    fc1 = dense(slayer4, (15,256)) #5,64
    fc2 = dense(fc1, (15,256)) #15,128
    fc3 = tf.nn.dropout(dense(fc2, (15,256)), keep_prob=.7)
    fc4 = dense(fc3, (15,256))
    fc5 = tf.nn.dropout(dense(fc4, (30,512)), keep_prob=.7) #30,256
    fc6 = dense(fc5, (30, 512))
    fc7 = dense(fc6, (30, 512, 2))

    tf.summary.image('Fully_Connected_5',tf.reshape(fc7[:,:,:,0], [-1,30,512,1]))

    # Upsampleeeeeeee
    ulayer0 = upsample(fc7, (30,512,32))
    ulayer1 = tf.nn.dropout(upsample(ulayer0,(60,1024,16)),keep_prob=.7)
    ulayer2 = upsample(ulayer1, (120,2048,8)) # hyper-resolve factor of 2
    ulayer3 = upsample(ulayer2, (120,2048,4)) # keep hyper-resolve but drop channels 
    ulayer4 = tf.layers.max_pooling2d(inputs=ulayer3,
                                      pool_size=[2,2],
                                      strides=[2,2]) #downsample back to proper size

    final_conv = upsample(ulayer4, (60,1024,2))

    tf.summary.image('Output_Layer',tf.reshape(final_conv[:,:,:,0], [-1,60,1024,1]))

    final_conv = tf.reshape(final_conv, [-1,60*1024,2])

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
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=.1)
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
   data_out = np.zeros((batch,t_num,f_num,2))
   for b in range(batch):
       data_ = np.copy(data_a[b,:,:])
 #      data_ = np.nan_to_num(np.log10(np.abs(data_)))
       data_ /= np.nanmax(np.abs(data_))
       data_ -= np.nanmean(data_)
       data_ = np.log10(np.abs(data_))
       data_out[b,:,:,0] = data_
       data_angle = np.angle(np.copy(data_a[b,:,:]))
       data_angle -= np.mean(data_angle)
       data_out[b,:,:,1] = data_angle
   return np.nan_to_num(data_out)


def main(args):
    tset_size = 288
    trainlen = 220
    steps = 10000000
    # load data
    f = h5py.File('RealVisRFI_v2.h5','r')
    # We want to add real data in between sim data w/ xrfi flags
    rdf = h5py.File('RealData.h5','r')
    #real_data = preprocess(rdf['data']) #import_test_data('zen.2457555.40356.xx.HH.uvcT').reshape(1,-1,1024) 
    #real_flags = rdf['flags']#xrfi.xrfi(np.abs(rdf['data'].reshape(-1,1024)))
    all_data = preprocess(f['data'])
    train_data = np.asarray(all_data)[:trainlen,:,:,:]
    train_data = np.asarray(train_data, dtype=np.float32)
    train_labels = np.reshape(np.asarray(f['flag'])[:trainlen,:,:], (trainlen, 1024*60))
    train_labels = np.asarray(train_labels, dtype=np.int32)
    eval_data = np.asarray(all_data)[trainlen:,:,:,:]
    eval_data = np.asarray(eval_data,dtype=np.float32)
    eval_labels = np.asarray(f['flag'],dtype=np.int32)[trainlen:,:,:]
    eval_labels = np.reshape(eval_labels, (tset_size-trainlen, 1024*60))

    #real_data = np.asarray(f['data'])[100,:,:]
    real_data = np.asarray(all_data)[101,:,:,:]
    real_data = np.asarray(real_data, dtype=np.float32)
    real_labels = np.reshape(np.asarray(f['flag'])[101,:,:], (60,1024))
    real_labels = np.asarray(real_labels, dtype=np.int32)

    #real_data_ = np.asarray(real_data, dtype=np.float32)
    #real_data = real_data_.reshape(-1,1024)
    rd_plot = np.asarray(rdf['data'], dtype=complex).reshape(-1,1024)
    # create Estimator
    rfiCNN = tf.estimator.Estimator(model_fn=cnn,model_dir='/pylon5/as5fp5p/jkerriga/checkpoint/')

    train_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x":train_data},
        y=train_labels,
        batch_size=15,
        num_epochs=1000,
        shuffle=True,
    )

    rfiCNN.train(input_fn=train_input_fn, steps=steps)

    eval_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x":eval_data},
        y=eval_labels,
        num_epochs=1000,
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
    for i,predicts in enumerate(rfiPredict):
        print np.shape(i),np.shape(predicts['probabilities'])
        idxs = np.where(train_labels[1,:] == 1)
        print predicts['probabilities'][idxs]

        plt.subplot(411)
        plt.imshow(predicts['classes'].reshape(-1,1024),aspect='auto')
        plt.title('Predicted Flags')
        plt.colorbar()

        plt.subplot(412)
        plt.imshow(real_labels.reshape(-1,1024),aspect='auto')
        plt.title('XRFI Flags')
        plt.colorbar()

        plt.subplot(413)
	plt.imshow(real_data[:,:,0],aspect='auto')
        plt.colorbar()
        plt.title('Vis. Log Normalized Amp.')

        plt.subplot(414)
        plt.imshow(real_data[:,:,1],aspect='auto')
        plt.colorbar()
        plt.title('Vis. Phs.')
        

        plt.savefig('RealData.png')

if __name__ == "__main__":
    tf.app.run()
