import numpy as np
import tensorflow as tf
import h5py
import matplotlib
matplotlib.use("AGG")
import matplotlib.pyplot as plt
from pyuvdata import UVData
from hera_qm import xrfi
tf.logging.set_verbosity(tf.logging.INFO)

### This code needs refactoring
def import_test_data(filename,bl_tup=(9,89),rescale=1.0):
    uvd = UVData()
    uvd.read_miriad(filename)
    a1,a2 = bl_tup
    data = np.nan_to_num(np.copy(uvd.get_data(a1,a2)))
    data*=rescale
    return data

def cnn(features,labels,mode):
    """
    Model for CNN

    features: visibility array
    labels: RFI flag array
    mode: used by tensorflow to distinguish training and testing
    """ 

    activation=tf.nn.elu
            # kernel size
    kt = 7 # success of finding RFI in real data seems to strongly depend on these 
    kf = 7 # 7,7 seems like the ideal spot so far
    ks1 = 10 # 1
    ks2 = 10 # 2
    ks3 = 10 # 3
    ks4 = 10 # 4
    # 4D tensor: batch size, height (ntimes), width (nfreq), channels (1)
    input_layer = tf.reshape(features["x"],[-1,60,1024,2])

    chan_conv = tf.layers.conv2d(inputs=input_layer,filters=30,kernel_size=[kt,kf],padding="same",activation=activation)
    chan_conv = tf.contrib.layers.batch_norm(chan_conv,scale=True)
    chan_pool = tf.layers.max_pooling2d(inputs=chan_conv,pool_size=[1,2],strides=2)
    print np.shape(chan_pool)
    # Conv. layer 1
    # in: [-1,60,1024,30]
    # out: [-1,60,1024,16]
    conv1a = tf.layers.conv2d(inputs=chan_pool,
                             filters=16,
                             kernel_size=[kt,kf],
                             padding="same",
                             activation=activation)

    conv1b = tf.layers.conv2d(inputs=conv1a,
                             filters=16,
                             kernel_size=[kt,kf],
                             padding="same",
                             activation=activation)

    conv1c = tf.layers.conv2d(inputs=conv1b,
                             filters=16,
                             kernel_size=[kt,kf],
                             padding="same",
                             activation=activation)
    conv1c = tf.contrib.layers.batch_norm(conv1c,scale=True)
    # Pool layer 1 (max pooling), 2x2 filter with stride of 2
    # in: [-1,60,1024,16]
    # out: [-1,30,512,16]
    pool1 = tf.layers.max_pooling2d(inputs=conv1c,
                                    pool_size=[2,2],
                                    strides=2)
    print np.shape(pool1)
    # Conv. layer 2
    # in: [-1,30,512,16]
    # out: [-1,30,512,32]

    conv2a = tf.layers.conv2d(inputs=pool1,
                             filters=32,
                             kernel_size=[kt+2,kf+2],
                             padding="same",
                             activation=activation)

    conv2b = tf.layers.conv2d(inputs=conv2a,
                             filters=32,
                             kernel_size=[kt+2,kf+2],
                             padding="same",
                             activation=activation)

    conv2c = tf.layers.conv2d(inputs=conv2b,
                             filters=32,
                             kernel_size=[kt+2,kf+2],
                             padding="same",
                             activation=activation)

    conv2c = tf.contrib.layers.batch_norm(conv2c,scale=True)
    # Pool layer 2 (max pooling), 2x2 filter with stride of 2
    # in: [-1,30,512,32]
    # out: [-1,15,256,32]
    pool2 = tf.layers.max_pooling2d(inputs=conv2c,
                                    pool_size=[2,2],
                                    strides=2)

    # Conv. layer 3
    # in: [-1,15,256,32]
    # out: [-1,15,256,64]
    conv3a = tf.layers.conv2d(inputs=pool2,
                             filters=128,
                             kernel_size=[kt+4,kf+4],
                             padding="same",
                             activation=activation)

    conv3b = tf.layers.conv2d(inputs=conv3a,
                             filters=128,
                             kernel_size=[kt+4,kf+4],
                             padding="same",
                             activation=activation)

    conv3c = tf.layers.conv2d(inputs=conv3b,
                             filters=128,
                             kernel_size=[kt+4,kf+4],
                             padding="same",
                             activation=activation)

    conv3c = tf.contrib.layers.batch_norm(conv3c,scale=True)
    # Pool layer 3
    # in: [-1,15,256,64]
    # out: [-1,5,85,64] (???!!!)
    pool3 = tf.layers.max_pooling2d(inputs=conv3c,
                                    pool_size=[2,2],
                                    strides=3) #padding?

    print '##### pool3 shape: ',np.shape(pool3)
    # Conv. layer 4
    # in: [-1,5,85,64]  XXX
    # out: [-1,5,85,16]
    conv4a = tf.layers.conv2d(inputs=pool3,
                             filters=128*5,
                             kernel_size=[kt+6,kf+6],
                             padding="same",
                             activation=activation)

    conv4b = tf.layers.conv2d(inputs=conv4a,
                             filters=128*5,
                             kernel_size=[kt+6,kf+6],
                             padding="same",
                             activation=activation)

    conv4c = tf.layers.conv2d(inputs=conv4b,
                             filters=128*5,
                             kernel_size=[kt+6,kf+6],
                             padding="same",
                             activation=activation)


    conv4c = tf.contrib.layers.batch_norm(conv4c,scale=True)
#    avg_pool = tf.layers.average_pooling2d(inputs=conv4c,pool_size=[4,4],strides=2,padding='same')

    # Flatten
    # in: [-1,5,85,16] XXX
    # out: [-1,5*85*16=6800]
    flatten = tf.reshape(conv4c,[-1,55040])

    # Dense layer 1
    # in: [-1,6800]
    # out: [-1,2048]
    dense1 = tf.layers.dense(inputs=flatten, units=11008, activation=activation)
    dropout1 = tf.layers.dropout(inputs=dense1,rate=0.4,training=mode==tf.estimator.ModeKeys.TRAIN)
    dropout1 = tf.contrib.layers.batch_norm(dropout1,scale=True)
    #15,256,64 pool2?
    #dropout1 = tf.reshape(dropout1, [-1,15,256,32])
    #dropout1 = dropout1 + conv2a
#pool3 2, 43, 128
    dropout1 = tf.reshape(dropout1, [-1,2, 43, 128])
    dropout1 = tf.concat([dropout1,pool3],3)
    dropout1 = tf.reshape(dropout1, [-1,22016])

    # in: [-1,2048]
    # out: [-1,1024]
    dense2 = tf.layers.dense(inputs=dropout1, units=1024, activation=activation)
    dropout2 = tf.layers.dropout(inputs=dense2,rate=0.2,training=mode==tf.estimator.ModeKeys.TRAIN)
    dropout2 = tf.contrib.layers.batch_norm(dropout2,scale=True)
    # in: [-1,1024]
    # out: [-1,60*1024]
    output = tf.layers.dense(inputs=dropout2, units=512*30*2*8)
    output_reshape = tf.contrib.layers.batch_norm(tf.reshape(output, [-1,30,512,16]),scale=True)
    conv1a = tf.contrib.layers.batch_norm(conv1a)
	
    print 'output_reshape ',np.shape(output_reshape)
    print 'conv1a ',np.shape(conv1a)
    output_reshape = tf.concat([output_reshape,conv1a],3)
    print 'Output of concat: ',np.shape(output_reshape)
    final_conv = tf.layers.conv2d(inputs=output_reshape,
                             filters=8,
                             kernel_size=[kt,kf],
                             padding="same",
                             activation=activation)
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
   dim = 0
   data_a = np.log10(np.abs(np.copy(np.array(data))))
   data_b = np.log10(np.abs(np.copy(np.array(data))))
   batch,t_num,f_num = np.shape(data)
   # initialize output array
   data_out = np.zeros((batch,t_num,f_num,2))
   # there should be a more elegant way to do this using map
   # but there's an issue when passing data
   #if dim == 0:
   for b in range(batch):
       data_out[b,:,:,0] = timediff(data_a[b,:,:])
       data_out[b,:,:,1] = timediff(data_b[b,:,:])   
   return np.abs(data_out)

def main(args):
    tset_size = 1000
    trainlen = 800
    steps = 10
    # load data
    f = h5py.File('SimVisRFI.h5', 'r')
    # We want to add real data in between sim data w/ xrfi flags
    real_data = import_test_data('zen.2457555.40356.xx.HH.uvcT').reshape(1,-1,1024) 
    print 'real_data shape: ',np.shape(real_data.reshape(-1,1024))
    real_flags = xrfi.xrfi(np.abs(real_data.reshape(-1,1024))).reshape(1,-1,1024)
    all_data = preprocess(f['data'])
    print np.shape(all_data)
    train_data = np.asarray(all_data)[:trainlen,:,:,:]
    print np.shape(train_data)
    train_data = np.asarray(train_data, dtype=np.float32)
    #train_data[np.isinf(train_data)] = 0.
    train_labels = np.reshape(np.asarray(f['flag'])[:trainlen,:,:], (trainlen, 1024*60))
    train_labels = np.asarray(train_labels, dtype=np.int32)
    eval_data = np.asarray(all_data)[trainlen:,:,:,:]
#    eval_data = real_data
    eval_data = np.asarray(eval_data,dtype=np.float32)
    #eval_data[np.isinf(eval_data)] = 0.
    eval_labels = np.asarray(f['flag'],dtype=np.int32)[trainlen:,:,:]
    #eval_labels = real_flags
    eval_labels = np.reshape(eval_labels, (tset_size-trainlen, 1024*60))

#    real_data = import_test_data('zen.2457555.40356.xx.HH.uvcT').reshape(1,-1,1024)
    real_data_ = np.asarray(preprocess(real_data), dtype=np.float32)
    real_data = real_data.reshape(-1,1024)
    # create Estimator
    rfiCNN = tf.estimator.Estimator(model_fn=cnn,model_dir='./checkpoint/')

    train_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x":train_data},
        y=train_labels,
        batch_size=10,
        num_epochs=100,
        shuffle=True,
    )

    rfiCNN.train(input_fn=train_input_fn, steps=steps)

    eval_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x":eval_data},
        y=eval_labels,
        num_epochs=3,
        shuffle=False)

    #real_data_ = real_data_.reshape(-1,1024,2)

    test_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x":real_data_},
        shuffle=False
    )
#    eval_results = rfiCNN.evaluate(input_fn=eval_input_fn)
#    print(eval_results)

    rfiPredict = rfiCNN.predict(input_fn=test_input_fn)
    for i,predicts in enumerate(rfiPredict):
        print np.shape(i),np.shape(predicts['probabilities'])
        idxs = np.where(train_labels[1,:] == 1)
        print predicts['probabilities'][idxs]

        plt.subplot(311)
        plt.imshow(predicts['classes'].reshape(-1,1024),aspect='auto')
        plt.colorbar()
        plt.subplot(312)
	plt.imshow(np.log10((np.abs(real_data).reshape(-1,1024))),aspect='auto')
        #plt.imshow(train_labels[1,:].reshape(-1,1024) - predicts['classes'].reshape(-1,1024),aspect='auto')
        plt.colorbar()
	plt.subplot(313)
	plt.imshow(np.log10((np.abs(real_data*np.logical_not(predicts['classes'].reshape(-1,1024))))),aspect='auto')
	plt.colorbar()
        plt.savefig('RealData.png')
#        pl.show()

if __name__ == "__main__":
    tf.app.run()
