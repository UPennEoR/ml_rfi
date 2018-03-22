import numpy as np
import tensorflow as tf
import h5py
import pylab as pl
tf.logging.set_verbosity(tf.logging.INFO)
from pyuvdata import UVData

def import_test_data(filename,bl_tup=(9,89),rescale=1.0):
    uvd = UVData()
    uvd.read_miriad(filename)
    a1,a2 = bl_tup
    data = uvd.get_data(a1,a2)
    data*=rescale
    return data

def cnn(features,labels,mode):
    """
    Model for CNN

    features: visibility array
    labels: RFI flag array
    mode: used by tensorflow to distinguish training and testing
    """ 
    activation = tf.nn.relu
    # kernel size
    ks1 = 7 # 1
    ks2 = 5 # 2
    ks3 = 3 # 3
    ks4 = 1 # 4
    # 4D tensor: batch size, height (ntimes), width (nfreq), channels (1)
    input_layer = tf.reshape(features["x"],[-1,60,1024,1])

    # Conv. layer 1
    # in: [-1,60,1024,1]
    # out: [-1,60,1024,16]
    conv1a = tf.layers.conv2d(inputs=input_layer,
                             filters=16,
                             kernel_size=[ks1,ks1],
                             padding="same",
                             activation=activation)

    conv1b = tf.layers.conv2d(inputs=conv1a,
                             filters=16,
                             kernel_size=[ks1,ks1],
                             padding="same",
                             activation=activation)

    conv1c = tf.layers.conv2d(inputs=conv1b,
                             filters=16,
                             kernel_size=[ks1,ks1],
                             padding="same",
                             activation=activation)

    # Pool layer 1 (max pooling), 2x2 filter with stride of 2
    # in: [-1,60,1024,16]
    # out: [-1,30,512,16]
    pool1 = tf.layers.max_pooling2d(inputs=conv1c,
                                    pool_size=[2,2],
                                    strides=2)

    # Conv. layer 2
    # in: [-1,30,512,16]
    # out: [-1,30,512,32]

    conv2a = tf.layers.conv2d(inputs=pool1,
                             filters=32,
                             kernel_size=[ks2,ks2],
                             padding="same",
                             activation=activation)

    conv2b = tf.layers.conv2d(inputs=conv2a,
                             filters=32,
                             kernel_size=[ks2,ks2],
                             padding="same",
                             activation=activation)

    conv2c = tf.layers.conv2d(inputs=conv2b,
                             filters=32,
                             kernel_size=[ks2,ks2],
                             padding="same",
                             activation=activation)
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
                             filters=64,
                             kernel_size=[ks3,ks3],
                             padding="same",
                             activation=activation)

    conv3b = tf.layers.conv2d(inputs=conv3a,
                             filters=64,
                             kernel_size=[ks3,ks3],
                             padding="same",
                             activation=activation)

    conv3c = tf.layers.conv2d(inputs=conv3b,
                             filters=64,
                             kernel_size=[ks3,ks3],
                             padding="same",
                             activation=activation)
    # Pool layer 3
    # in: [-1,15,256,64]
    # out: [-1,5,85,64] (???!!!)
    pool3 = tf.layers.max_pooling2d(inputs=conv3c,
                                    pool_size=[2,2],
                                    strides=3) #padding?

    # Conv. layer 4
    # in: [-1,5,85,64]  XXX
    # out: [-1,5,85,16]
    conv4a = tf.layers.conv2d(inputs=pool3,
                             filters=16,
                             kernel_size=[ks4,ks4],
                             padding="same",
                             activation=activation)

    conv4b = tf.layers.conv2d(inputs=conv4a,
                             filters=16,
                             kernel_size=[ks4,ks4],
                             padding="same",
                             activation=activation)

    conv4c = tf.layers.conv2d(inputs=conv4b,
                             filters=16,
                             kernel_size=[ks4,ks4],
                             padding="same",
                             activation=activation)
    # Flatten
    # in: [-1,5,85,16] XXX
    # out: [-1,5*85*16=6800]
    flatten = tf.reshape(conv4c,[-1,6800])

    # Dense layer 1
    # in: [-1,6800]
    # out: [-1,2048]
    dense1 = tf.layers.dense(inputs=flatten, units=2048, activation=activation)
    dropout1 = tf.layers.dropout(inputs=dense1,rate=0.5,training=mode==tf.estimator.ModeKeys.TRAIN)

    # in: [-1,2048]
    # out: [-1,1024]
    dense2 = tf.layers.dense(inputs=dropout1, units=1024, activation=activation)
    dropout2 = tf.layers.dropout(inputs=dense2,rate=0.5,training=mode==tf.estimator.ModeKeys.TRAIN)

    # in: [-1,1024]
    # out: [-1,60*1024]
    output = tf.layers.dense(inputs=dropout2, units=1024*60*2)
    output_reshape = tf.reshape(output, [-1,1024*60,2])

    predictions = {
        "classes": tf.argmax(input=output_reshape, axis=2),
        "probabilities": tf.nn.softmax(output_reshape,name="softmax_tensor")
    }

    try:    
        loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=output_reshape)
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

def normscale(data):
    mu = np.mean(data)
    data -= mu
    data /= data.max()
    return data

def main(args):
    trainlen=666
    # load data
    f = h5py.File('SimVisRFI_speckle.h5', 'r')
    train_data = np.asarray(f['data'])[:trainlen,:,:]
    train_data = np.asarray(normscale(np.abs(train_data)), dtype=np.float32)
    train_data[np.isinf(train_data)] = 0.
    train_labels = np.reshape(np.asarray(f['flag'])[:trainlen,:,:], (trainlen, 1024*60))
    train_labels = np.asarray(train_labels, dtype=np.int32)
    eval_data = np.asarray(f['data'])[trainlen:,:,:]
    eval_data = np.asarray(normscale(np.abs(eval_data)),dtype=np.float32)
    eval_data[np.isinf(eval_data)] = 0.
    eval_labels = np.asarray(f['flag'],dtype=np.int32)[trainlen:,:,:]
    eval_labels = np.reshape(eval_labels, (1000-trainlen, 1024*60))
    
    real_data = import_test_data('zen.2457555.40356.xx.HH.uvcT')
    real_data = np.asarray(normscale(np.abs(real_data)), dtype=np.float32)
    
    # create Estimator
    rfiCNN = tf.estimator.Estimator(model_fn=cnn,model_dir='./checkpoint/')

    train_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x":train_data},
        y=train_labels,
        batch_size=2,
        num_epochs=3,
        shuffle=True
    )

    rfiCNN.train(input_fn=train_input_fn, steps=300)

    eval_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x":eval_data},
        y=eval_labels,
        num_epochs=3,
        shuffle=False)

    test_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x":real_data},
        shuffle=False
    )
    eval_results = rfiCNN.evaluate(input_fn=eval_input_fn)
    print(eval_results)

    rfiPredict = rfiCNN.predict(input_fn=test_input_fn)
    for i,predicts in enumerate(rfiPredict):
        print np.shape(i),np.shape(predicts['probabilities'])
        idxs = np.where(train_labels[1,:] == 1)
        print predicts['probabilities'][idxs]

        pl.subplot(211)
        pl.imshow(predicts['classes'].reshape(-1,1024),aspect='auto')
        pl.colorbar()
        pl.subplot(212)
        pl.imshow(train_labels[1,:].reshape(-1,1024) - predicts['classes'].reshape(-1,1024),aspect='auto')
        pl.colorbar()
        pl.savefig('Speckles.png')
#        pl.show()

if __name__ == "__main__":
    tf.app.run()
