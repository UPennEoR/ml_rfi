import numpy as np
import tensorflow as tf
import h5py
tf.logging.set_verbosity(tf.logging.INFO)

def cnn(features,labels,mode):
    """
    Model for CNN

    features: visibility array
    labels: RFI flag array
    mode: used by tensorflow to distinguish training and testing
    """
    # 4D tensor: batch size, height (ntimes), width (nfreq), channels (1)
    input_layer = tf.reshape(features["x"],[-1,60,1024,1])

    # Conv. layer 1
    # in: [-1,60,1024,1]
    # out: [-1,60,1024,16]
    conv1 = tf.layers.conv2d(inputs=input_layer,
                             filters=16,
                             kernel_size=[5,5],
                             padding="same",
                             activation=tf.nn.relu)

    # Pool layer 1 (max pooling), 2x2 filter with stride of 2
    # in: [-1,60,1024,16]
    # out: [-1,30,512,16]
    pool1 = tf.layers.max_pooling2d(inputs=conv1,
                                    pool_size=[2,2],
                                    strides=2)

    # Conv. layer 2
    # in: [-1,30,512,16]
    # out: [-1,30,512,32]
    conv2 = tf.layers.conv2d(inputs=pool1,
                             filters=32,
                             kernel_size=[5,5],
                             padding="same",
                             activation=tf.nn.relu)

    # Pool layer 2 (max pooling), 2x2 filter with stride of 2
    # in: [-1,30,512,32]
    # out: [-1,15,256,32]
    pool2 = tf.layers.max_pooling2d(inputs=conv2,
                                    pool_size=[2,2],
                                    strides=2)

    # Conv. layer 3
    # in: [-1,15,256,32]
    # out: [-1,15,256,64]
    conv3 = tf.layers.conv2d(inputs=pool2,
                             filters=64,
                             kernel_size=[5,5],
                             padding="same",
                             activation=tf.nn.relu)

    # Pool layer 3
    # in: [-1,15,256,64]
    # out: [-1,5,85,64] (???!!!)
    pool3 = tf.layers.max_pooling2d(inputs=conv3,
                                    pool_size=[2,2],
                                    strides=3) #padding?

    # Conv. layer 4
    # in: [-1,5,85,64]  XXX
    # out: [-1,5,85,16]
    conv4 = tf.layers.conv2d(inputs=pool3,
                             filters=16,
                             kernel_size=[5,5],
                             padding="same",
                             activation=tf.nn.relu)

    # Flatten
    # in: [-1,5,85,16] XXX
    # out: [-1,5*85*16=6800]
    flatten = tf.reshape(conv4,[-1,6800])

    # Dense layer 1
    # in: [-1,6800]
    # out: [-1,2048]
    dense1 = tf.layers.dense(inputs=flatten, units=2048, activation=tf.nn.relu)
    dropout1 = tf.layers.dropout(inputs=dense1,rate=0.5,training=mode==tf.estimator.ModeKeys.TRAIN)

    # in: [-1,2048]
    # out: [-1,1024]
    dense2 = tf.layers.dense(inputs=dropout1, units=1024, activation=tf.nn.relu)
    dropout2 = tf.layers.dropout(inputs=dense2,rate=0.5,training=mode==tf.estimator.ModeKeys.TRAIN)

    # in: [-1,1024]
    # out: [-1,60*1024]
    output = tf.layers.dense(inputs=dropout2, units=1024*60*2)
    output_reshape = tf.reshape(output, [-1,1024*60,2])
    probabilities = tf.nn.softmax(output_reshape,name="softmax_tensor")

    #if mode == tf.estimator.ModeKeys.PREDICT:
    #    return tf.estimator.EstimatorSpec(mode=mode,predictions=)
    loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=output_reshape)

    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=1e-4)
        train_op = optimizer.minimize(loss=loss,global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode=mode,loss=loss,train_op=train_op)

def main(args):
    # load data
    f = h5py.File('SimVisRFI.h5', 'r')
    train_data = np.abs(np.asarray(f['data']))
    train_data = np.asarray(train_data, dtype=np.float32)
    train_labels = np.reshape(np.asarray(f['flag']), (1000, 1024*60))
    train_labels = np.asarray(train_labels, dtype=np.int32)
    print(train_labels.dtype)
    #import IPython;IPython.embed()
    # test_data = #np.array
    # separate into train and test

    # create Estimator
    rfiCNN = tf.estimator.Estimator(model_fn=cnn,model_dir='/tmp/blah')

    train_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x":train_data},
        y=train_labels,
        batch_size=10,
        num_epochs=None,
        shuffle=True
    )
    rfiCNN.train(input_fn=train_input_fn, steps=10)

if __name__ == "__main__":
    tf.app.run()
