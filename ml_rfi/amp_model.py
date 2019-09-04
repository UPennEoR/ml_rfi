# -*- coding: utf-8 -*-
# Copyright (c) 2019 The HERA Team
# Licensed under the 2-clause BSD License

from __future__ import division, print_function, absolute_import

import tensorflow as tf

from . import helper_functions as hf


def AmpFCN(x, reuse=None, mode_bn=True, d_out=0.0):
    """
    Define an amplitude-only fully convolutional network (FCN).

    This function defines an amplitude-only Fully Convolutional Neural Network
    Model. Uses 6 convolutional layers in the downsampling stage, into 2 fully
    connected layers, and upsampled through 4 transpose convolutional layers.

    Parameters
    ----------
    x : ndarray
        Four-dimensional array of size (BatchSize * FoldFactor, Time,
        Frequency, Channels).
    reuse :
        Option that is passed to a TensorFlow variable_scope option
        defining option reuse. Should be one of: True, False,
        tf.AUTO_REUSE, None. Default is None (warn user when variables
        are potentially reused unintentionally).
    mode_bn : bool
        If True, add a batch-normalization layer to the network following
        convolutional layers. Default is True (add batch normalization layers).
    d_out : float
        Dropout fraction to use for dropout layers. Default is 0.0 (no dropout).

    Returns
    -------
    final_conv : ndarray
        Four-dimensional boolean array of same size as x, which contains
        True where RFI has been detected and False elsewhere.
    """

    with tf.variable_scope("FCN", reuse=reuse):
        kt = 3
        pad_size = 68
        sh = x.get_shape().as_list()
        s = 1
        activation = tf.nn.leaky_relu
        input_layer = tf.cast(
            tf.reshape(x, [-1, sh[1], sh[2], sh[3]]), dtype=tf.float32
        )
        tf.summary.image(
            "IP_Amp", tf.reshape(input_layer[0, :, :, 0], [1, pad_size, pad_size, 1])
        )

        slayer1 = hf.stacked_layer(
            input_layer[:, :, :, :1],
            s * 8,
            kt,
            kt,
            activation,
            [2, 2],
            [2, 2],
            bnorm=True,
            mode=mode_bn,
        )
        s1sh = slayer1.get_shape().as_list()
        tf.summary.image(
            "S1",
            tf.reshape(
                tf.reduce_max(slayer1[0, :, :, :], axis=-1),
                [1, int(pad_size / 2), int(pad_size / 2), 1],
            ),
        )
        slayer2 = hf.stacked_layer(
            slayer1,
            s * 16,
            kt,
            kt,
            activation,
            [2, 2],
            [2, 2],
            bnorm=True,
            mode=mode_bn,
        )
        slayer3 = hf.stacked_layer(
            slayer2,
            s * 32,
            kt,
            kt,
            activation,
            [2, 2],
            [2, 2],
            bnorm=True,
            mode=mode_bn,
            dropout=0.0,
        )
        s3sh = slayer3.get_shape().as_list()
        slayer4 = tf.layers.dropout(
            hf.stacked_layer(
                slayer3,
                s * 64,
                kt,
                kt,
                activation,
                [2, 2],
                [2, 2],
                bnorm=True,
                mode=mode_bn,
            ),
            rate=0.0,
        )
        slayer5 = hf.stacked_layer(
            slayer4,
            s * 128,
            kt,
            kt,
            activation,
            [2, 2],
            [2, 2],
            bnorm=True,
            mode=mode_bn,
            dropout=0.0,
        )
        slayer6 = hf.stacked_layer(
            slayer5,
            s * 256,
            kt,
            kt,
            activation,
            [2, 2],
            [2, 2],
            bnorm=True,
            mode=mode_bn,
        )
        s6sh = slayer6.get_shape().as_list()

        # Fully connected-convolutional layers
        slayer7 = hf.stacked_layer(
            slayer6,
            s * 512,
            1,
            1,
            activation,
            [1, 1],
            [1, 1],
            bnorm=True,
            dropout=d_out,
            mode=mode_bn,
        )
        slayer8 = hf.stacked_layer(
            slayer7,
            s * 1024,
            1,
            1,
            activation,
            [1, 1],
            [1, 1],
            bnorm=True,
            dropout=d_out,
            mode=mode_bn,
        )

        # Transpose convolutional layers (upsampling)
        upsamp1 = tf.layers.conv2d_transpose(
            slayer8,
            filters=s * 256,
            kernel_size=[s6sh[1], s6sh[1]],
            activation=activation,
        )
        upsamp1 = tf.layers.dropout(
            tf.layers.batch_normalization(
                upsamp1, scale=True, center=True, training=mode_bn, fused=True
            ),
            rate=0.7,
        )
        upsamp1 = tf.concat(
            [
                tf.layers.dropout(hf.tfnormalize(upsamp1), rate=0.7),
                tf.layers.dropout(hf.tfnormalize(slayer6), rate=0.7),
            ],
            axis=-1,
        )
        upsamp2 = tf.layers.conv2d_transpose(
            upsamp1,
            filters=s * 128,
            kernel_size=[s6sh[1] + 1, s6sh[1] + 1],
            activation=activation,
        )
        upsamp2 = tf.layers.dropout(
            tf.layers.batch_normalization(
                upsamp2, scale=True, center=True, training=mode_bn, fused=True
            ),
            rate=0.7,
        )
        upsamp2 = tf.concat(
            [
                tf.layers.dropout(hf.tfnormalize(upsamp2), rate=0.7),
                tf.layers.dropout(hf.tfnormalize(slayer5), rate=0.7),
            ],
            axis=-1,
        )
        upsamp3 = tf.layers.conv2d_transpose(
            upsamp2,
            filters=s * 32,
            kernel_size=[s3sh[1] - 2 * s6sh[1] + 1, s3sh[1] - 2 * s6sh[1] + 1],
            activation=activation,
        )
        upsh3 = upsamp3.get_shape().as_list()
        upsamp3 = tf.layers.dropout(
            tf.layers.batch_normalization(
                upsamp3, scale=True, center=True, training=mode_bn, fused=True
            ),
            rate=0.0,
        )
        upsamp3 = tf.concat(
            [
                tf.layers.dropout(hf.tfnormalize(upsamp3), rate=0.7),
                tf.layers.dropout(hf.tfnormalize(slayer3), rate=0.7),
            ],
            axis=-1,
        )
        out_filter = 2
        upsamp4 = tf.layers.conv2d_transpose(
            upsamp3,
            filters=out_filter,
            kernel_size=[int(sh[1] - upsh3[1]) + 1, int(sh[1] - upsh3[1]) + 1],
            activation=activation,
        )
        upsamp4 = tf.layers.dropout(
            tf.layers.batch_normalization(
                upsamp4, scale=True, center=True, training=mode_bn, fused=True
            ),
            rate=0.7,
        )
        tf.summary.image(
            "Flag Guess", tf.reshape(upsamp4[0, :, :, 1], [1, pad_size, pad_size, 1])
        )
        final_conv = tf.reshape(upsamp4, [-1, sh[1] * sh[1], 2])
        tf.summary.image(
            "ArgMax",
            tf.cast(
                tf.reshape(
                    tf.argmax(upsamp4[0, :, :, :], axis=-1), [1, pad_size, pad_size, 1]
                ),
                dtype=tf.float32,
            ),
        )

    return final_conv
