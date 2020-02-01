# -*- coding: utf-8 -*-
# Copyright 2019 The HERA Collaboration
# Licensed under the 2-clause BSD License

import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Activation, BatchNormalization, concatenate, Conv2D
from tensorflow.keras.layers import Conv2DTranspose, Dropout, Input, LeakyReLU
from tensorflow.keras.layers import MaxPooling2D, Reshape


def stacked_layer(
    input_layer,
    nfilters,
    kernel_size=3,
    batch_normalize=True,
    dropout_rate=0.7,
    alpha=0.2,
    pool_size=(2, 2),
    pool_stride=(2, 2),
):
    """Make a "stacked layer" for adding to the model.

    A "stacked layer" is a series of 3 two-dimensional convolutional layers,
    followed by a max pooling layer. There is an optional dropout layer
    connecting convolutional layers 2 and 3, as well as an optional batch
    normalization immediately prior to the max pooling layer.

    Parameters
    ----------
    input_layer : Keras layer
        The Keras input layer that the stacked layer to should be applied to.
    nfilters : int
        The number of convolutional filters that should be included in the
        stacked layer.
    kernel_size : int or tuple of ints
        The size of the convolutional kernel that should be used in the
        convolutional layers. If a single int, the same size will be used for
        all dimensions.
    batch_normalize : bool
        Whether to include a batch normalization layer before the max pooling
        layer.
    dropout_rate : float
        The dropout rate to include in between convolutional layers 2 and 3.
        No dropout layer is applied if set to 0.
    alpha : float
        The alpha parameter to use for a LeakyReLU layer. Set to 0 for a
        regular (non-leaky) ReLU.
    pool_size : tuple of ints
        The size of pool to use for the max pooling layer.
    pool_stride : tuple of ints
        The stride of the pool to use in the max pooling layer.

    Returns
    -------
    out_layer : Keras layer
        The output Keras layer following the stacked layer.
    """
    layer = Conv2D(nfilters, kernel_size=kernel_size, padding="same")(input_layer)
    if alpha > 0.0:
        layer1 = LeakyReLU(alpha=alpha)(layer)
    else:
        layer1 = Activation("relu")(layer)
    layer2 = Conv2D(nfilters, kernel_size=kernel_size, padding="same")(layer1)
    if alpha > 0.0:
        layer3 = LeakyReLU(alpha=alpha)(layer2)
    else:
        layer3 = Activation("relu")(layer2)
    if dropout_rate > 0.0:
        layer4 = Dropout(dropout_rate)(layer3)
    else:
        layer4 = layer3
    layer5 = Conv2D(nfilters, kernel_size=1, padding="same")(layer4)
    if alpha > 0.0:
        layer6 = LeakyReLU(alpha=alpha)(layer5)
    else:
        layer6 = Activation("relu")(layer5)
    if batch_normalize:
        layer7 = BatchNormalization()(layer6)
    else:
        layer7 = layer6
    out_layer = MaxPooling2D(pool_size=pool_size, strides=pool_stride)(layer7)

    # cleanup
    del layer, layer1, layer2, layer3, layer4, layer5, layer6, layer7

    return out_layer


def upsample_layer(
    input_layer,
    nfilters,
    kernel_size=3,
    conv_stride=(1, 1),
    batch_normalize=True,
    dropout_rate=0.7,
    alpha=0.2,
):
    """Make an upsampling layer.

    This function will add a convolutional transpose layer (sometimes called
    a "deconvolutional layer") to the network. It "undoes" the downsampling that
    happens as part of a convolutional layer. The full upsampling layer also
    includes a LeakyReLU activation layer, and optionally a batch normalization
    layer and dropout layer.

    Parameters
    ----------
    input_layer : Keras layer
        The Keras layer to apply the covolutional transpose layer to.
    nfilters : int
        The number of convolutional filters to apply as part of the transpose
        layer.
    kernel_size : int or tuple of ints
        The size of the convolutional kernel that should be used in the
        convolutional layers. If a single int, the same size will be used for
        all dimensions.
    conv_stride : tuple of ints
        The stride to use in the transpose layer.
    batch_normalize : bool
        Whether to include a batch normalization layer after the LeakyReLU layer.
    dropout_rate : float
        The dropout rate to use for the dropout layer. No dropout layer is applied
        if the dropout rate is set to 0.
    alpha : float
        The alpha value to use for a LeakyReLU layer. Set to 0 for a regular
        (non-leaky) ReLU.

    Returns
    -------
    layer : Keras layer
        The output layer following the upsampling.
    """
    layer1 = Conv2DTranspose(
        nfilters, kernel_size=kernel_size, strides=conv_stride, padding="same"
    )(input_layer)
    if alpha > 0.0:
        layer2 = LeakyReLU(alpha=alpha)(layer1)
    else:
        layer2 = Activation("relu")(layer1)
    if batch_normalize:
        layer3 = BatchNormalization()(layer2)
    else:
        layer3 = layer2
    if dropout_rate > 0.0:
        out_layer = Dropout(dropout_rate)(layer3)
    else:
        out_layer = layer3

    # cleanup
    del layer1, layer2, layer3

    return out_layer


def normalize_and_concatenate(layer1, layer2, layer3, dropout_rate=0.7):
    """Normalize layers and concatenate.

    Parameters
    ----------
    layer1 : Keras layer
        The first layer to be normalized and concatenated.
    layer2 : Keras layer
        The second layer to be normalizaed and concatenated.
    layer3 : Keras layer
        The third layer to be normalized and concatenated.
    dropout_rate : float
        The dropout rate to be applied prior to concatenation. No dropout layers
        are applied if the dropout rate is set to 0.

    Returns
    -------
    Keras layer
        A Keras layer with all of the input layers concatenated together.
    """
    layer1 = BatchNormalization(trainable=False)(layer1)
    layer2 = BatchNormalization(trainable=False)(layer2)
    layer3 = BatchNormalization(trainable=False)(layer3)
    if dropout_rate > 0.0:
        layer1 = Dropout(dropout_rate)(layer1)
        layer2 = Dropout(dropout_rate)(layer2)
        layer3 = Dropout(dropout_rate)(layer3)

    return concatenate([layer1, layer2, layer3])


def amp_phs_model(
    input_shape,
    kernel_size=3,
    filter_factor=2,
    batch_normalize=True,
    dropout_rate=0.0,
    alpha=0.2,
    pool_size=(2, 2),
    pool_stride=(2, 2),
):
    """Make a Keras model for using amplitude and phase to predict RFI.

    This funciton will make an "amplitude and phase model" for predicting RFI
    instances in a visibility dataset as in Kerrigan et al. (2019). Use the
    default parameters to get one similar to that used in the primary analysis
    of the paper. The output layer should have the same shape as the input
    shape. Note that a softmax activation layer is applied at the end, which
    should categorize the input pixel as 0 (no RFI) or 1 (yes RFI).

    Parameters
    ----------
    input_shape : tuple of ints
        The shape of the input dataset. Should be a 3-dimensional tuple
        representing (Ntimes, Nfreqs, Nfeatures). For the model as designed
        below, there are separate amplitude and phase inputs, so Nfeatures
        should be 1.
    kernel_size : int
        The size of convolutional kernel to use for convolutional layers in
        the network.
    filter_factor : int
        The "filter factor" to use as the stacked layers progress. The filter
        factor will be be applied on top of increasing powers of two.
    batch_normalize : bool
        Whether to apply batch normalization layers througout the network.
    dropout_rate : float
        The dropout rate to use in various layers. A value of 0 will mean that
        no dropout layers are applied.
    alpha : float
        The alpha value to use in LeakyReLU layers. A value of 0 will result
        in a regular (non-leaky) ReLU.
    pool_size : tuple of ints
        The size of pool to use in max pooling layers. Note that the minimum
        size of the input dataset is determined by the size and stride of the
        max pooling layers.
    pool_stride : tuple of ints
        The stride of pool to use in max pooling layers. Note that the minimum
        size of the input dataset is determined by the size and stride of the
        max pooling layers.

    Returns
    -------
    model : Keras model
        A Keras model with the model architecture prescribed.
    """
    # define inputs -- one for amplitude, one for phase
    amp_input = Input(shape=input_shape, name="amp_input")
    phs_input = Input(shape=input_shape, name="phs_input")

    # add stacked layers for amplitude branch
    nfilters = 8 * filter_factor
    s1a = stacked_layer(
        amp_input,
        nfilters,
        kernel_size=kernel_size,
        batch_normalize=batch_normalize,
        dropout_rate=dropout_rate,
        alpha=alpha,
        pool_size=pool_size,
        pool_stride=pool_stride,
    )
    nfilters = 16 * filter_factor
    s2a = stacked_layer(
        s1a,
        nfilters,
        kernel_size=kernel_size,
        batch_normalize=batch_normalize,
        dropout_rate=dropout_rate,
        alpha=alpha,
        pool_size=pool_size,
        pool_stride=pool_stride,
    )
    nfilters = 32 * filter_factor
    s3a = stacked_layer(
        s2a,
        nfilters,
        kernel_size=kernel_size,
        batch_normalize=batch_normalize,
        dropout_rate=dropout_rate,
        alpha=alpha,
        pool_size=pool_size,
        pool_stride=pool_stride,
    )
    nfilters = 64 * filter_factor
    s4a = stacked_layer(
        s3a,
        nfilters,
        kernel_size=kernel_size,
        batch_normalize=batch_normalize,
        dropout_rate=dropout_rate,
        alpha=alpha,
        pool_size=pool_size,
        pool_stride=pool_stride,
    )
    nfilters = 128 * filter_factor
    s5a = stacked_layer(
        s4a,
        nfilters,
        kernel_size=kernel_size,
        batch_normalize=batch_normalize,
        dropout_rate=dropout_rate,
        alpha=alpha,
        pool_size=pool_size,
        pool_stride=pool_stride,
    )
    nfilters = 256 * filter_factor
    s6a = stacked_layer(
        s5a,
        nfilters,
        kernel_size=kernel_size,
        batch_normalize=batch_normalize,
        dropout_rate=dropout_rate,
        alpha=alpha,
        pool_size=pool_size,
        pool_stride=pool_stride,
    )

    # add the same for phase
    nfilters = 8 * filter_factor
    s1p = stacked_layer(
        phs_input,
        nfilters,
        kernel_size=kernel_size,
        batch_normalize=batch_normalize,
        dropout_rate=dropout_rate,
        alpha=alpha,
        pool_size=pool_size,
        pool_stride=pool_stride,
    )
    nfilters = 16 * filter_factor
    s2p = stacked_layer(
        s1p,
        nfilters,
        kernel_size=kernel_size,
        batch_normalize=batch_normalize,
        dropout_rate=dropout_rate,
        alpha=alpha,
        pool_size=pool_size,
        pool_stride=pool_stride,
    )
    nfilters = 32 * filter_factor
    s3p = stacked_layer(
        s2p,
        nfilters,
        kernel_size=kernel_size,
        batch_normalize=batch_normalize,
        dropout_rate=dropout_rate,
        alpha=alpha,
        pool_size=pool_size,
        pool_stride=pool_stride,
    )
    nfilters = 64 * filter_factor
    s4p = stacked_layer(
        s3p,
        nfilters,
        kernel_size=kernel_size,
        batch_normalize=batch_normalize,
        dropout_rate=dropout_rate,
        alpha=alpha,
        pool_size=pool_size,
        pool_stride=pool_stride,
    )
    nfilters = 128 * filter_factor
    s5p = stacked_layer(
        s4p,
        nfilters,
        kernel_size=kernel_size,
        batch_normalize=batch_normalize,
        dropout_rate=dropout_rate,
        alpha=alpha,
        pool_size=pool_size,
        pool_stride=pool_stride,
    )
    nfilters = 256 * filter_factor
    s6p = stacked_layer(
        s5p,
        nfilters,
        kernel_size=kernel_size,
        batch_normalize=batch_normalize,
        dropout_rate=dropout_rate,
        alpha=alpha,
        pool_size=pool_size,
        pool_stride=pool_stride,
    )

    # add batch normalization to stacked layers
    norm_amp = BatchNormalization(trainable=False)(s6a)
    norm_phs = BatchNormalization(trainable=False)(s6p)

    if dropout_rate > 0.0:
        # add dropout layers after the final stacked layers
        norm_amp = Dropout(dropout_rate)(norm_amp)
        norm_phs = Dropout(dropout_rate)(norm_phs)

    # concatenate the amplitude and phase branches together
    norm_tot = concatenate([norm_amp, norm_phs])

    # start upsampling to get back to the original size
    in_shape = norm_tot.shape
    nfilters = 128 * filter_factor
    target_shape = s5a.shape
    cx = int(np.ceil(int(target_shape[1]) / int(in_shape[1])))
    cy = int(np.ceil(int(target_shape[2]) / int(in_shape[2])))
    conv_stride = (cx, cy)
    target_size = int(target_shape[1]) * int(target_shape[2]) * int(target_shape[3]) / 2
    input_size = cx * int(in_shape[1]) * cy * int(in_shape[2])
    upsample1 = upsample_layer(
        norm_tot,
        nfilters,
        kernel_size=kernel_size,
        conv_stride=conv_stride,
        batch_normalize=batch_normalize,
        dropout_rate=dropout_rate,
        alpha=alpha,
    )
    target_shape = (int(target_shape[1]), int(target_shape[2]), int(target_shape[3]))
    upsample1 = normalize_and_concatenate(upsample1, s5a, s5p)

    # upsample again
    nfilters = 32 * filter_factor
    in_shape = upsample1.shape
    target_shape = s3a.shape
    cx = int(np.ceil(int(target_shape[1]) / int(in_shape[1])))
    cy = int(np.ceil(int(target_shape[2]) / int(in_shape[2])))
    conv_stride = (cx, cy)
    upsample2 = upsample_layer(
        upsample1,
        nfilters,
        kernel_size=kernel_size,
        conv_stride=conv_stride,
        batch_normalize=batch_normalize,
        dropout_rate=dropout_rate,
        alpha=alpha,
    )
    upsample2 = normalize_and_concatenate(upsample2, s3a, s3p)

    # upsample again
    nfilters = 8 * filter_factor
    in_shape = upsample2.shape
    target_shape = s1a.shape
    cx = int(np.ceil(int(target_shape[1]) / int(in_shape[1])))
    cy = int(np.ceil(int(target_shape[2]) / int(in_shape[2])))
    conv_stride = (cx, cy)
    upsample3 = upsample_layer(
        upsample2,
        nfilters,
        kernel_size=kernel_size,
        conv_stride=conv_stride,
        batch_normalize=False,
        dropout_rate=0.0,
        alpha=alpha,
    )
    upsample3 = normalize_and_concatenate(upsample3, s1a, s1p)

    # final upsample to meet input layers
    nfilters = 1
    conv_stride = (2, 2)
    upsample4 = upsample_layer(
        upsample3,
        nfilters,
        kernel_size=kernel_size,
        conv_stride=conv_stride,
        batch_normalize=False,
        dropout_rate=0.0,
        alpha=alpha,
    )
    upsample4 = normalize_and_concatenate(upsample4, amp_input, phs_input)

    # make output layer
    # we have two classes, to we need 2 Conv2D filters
    upsample4 = Conv2D(2, kernel_size=1, padding="same")(upsample4)
    if alpha > 0.0:
        upsample4 = LeakyReLU(alpha=alpha)(upsample4)
    else:
        upsample4 = Activation("relu")(upsample4)

    # reshape to the right size
    out_dims = (input_shape[0], input_shape[1], 2)
    output = Reshape(out_dims)(upsample4)

    # add a softmax output
    output = Activation("softmax")(output)

    # check that the output layer is the right size for the input
    output_shape = output.shape[1:3]
    input_shape = input_shape[:2]
    if output_shape != input_shape:
        raise ValueError(
            "The output shape is different from the input shape; "
            "output has shape {} and input has shape {}. Please "
            "check the size and stride of max pooling layers and "
            "try again.".format(str(output_shape), str(input_shape))
        )

    # Define the model
    model = Model(inputs=[amp_input, phs_input], outputs=[output])

    return model
