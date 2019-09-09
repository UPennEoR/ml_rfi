# -*- coding: utf-8 -*-
# Copyright (c) 2019 The HERA Team
# Licensed under the 2-clause BSD License

from __future__ import print_function, division, absolute_import

from time import time
import random
from copy import copy

import numpy as np
import h5py
from sklearn.metrics import confusion_matrix
from scipy import ndimage
import tensorflow as tf
import keras
from keras import backend as K
from keras.models import load_model

from . import keras_model

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
    absX = np.where(absX <= 0.0, (1e-8) * np.random.randn(sh[0], sh[1]), absX)
    LOGabsX = np.nan_to_num(np.log10(absX))
    return np.nan_to_num((LOGabsX - np.nanmean(LOGabsX)) / np.nanstd(np.abs(LOGabsX)))


def normphs(X):
    """
    Normalization for the phase in the folding proces.
    """
    sh = np.shape(X)
    return np.array(np.sin(np.angle(X)))


def tfnormalize(X):
    """
    Skip connection layer normalization.
    """
    sh = np.shape(X)
    X_norm = tf.contrib.layers.layer_norm(X, trainable=False)
    return X


def foldl(data, ch_fold=16, padding=2):
    """
    Folding function for carving up a waterfall visibility flags for prediction in the FCN.
    """
    sh = np.shape(data)
    _data = data.T.reshape(ch_fold, sh[1] / ch_fold, -1)
    _DATA = np.array(map(transpose, _data))
    _DATApad = np.array(
        map(
            np.pad,
            _DATA,
            len(_DATA) * [((padding + 2, padding + 2), (padding, padding))],
            len(_DATA) * ["reflect"],
        )
    )
    return _DATApad


def pad(data, padding=2):
    """
    Padding function applied to folded spectral windows.
    Reflection is default padding.
    """

    sh = np.shape(data)
    t_pad = 16
    data_pad = np.pad(
        data, pad_width=((t_pad + 2, t_pad + 2), (t_pad, t_pad)), mode="reflect"
    )

    return data_pad


def unpad(data, diff=4, padding=2):
    """
    Unpadding function for recovering flag predictions.
    """
    sh = np.shape(data)
    t_unpad = sh[0]
    return data[padding[0] : sh[0] - padding[0], padding[1] : sh[1] - padding[1]]


def store_iterator(it):
    a = [x for x in it]

    return np.array(a)


def fold(data, ch_fold=16, padding=2):
    """
    Folding function for carving waterfall visibilities with additional normalized log
    and phase channels.
    Input: (Batch, Time, Frequency)
    Output: (Batch*FoldFactor, Time, Reduced Frequency, Channels)
    """
    sh = np.shape(data)
    _data = data.T.reshape(ch_fold, int(sh[1] / ch_fold), -1)
    _DATA = store_iterator(map(transpose, _data))
    _DATApad = store_iterator(map(pad, _DATA))

    DATA = np.stack(
        (
            store_iterator(map(normalize, _DATApad)),
            store_iterator(map(normphs, _DATApad)),
            np.mod(store_iterator(map(normphs, _DATApad)), np.pi),
        ),
        axis=-1,
    )

    return DATA


def unfoldl(data_fold, ch_fold=16, padding=2):
    """
    Unfolding function for recombining the carved label (flag) frequency windows back into a complete
    waterfall visibility.
    Input: (Batch*FoldFactor, Time, Reduced Frequency, Channels)
    Output: (Batch, Time, Frequency)
    """
    sh = np.shape(data_fold)
    data_unpad = data_fold[
        :, (padding + 2) : (sh[1] - (padding + 2)), padding : sh[2] - padding
    ]
    ch_fold, ntimes, dfreqs = np.shape(data_unpad)
    data_ = np.transpose(data_unpad, (0, 2, 1))
    _data = data_.reshape(ch_fold * dfreqs, ntimes).T

    return _data


def stacked_layer(
    input_layer,
    num_filter_layers,
    kt,
    kf,
    activation,
    stride,
    pool,
    bnorm=True,
    name="None",
    dropout=None,
    maxpool=True,
    mode=True,
):
    """
    Creates a 3x stacked layer of convolutional layers. Each layer uses the same kernel size.
    Batch normalized output is default and recommended for faster convergence, although
    not every may require it (???).
    Input: Tensor Variable (Batch*FoldFactor, Time, Reduced Frequency, Input Filter Layers)
    Output: Tensor Variable (Batch*FoldFactor, Time/2, Reduced Frequency/2, num_filter_layers)
    """
    conva = tf.layers.conv2d(
        inputs=input_layer,
        filters=num_filter_layers,
        kernel_size=[kt, kt],
        strides=[1, 1],
        padding="same",
        activation=activation,
    )
    if kt - 2 < 0:
        kt = 3
    if dropout is not None:
        convb = tf.layers.dropout(
            tf.layers.conv2d(
                inputs=conva,
                filters=num_filter_layers,
                kernel_size=[kt, kt],
                strides=[1, 1],
                padding="same",
                activation=activation,
            ),
            rate=dropout,
        )
    else:
        convb = tf.layers.conv2d(
            inputs=conva,
            filters=num_filter_layers,
            kernel_size=[kt, kt],
            strides=[1, 1],
            padding="same",
            activation=activation,
        )
    shb = convb.get_shape().as_list()

    convc = tf.layers.conv2d(
        inputs=convb,
        filters=num_filter_layers,
        kernel_size=(1, 1),
        padding="same",
        activation=activation,
    )
    if bnorm:
        bnorm_conv = tf.layers.batch_normalization(
            convc, scale=True, center=True, training=mode, fused=True
        )
    else:
        bnorm_conv = convc
    if maxpool:
        pool = tf.layers.max_pooling2d(
            inputs=bnorm_conv, pool_size=pool, strides=stride
        )
    elif maxpool is None:
        pool = bnorm_conv
    else:
        pool = tf.layers.average_pooling2d(
            inputs=bnorm_conv, pool_size=pool, strides=stride
        )

    return pool


def batch_accuracy(labels, predictions):
    """
    Returns the RFI class accuracy.
    """
    labels = tf.cast(labels, dtype=tf.int64)
    predictions = tf.cast(predictions, dtype=tf.int64)
    correct = tf.reduce_sum(
        tf.cast(tf.equal(tf.add(labels, predictions), 2), dtype=tf.int64)
    )
    total = tf.reduce_sum(labels)
    return tf.divide(correct, total)


def accuracy(labels, predictions):
    """
    Numpy version of RFI class accuracy.
    """
    correct = 1.0 * np.sum((labels + predictions) == 2)
    total = 1.0 * np.sum(labels == 1)
    print("correct", correct)
    print("total", total)
    try:
        return correct / total
    except BaseException:
        return 1.0


def MCC(tp, tn, fp, fn):
    """
    Calculates the Mathews Correlation Coefficient.
    """
    if tp == 0 and fn == 0:
        return tp * tn - fp * fn
    else:
        return (tp * tn - fp * fn) / np.sqrt(
            (1.0 * (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
        )


def f1(tp, tn, fp, fn):
    """
    Calculates the F1 Score.
    """
    precision = tp / (1.0 * (tp + fp))
    recall = tp / (1.0 * (tp + fn))
    return 2.0 * precision * recall / (precision + recall)


def SNRvsTPR(data, true_flags, flags):
    """
    Calculates the signal-to-noise ratio versus true positive rate (recall).
    """
    SNR = np.linspace(0.0, 4.0, 30)
    snr_tprs = []
    data_ = np.copy(data)
    flags_ = np.copy(flags)
    true_flags_ = np.copy(true_flags)
    for snr_ in SNR:
        snr_map = np.log10(data_ * flags_ / np.std(data_ * np.logical_not(true_flags)))
        snr_inds = snr_map < snr_
        confuse_mat = confusion_matrix(
            true_flags_[snr_inds].astype(int).reshape(-1),
            flags_[snr_inds].astype(int).reshape(-1),
        )
        if np.size(confuse_mat) == 1:
            tp = 1e-10
            tn = confuse_mat[0][0]
            fp = 1e-10
            fn = 1e-10
        else:
            try:
                tn, fp, fn, tp = confuse_mat.ravel()
            except BaseException:
                tp = np.nan
                fn = np.nan
        snr_tprs.append(MCC(tp, tn, fp, fn))
        data_[snr_inds] = 0.0
    return snr_tprs


def hard_thresh(layer, thresh=0.5):
    """
    Thresholding function for predicting based on raw FCN output.
    """
    layer_sigmoid = 1.0 / (1.0 + np.exp(-layer))
    return np.where(layer_sigmoid > thresh, np.ones_like(layer), np.zeros_like(layer))


def softmax(X):
    return np.exp(X) / np.sum(np.exp(X), axis=-1)


def ROC_stats(ground_truth, logits):
    ground_truth = np.reshape(ground_truth, [-1])
    thresholds = np.linspace(-1, 4.0, 30)
    FPR = []
    TPR = []
    MCC_arr = []
    F2 = []
    for thresh in thresholds:
        pred_ = hard_thresh(logits, thresh=thresh).reshape(-1)
        tn, fp, fn, tp = confusion_matrix(ground_truth, pred_).ravel()
        recall = tp / (1.0 * (tp + fn))
        precision = tp / (1.0 * (tp + fp))
        TPR.append(tp / (1.0 * (tp + fn)))
        FPR.append(fp / (1.0 * (fp + tn)))
        MCC_arr.append(MCC(tp, tn, fp, fn))
        F2.append(5.0 * recall * precision / (4.0 * precision + recall))
    best_thresh = thresholds[np.nanargmax(F2)]
    return FPR, TPR, MCC_arr, F2, best_thresh


def load_pipeline_dset(stage_type):
    """
    Additional loading function for specific evaluation datasets.
    """
    # f = h5py.File('JK_5Jan2019.h5','r')
    f = h5py.File("IDR21TrainingData_Raw_vX.h5", "r")
    # f = h5py.File('IDR21InitialFlags_v2.h5','r')
    # f = h5py.File('IDR21TrainingData_Raw_v2.h5')
    # f = h5py.File('IDR21TrainingData.h5','r')
    # f = h5py.File('RealVisRFI_v5.h5','r')
    # f = h5py.File('RawRealVis_v1.h5','r')
    # f = h5py.File('SimVis_Blips_100.h5','r')
    # f = h5py.File('SimVis_1000_v9.h5','r')
    try:
        if stage_type == "uv":
            return f["uv"]
        elif stage_type == "uvO":
            return f["uvO"]
        elif stage_type == "uvOC":
            return f["uvOC"]
        elif stage_type == "uvOCRS":
            return f["uvOCRS"]
        elif stage_type == "uvOCRSD":
            return f["uvOCRSD"]
    except BaseException:
        return f


def stride(input_data, input_labels):
    """
    Takes an input waterfall visibility with labels and strides across frequency,
    producing (Nchan - 64)/S new waterfalls to be folded.
    """
    spw_hw = 32  # spectral window half width
    nchans = 1024
    fold = nchans / (2 * spw_hw)
    sample_spws = random.sample(range(0, 60), fold)

    x = np.array(
        [
            input_data[:, i - spw_hw : i + spw_hw]
            for i in range(spw_hw, 1024 - spw_hw, (nchans - 2 * spw_hw) / 60)
        ]
    )
    x_labels = np.array(
        [
            input_labels[:, i - spw_hw : i + spw_hw]
            for i in range(spw_hw, 1024 - spw_hw, (nchans - 2 * spw_hw) / 60)
        ]
    )
    X = np.array([x[i].T for i in sample_spws])
    X_labels = np.array([x_labels[i].T for i in sample_spws])
    X_ = X.reshape(-1, 60).T
    X_labels = X_labels.reshape(-1, 60).T
    return X_, X_labels


def patchwise(data, labels):
    """
    A spectral window is strided over the visibility
    augmenting the existing training or evaluation
    datasets.
    """
    strided_dp = np.array(map(stride, data, labels))
    data_strided = np.copy(strided_dp[:, 0, :, :])
    labels_strided = np.copy(strided_dp[:, 1, :, :].astype(int))
    return data_strided, labels_strided


def expand_dataset(data, labels):
    """
    Comprehensive data augmentation function. Uses reflections, patchwise, gaussian noise, and
    gaussian blurring, to improve robustness of the DFCN model which increases performance
    when applied to real data.
    Bloat factor is how large to increase the dataset size.
    """
    bloat = 5
    sh = np.shape(data)
    out_data = []
    out_labels = []
    for i in range(bloat * sh[0]):
        rnd_num = np.random.rand()
        rnd_data_ind = np.random.randint(0, sh[0])
        order = np.random.choice(np.logspace(-4, -1, 10))
        noise = np.random.randn(sh[1], sh[2]) + 1j * np.random.randn(sh[1], sh[2])
        noise_data = np.copy(data[rnd_data_ind])
        noise_labels = np.copy(labels[rnd_data_ind])
        noise_data[:, :, 0] += order * np.abs(noise)
        if sh[3] > 1:
            noise_data[:, :, 1] += order * np.angle(noise)
        blur_sigma = np.random.uniform(0.0, 0.5)
        noise_data = ndimage.gaussian_filter(noise_data, sigma=blur_sigma)
        labels_blur = ndimage.gaussian_filter(noise_labels, sigma=blur_sigma)
        noise_labels = np.where(
            labels_blur > 0.1, np.ones_like(labels_blur), np.zeros_like(labels_blur)
        )
        if rnd_num < 0.3:
            out_data.append(noise_data[::-1, :, :])
            out_labels.append(noise_labels[::-1, :])
        elif rnd_num >= 0.3 and rnd_num < 0.6:
            out_data.append(noise_data[:, ::-1, :])
            out_labels.append(noise_labels[:, ::-1])
        elif rnd_num >= 0.6:
            out_data.append(noise_data[::-1, ::-1, :])
            out_labels.append(noise_labels[::-1, ::-1])
    return np.array(out_data), np.array(out_labels)


def expand_validation_dataset(data, labels):
    """
    Validation dataset augmentation trick for expanding a small dataset with a
    well known ground truth.
    """
    bloat = 10
    sh = np.shape(data)
    out_data = []
    out_labels = []
    for i in range(bloat * sh[0]):
        rnd_data_ind = np.random.randint(0, sh[0])
        spi = np.random.uniform(-2.7, -0.1)
        nos_jy = np.random.rand(sh[1], sh[2]) + 1j * np.random.rand(sh[1], sh[2])
        nos_jy *= (np.linspace(0.1, 0.2, 1024) / 0.1) ** (spi)
        nos_jy *= random.sample(np.logspace(-3, -1), 1)[0] * np.nanmean(
            np.abs(data[rnd_data_ind])
        )
        data_ = np.copy(data[rnd_data_ind]) + nos_jy
        labels_ = np.copy(labels[rnd_data_ind])
        if np.random.rand() > 0.5:
            data_ = data_[::-1, :]
            labels_ = labels_[::-1, :]
        if np.random.rand() > 0.5:
            data_ = data_[:, ::-1]
            labels_ = labels_[:, ::-1]
        if np.random.rand() > 0.5:
            data_, labels_ = patchwise([data_], [labels_])
        out_data.append(data_.reshape(-1, 1024))
        out_labels.append(labels_.reshape(-1, 1024))
    return out_data, out_labels


class RFIDataset:
    def __init__(self):
        """
        RFI class that handles loading, partitioning, and augmenting datasets.
        """
        print("Welcome to the HERA RFI training and evaluation dataset suite.")

    def load(
        self,
        tdset,
        vdset,
        batch_size,
        psize,
        hybrid=False,
        chtypes="AmpPhs",
        fold_factor=16,
        cut=False,
        patchwise_train=False,
        expand=False,
        predict=False,
    ):
        # load data
        if cut:
            self.cut = 14
        else:
            self.cut = 16
        self.chtypes = chtypes
        self.batch_size = batch_size
        self.iter_ct = 0
        self.pred_ct = 0
        print("A batch size of %i has been set." % self.batch_size)

        if vdset == "vanilla":
            f1 = h5py.File("SimVis_2000_v911.h5", "r")
        elif vdset == "":
            f1 = h5py.File("SimVis_2000_v911.h5", "r")
        else:
            f1 = load_pipeline_dset(vdset)

        if tdset == "v5":
            f2 = h5py.File("SimVis_v5.h5", "r")  # Load in simulated data
        elif tdset == "v11":
            f2 = h5py.File("SimVis_1000_v11.h5", "r")
        elif tdset == "v7":
            f2 = h5py.File("SimVis_2000_v7.h5", "r")
        elif tdset == "v8":
            f2 = h5py.File("SimVis_2000_v8.h5", "r")
        elif tdset == "v9":
            f2 = h5py.File("SimVis_1000_v9.h5", "r")
        elif tdset == "v911":
            f2 = h5py.File("SimVis_2000_v911.h5", "r")
        elif tdset == "v12":
            f2 = h5py.File("SimVis_2000_v12.h5", "r")
        elif tdset == "v13":
            # This is v9 + v11 + FineTune
            f2 = h5py.File("SimVis_2000_v911.h5", "r")
        elif tdset == "v4":
            f2 = h5py.File("SimVisRFI_15_120_v4.h5", "r")

        self.psize = psize  # Pixel pad size for individual carved bands

        # We want to augment our training dataset with the entirety of the simulated data
        # but with only half of the real data. The remaining real data half will become
        # the evaluation dataset
        f1_len = len(f1["data"])
        f1_sub = np.random.choice(f1_len)
        f2_len = len(f2["data"])

        f1_r = int(f1_len)
        f2_s = int(f2_len)

        f_factor_r = f1_r * [fold_factor]
        pad_r = f1_r * [self.psize]
        f_factor_s = f2_s * [fold_factor]
        pad_s = f2_s * [self.psize]
        self.dset_size = np.copy(f1_r) + np.copy(f2_s)
        self.fold_factor = fold_factor
        print("Size of real dataset: ", f1_r)
        print("")
        # Cut up real dataset and labels
        samples = range(f1_r)
        rnd_ind = np.random.randint(0, f1_r)

        dreal_choice = np.random.choice(range(0, f1_len), size=f1_r)
        dsim_choice = np.random.choice(range(0, f2_len), size=f2_s)
        data_real = np.array(f1["data"])[dreal_choice][:f1_r, :, :]
        labels_real = np.array(f1["flag"])[dreal_choice][:f1_r, :, :]
        data_sim = np.array(f2["data"])[dsim_choice][:f2_s, :, :]
        labels_sim = np.array(f2["flag"])[dsim_choice][:f2_s, :, :]
        self.data_real = np.array(np.copy(f1["data"]))
        self.labels_real = np.array(np.copy(f1["flag"]))
        self.data_sim = np.array(np.copy(f2["data"]))
        self.labels_sim = np.array(np.copy(f2["flag"]))
        time0 = time()

        if chtypes == "AmpPhs":
            f_real = (
                np.array(map(fold, data_real, f_factor_r, pad_r))[:, :, :, :, :2]
            ).reshape(
                -1, 2 * (self.psize + 2) + 60, 2 * self.psize + 1024 / fold_factor, 2
            )
            f_real_labels = np.array(
                map(foldl, labels_real, f_factor_r, pad_r)
            ).reshape(
                -1, 2 * (self.psize + 2) + 60, 2 * self.psize + 1024 / fold_factor
            )
            del data_real
            del labels_real
            # Cut up sim dataset and labels
            if patchwise_train:
                data_sim_patch, labels_sim_patch = patchwise(data_sim, labels_sim)
                data_sim = np.array(np.vstack((data_sim, data_sim_patch)))
                labels_sim = np.array(np.vstack((labels_sim, labels_sim_patch)))
                print("data_sim size: {0}".format(np.shape(data_sim)))
                f_sim = (np.array(map(fold, data_sim))[:, :, :, :, :2]).reshape(
                    -1, self.psize, self.psize, 2
                )
                f_sim_labels = np.array(map(foldl, labels_sim)).reshape(
                    -1, self.psize, self.psize
                )
                print("Expanded training dataset size: {0}".format(np.shape(f_sim)))
            else:
                f_sim = (
                    np.array(map(fold, data_sim, f_factor_s, pad_s))[:, :, :, :, :2]
                ).reshape(
                    -1,
                    2 * (self.psize + 2) + 60,
                    2 * self.psize + 1024 / fold_factor,
                    2,
                )
                f_sim_labels = np.array(
                    map(foldl, labels_sim, f_factor_s, pad_s)
                ).reshape(
                    -1, 2 * (self.psize + 2) + 60, 2 * self.psize + 1024 / fold_factor
                )
                if expand:
                    f_sim, f_sim_labels = expand_dataset(f_sim, f_sim_labels)
                del data_sim
                del labels_sim
        elif chtypes == "AmpPhs2":
            f_real = np.array(map(fold, data_real, f_factor_r, pad_r)).reshape(
                -1, self.psize, self.psize, 3
            )
            f_real_labels = np.array(
                map(foldl, labels_real, f_factor_r, pad_r)
            ).reshape(-1, self.psize, self.psize)
            # Cut up sim dataset and labels
            f_sim = np.array(map(fold, data_sim, f_factor_s, pad_s)).reshape(
                -1, self.psize, self.psize, 3
            )
            f_sim_labels = np.array(map(foldl, labels_sim, f_factor_s, pad_s)).reshape(
                -1, self.psize, self.psize
            )
        elif chtypes == "Amp":
            f_real = (np.array(map(fold, data_real))[:, :, :, :, 0]).reshape(
                -1, 2 * (self.psize + 2) + 60, 2 * self.psize + 1024 / fold_factor, 1
            )
            print("f_real: ", np.shape(f_real))
            f_real_labels = np.array(map(foldl, labels_real)).reshape(
                -1, 2 * (self.psize + 2) + 60, 2 * self.psize + 1024 / fold_factor
            )
            if patchwise_train:
                data_sim_patch, labels_sim_patch = patchwise(data_sim, labels_sim)
                data_sim = np.array(np.vstack((data_sim, data_sim_patch)))
                labels_sim = np.array(np.vstack((labels_sim, labels_sim_patch)))
                f_sim = (np.array(map(fold, data_sim))[:, :, :, :, 0]).reshape(
                    -1, self.psize, self.psize, 1
                )
                f_sim_labels = np.array(map(foldl, labels_sim)).reshape(
                    -1, self.psize, self.psize
                )
            else:
                f_sim = (
                    np.array(map(fold, data_sim, f_factor_s, pad_s))[:, :, :, :, 0]
                ).reshape(
                    -1,
                    2 * (self.psize + 2) + 60,
                    2 * self.psize + 1024 / fold_factor,
                    1,
                )
                f_sim_labels = np.array(
                    map(foldl, labels_sim, f_factor_s, pad_s)
                ).reshape(
                    -1, 2 * (self.psize + 2) + 60, 2 * self.psize + 1024 / fold_factor
                )
        elif chtypes == "Phs":
            f_real = np.array(map(fold, data_real, f_factor_r, pad_r)).reshape(
                -1, self.psize, self.psize, 1
            )
            f_real_labels = np.array(
                map(foldl, labels_real, f_factor_r, pad_r)
            ).reshape(-1, self.psize, self.psize)
            f_sim = np.array(map(fold, data_sim, f_factor_s, pad_s)).reshape(
                -1, self.psize, self.psize, 1
            )
            f_sim_labels = np.array(map(foldl, labels_sim, f_factor_s, pad_s)).reshape(
                -1, self.psize, self.psize
            )

        print("Training dataset loaded.")
        print("Training dataset size: ", np.shape(f_real))

        print("Simulated training dataset loaded.")
        print("Training dataset size: ", np.shape(f_sim))

        real_sh = np.shape(f_real)

        if chtypes == "AmpPhsCmp":
            d_type = np.complex64
        else:
            d_type = np.float64
        real_len = np.shape(f_real)[0]
        if hybrid:
            print("Hybrid training dataset selected.")
            # We want to mix the real and simulated datasets
            # and then keep some real datasets for evaluation
            real_len = np.shape(f_real)[0]
            self.eval_data = np.asarray(
                f_real[: int(real_len / 2), :, :, :], dtype=d_type
            )
            self.eval_labels = np.asarray(
                f_real_labels[: int(real_len / 2), :, :], dtype=np.int32
            ).reshape(-1, real_sh[1] * real_sh[2])

            train_data = np.vstack((f_real[int(real_len / 2) :, :, :, :], f_sim))
            train_labels = np.vstack(
                (f_real_labels[int(real_len / 2) :, :, :], f_sim_labels)
            )
            hybrid_len = np.shape(train_data)[0]
            mix_ind = np.random.permutation(hybrid_len)

            self.train_data = train_data[mix_ind, :, :, :]
            self.train_labels = train_labels[mix_ind, :, :].reshape(
                -1, real_sh[1] * real_sh[2]
            )
            self.eval_len = np.shape(self.eval_data)[0]
            self.train_len = np.shape(self.train_data)[0]
        else:
            # Format evaluation dataset
            sim_len = np.shape(f_sim)[0]
            self.eval_data = np.asarray(
                f_sim[int(sim_len * 0.8) :, :, :, :], dtype=d_type
            )
            self.eval_labels = np.asarray(
                f_sim_labels[int(sim_len * 0.8) :, :, :], dtype=np.int32
            ).reshape(-1, real_sh[1] * real_sh[2])
            eval1 = np.shape(self.eval_data)[0]

            # Format training dataset
            self.train_data = np.asarray(
                f_sim[: int(sim_len * 0.8), :, :, :], dtype=d_type
            )
            self.train_labels = np.asarray(
                f_sim_labels[: int(sim_len * 0.8), :, :], dtype=np.int32
            ).reshape(-1, real_sh[1] * real_sh[2])

            train0 = np.shape(self.train_data)[0]
            self.test_data = self.eval_data[rnd_ind, :, :, :].reshape(
                1, real_sh[1], real_sh[2], real_sh[3]
            )
            self.test_labels = self.eval_labels[rnd_ind, :].reshape(
                1, real_sh[1] * real_sh[2]
            )
            self.eval_len = np.shape(self.eval_data)[0]
            self.train_len = np.shape(self.train_data)[0]

    def reload(self, fold_factor, psize, time_subsample=False, batch=None):
        d_type = np.float64
        f1_r = int(len(self.data_real))
        f2_s = int(len(self.data_sim))
        if batch:
            dreal_choice = np.random.choice(range(0, f1_r), size=batch)
            dsim_choice = np.random.choice(range(0, f2_s), size=batch)
        else:
            dreal_choice = np.random.choice(range(0, f1_r), size=f1_r)
            dsim_choice = np.random.choice(range(0, f2_s), size=f2_s)

        f_factor_r = f1_r * [fold_factor]
        pad_r = f1_r * [psize]
        f_factor_s = f2_s * [fold_factor]
        pad_s = f2_s * [psize]
        if time_subsample:
            t0 = np.random.randint(0, 20)
            t1 = np.random.randint(40, 60)
            pad_t0 = t0
            pad_t1 = 60 - t1
            data_sim = np.pad(
                self.data_sim[dsim_choice][:, t0:t1, :],
                ((0, 0), (pad_t0, pad_t1), (0, 0)),
                mode="reflect",
            )
            labels_sim = np.pad(
                self.labels_sim[dsim_choice][:, t0:t1, :],
                ((0, 0), (pad_t0, pad_t1), (0, 0)),
                mode="reflect",
            )
            f_sim = (
                np.array(map(fold, data_sim, f_factor_s, pad_s))[:, :, :, :, :2]
            ).reshape(-1, 2 * (psize + 2) + 60, 2 * psize + 1024 / fold_factor, 2)
            f_sim_labels = np.array(map(foldl, labels_sim, f_factor_s, pad_s)).reshape(
                -1, 2 * (psize + 2) + 60, 2 * psize + 1024 / fold_factor
            )
            print("Permuting dataset along time and frequency.")
            if expand:
                f_sim, f_sim_labels = expand_dataset(f_sim, f_sim_labels)
        else:
            f_sim = (
                np.array(map(fold, self.data_sim, f_factor_s, pad_s))[:, :, :, :, :2]
            ).reshape(-1, 2 * (psize + 2) + 60, 2 * psize + 1024 / fold_factor, 2)
            f_sim_labels = np.array(
                map(foldl, self.labels_sim, f_factor_s, pad_s)
            ).reshape(-1, 2 * (psize + 2) + 60, 2 * psize + 1024 / fold_factor)

        sim_len = np.shape(f_sim)[0]
        sim_sh = np.shape(f_sim)
        print("Sim Shape", sim_sh)
        self.eval_data = np.asarray(
            f_sim[int(sim_len * 0.8) :, :, :, :], dtype=d_type
        ).reshape(-1, sim_sh[1], sim_sh[2], 2)
        self.eval_labels = np.asarray(
            f_sim_labels[int(sim_len * 0.8) :, :, :], dtype=np.int32
        ).reshape(-1, sim_sh[1] * sim_sh[2])
        eval1 = np.shape(self.eval_data)[0]
        # Format training dataset
        self.train_data = np.asarray(
            f_sim[: int(sim_len * 0.8), :, :, :], dtype=d_type
        ).reshape(-1, sim_sh[1], sim_sh[2], 2)
        self.train_labels = np.asarray(
            f_sim_labels[: int(sim_len * 0.8), :, :], dtype=np.int32
        ).reshape(-1, sim_sh[1] * sim_sh[2])
        self.eval_len = np.shape(self.eval_data)[0]
        self.train_len = np.shape(self.train_data)[0]

    def load_pyuvdata(self, filename, chtypes, fold_factor, psize):
        from pyuvdata import UVData

        uv = UVData()
        uv.read_miriad(filename)
        self.uv = copy(uv)
        self.antpairs = copy(uv.get_antpairs())
        self.dset_size = np.shape(self.uv.data_array)[0] / 60
        self.chtypes = chtypes
        self.fold_factor = fold_factor
        self.psize = psize

    def predict_pyuvdata(self):
        if self.chtypes == "AmpPhs":
            print(np.shape(self.uv.get_data((1, 11))))
            f_real = (
                np.array(fold(self.uv.get_data((1, 11)), self.fold_factor, self.psize))[
                    :, :, :, :2
                ]
            ).reshape(
                -1,
                2 * (self.psize + 2) + 60,
                2 * self.psize + 1024 / self.fold_factor,
                2,
            )
        elif self.chtypes == "Amp":
            f_real = (
                np.array(fold(self.uv.get_data(self.antpairs.pop(0)), self.cut, 2))[
                    :, :, :, 0
                ]
            ).reshape(
                -1,
                2 * (self.psize + 2) + 60,
                2 * self.psize + 1024 / self.fold_factor,
                1,
            )
        return f_real

    def next_train(self):
        if self.iter_ct == 0:
            self.indices = np.array(range(self.dset_size)).reshape(-1, self.batch_size)
        elif self.iter_ct >= self.dset_size / self.batch_size:
            self.iter_ct = 0

        batch_inds = self.indices[self.iter_ct, :]
        self.iter_ct += 1

        return self.train_data[batch_inds, :, :, :], self.train_labels[batch_inds, :]

    def change_batch_size(self, new_bs):
        self.batch_size = new_bs

    def permute_dset(self):
        indices = range(len(self.train_data))
        perm_indices = np.random.permutation(indices)
        self.train_data = self.train_data[perm_indices]
        self.train_labels = self.train_labels[perm_indices]

    def next_eval(self):
        rand_batch = random.sample(range(self.eval_len), self.batch_size)
        return self.eval_data[rand_batch, :, :, :], self.eval_labels[rand_batch, :]

    def next_predict(self):
        # Iterates through prediction dataset, doesn't take random samples
        if self.chtypes == "AmpPhs":
            f_real = (
                np.array(fold(self.data_real[self.pred_ct, :, :], self.cut, 16))[
                    :, :, :, :2
                ]
            ).reshape(
                -1,
                2 * (self.psize + 2) + 60,
                2 * self.psize + 1024 / self.fold_factor,
                2,
            )
            f_real_labels = np.array(
                foldl(self.labels_real[self.pred_ct, :, :], self.cut, 16)
            ).reshape(
                -1, 2 * (self.psize + 2) + 60, 2 * self.psize + 1024 / self.fold_factor
            )
        elif self.chtypes == "Amp":
            f_real = (
                np.array(fold(self.data_real[self.pred_ct, :, :], self.cut, 2))[
                    :, :, :, 0
                ]
            ).reshape(-1, self.psize, self.psize, 1)
            f_real_labels = np.array(
                foldl(self.labels_real[self.pred_ct, :, :], self.cut, 2)
            ).reshape(-1, self.psize, self.psize)
        data_return = self.data_real[self.pred_ct, :, :]
        self.pred_ct += 1
        return data_return, f_real, f_real_labels

    def random_test(self, samples):
        ind = random.sample(range(np.shape(self.eval_data)[0]), samples)
        if self.chtypes == "Amp":
            ch = 1
        elif self.chtypes == "AmpPhs":
            ch = 2
        elif self.chtypes == "AmpPhs2":
            ch = 3
        return (
            self.eval_data[ind, :, :, :].reshape(samples, self.psize, self.psize, ch),
            self.eval_labels[ind, :].reshape(samples, self.psize * self.psize),
        )

    def get_size(self):
        # Return dataset size
        return self.dset_size


# Keras helper functions
def keras_convert_wf(wf, Nt_min=64, Nf_min=64):
    """Convert an input waterfall to the correct type expected by the Keras model.

    Due to the max pooling in the Keras model, we may need to pad the input
    waterfall to be the correct shape. This padding will be done symmetrically
    on both ends of the waterfall.

    Parameters
    ----------
    wf : ndarray
        A 3d ndarray of shape (Nbatch, Ntimes, Nfreq) and complex dtype. Will
        be padded if dimension aare not sufficient for the given input.
    Nt_min : int
        The minimum size of the waterfall along the time dimension.
    Nf_min : int
        The minimum size of the waterfall along the frequency dimension.

    Returns
    -------
    wf_amp : ndarray
        A 4d ndarray of shape (Nbatch, Ntimes', Nfreq', 1) and real dtype.
        Ntimes' and Nfreq' will be padded if they are not sufficiently
        large for the given model. The last axis is log10(amp) of the
        input complex number.

    wf_phs : ndarray
        A 4d ndarray of shape (Nbatch, Ntimes', Nfreq', 1) and real dtype.
        Ntimes' and Nfreq' will be padded if they are not sufficiently
        large for the given model. The last axis is phase (angle) of the
        input complex number.
    """
    if len(wf.shape) != 3:
        raise ValueError("wf should be a 3-dimensional ndarray")
    if wf.dtype not in (np.complex, np.complex64, np.complex128):
        raise ValueError("wf should have a complex dtype")
    input_shape = wf.shape
    # convert to amplitude and phase
    wf_amp = np.empty((*input_shape, 1), dtype=np.float)
    wf_phs = np.empty((*input_shape, 1), dtype=np.float)

    wf_amp[:, :, :, 0] = np.log10(np.abs(wf))
    wf_phs[:, :, :, 0] = np.angle(wf)

    # maybe pad times
    if input_shape[1] < Nt_min:
        # pad it out
        diff = Nt_min - input_shape[1]
        Npad = diff // 2 + diff % 2  # extra 1 in case difference is odd
        wf_amp = np.pad(wf_amp, ((0, 0), (Npad, Npad), (0, 0), (0, 0)),
                        mode="reflect")
        wf_phs = np.pad(wf_phs, ((0, 0), (Npad, Npad), (0, 0), (0, 0)),
                        mode="reflect")

    # maybe pad freqs
    if input_shape[2] < Nf_min:
        # pad it out
        diff = Nf_min - input_shape[2]
        Npad = diff // 2 + diff % 2  # extra 1 in case difference is odd
        wf_amp = np.pad(wf_amp, ((0, 0), (0, 0), (Npad, Npad), (0, 0)),
                        mode="reflect")
        wf_phs = np.pad(wf_phs, ((0, 0), (0, 0), (Npad, Npad), (0, 0)),
                        mode="reflect")

    return wf_amp, wf_phs


def keras_convert_flags(flags, Nt_min=64, Nf_min=64):
    """Convert an input flag array to the correct type expected by the Keras model.

    Due to the max pooling in the Keras model, we may need to pad the input
    flags to be the correct shape. This padding will be done symmetrically
    on both ends of the flag waterfall.

    Parameters
    ----------
    flags : ndarray
        A 3d ndarray of shape (Nbatch, Ntimes, Nfreq) and boolean dtype. Will
        be padded if dimension aare not sufficient for the given input.
    Nt_min : int
        The minimum size of the waterfall along the time dimension.
    Nf_min : int
        The minimum size of the waterfall along the frequency dimension.

    Returns
    -------
    flags_out : ndarray
        A 4d ndarray of shape (Nbatch, Ntimes', Nfreq', 1) and integer dtype.
        Ntimes' and Nfreq' will be padded if they are not sufficiently
        large for the given model.
    """
    if len(flags.shape) != 3:
        raise ValueError("flags should be a 3-dimensional ndarray")
    input_shape = flags.shape
    # convert to int
    flags_out = flags.astype(np.int32).reshape((*input_shape, 1))

    # maybe pad times
    if input_shape[1] < Nt_min:
        # pad it out
        diff = Nt_min - input_shape[1]
        Npad = diff // 2 + diff % 2  # extra 1 in case difference is odd
        flags_out = np.pad(flags_out, ((0, 0), (Npad, Npad), (0, 0), (0, 0)),
                           mode="reflect")

    # maybe pad freqs
    if input_shape[2] < Nf_min:
        # pad it out
        diff = Nf_min - input_shape[2]
        Npad = diff // 2 + diff % 2  # extra 1 in case difference is odd
        flags_out = np.pad(flags_out, ((0, 0), (0, 0), (Npad, Npad), (0, 0)),
                           mode="reflect")

    return flags_out


def keras_recall_metric(y_true, y_pred):
    """Define a recall metric to use in Keras.

    Parameters
    ----------
    y_true : ndarray
        The "true" values according to Keras.
    y_pred : ndarray
        The "predicted" values according to Keras.

    Returns
    -------
    recall : float
        The recall value, defined as the number true positives divided by the
        total number of positives.
    """
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall


def keras_precision_metric(y_true, y_pred):
    """Define a precision metric to use in Keras.

    Parameters
    ----------
    y_true : ndarray
        The "true" values according to Keras.
    y_pred : ndarray
        The "predicted" values according to Keras.

    Returns
    -------
    precision : float
        The precision value, defined as the number of true positives divided
        by the number of predicted positives.
    """
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision


def keras_f2_metric(y_true, y_pred):
    """Define a F2 metric to use in Keras.

    Parameters
    ----------
    y_true : ndarray
        The "true" values according to Keras.
    y_pred : ndarray
        The "predicted" values according to Keras.

    Returns
    -------
    f2 : float
        The F2 metric, which is a weighted combination of the precision and
        recall metrics. The F2 score weights recall higher than precision, which
        places a greater emphasis on false negatives.
    """
    precision = keras_precision_metric(y_true, y_pred)
    recall = keras_recall_metric(y_true, y_pred)
    f2 = (1 + 2 ** 2) * (precision * recall) / (2 ** 2 * precision + recall + K.epsilon())
    return f2


class KerasFitter(object):
    """A class for fitting the Keras model given some input data.

    Parameters
    ----------
    None
    """
    def __init__(self):
        """Initialize the object."""

        return

    def load_data(self, data_fn, n_train, n_test):
        """Load the data and make sure it's the right size.

        Parameters
        ----------
        data_fn : str
            The full path to the data to use for training and testing.

        n_train : int
            The number of samples to use as training data.

        n_test : int
            The number of samples to use as testing/evaluation data.

        Returns
        -------
        None

        Notes
        -----
        This method will save the training and testing data on the object.

        Raises
        ------
        AssertionError
            This is raised if the training data and flags are not the same size,
            or if the testing data and flags are not the same size.
        """
        # save attributes
        self.data_fn = data_fn
        self.n_train = n_train
        self.n_test = n_test

        print("Reading data from {}...".format(self.data_fn))
        with h5py.File(self.data_fn, "r") as f:
            train_data = f["data"][:self.n_train, :, :]
            train_flag = f["flag"][:self.n_train, :, :]
            test_data = f["data"][self.n_train:self.n_train + self.n_test, :, :]
            test_flag = f["flag"][self.n_train:self.n_train + self.n_test, :, :]

        # resize data as necessary
        print("Formatting data...")
        self.train_data_amp, self.train_data_phs = keras_convert_wf(train_data)
        self.train_flag = keras_convert_flags(train_flag)
        self.test_data_amp, self.test_data_phs = keras_convert_wf(test_data)
        self.test_flag = keras_convert_flags(test_flag)

        # make sure things are the right size/shape
        assert self.train_data_amp.shape == self.train_data_phs.shape
        assert self.train_data_amp.shape == self.train_flag.shape
        assert self.test_data_amp.shape == self.test_data_phs.shape
        assert self.test_data_amp.shape == self.test_flag.shape

        return

    def train_model(
            self,
            batch_normalize=True,
            dropout_rate=0.4,
            alpha=0.2,
            pool_size=(2, 2),
            pool_stride=(2, 2),
            optimizer="adam",
            loss="sparse_categorical_crossentropy",
            metrics=["sparse_categorical_accuracy"],
            tb_callback=True,
            epochs=200,
            batch_size=32,
            verbose=False,
    ):
        """Train a Keras model with the loaded dataset.

        Many of the options for this method are based on those in the
        keras_model.amp_phs_model function. See the documentation there for
        more details.

        Parameters
        ----------
        batch_normalize : bool
            Whether to apply batch normalization layers to the models.
        dropout_rate : float
            What dropout rate to use for dropout layers.
        alpha : float
            What alpha value to use for Leaky ReLU layers.
        pool_size : tuple of ints
            What pool size to use.
        pool_stride : tuple of ints
            What pool stride to use.
        optimizer : str
            Which optimizer to use for training the model. It will be passed
            into the model.compile method as the "optimizer" kwarg.
        loss : str
            Which loss to use for training the model. It will be passed into
            the model.compile method as the "loss" kwarg.
        metrics : list of str or functions
            The metrics to use for the Keras training process. Entries in the
            list should be strings of pre-programmed Keras metrics or functions
            that can be applied to the output (e.g., as in the
            `keras_recall_metric` function defined above).
        tb_callback : bool
            Whether to have a callback to TensorBoard as part of the fit process.
            Output will be saved to a directory called "logs" in the current
            working directory.
        epochs : int
            The number of epochs to train the model for.
        batch_size : int
            The batch size to use during training.
        verbose : bool
            Whether to print a summary of the Keras model or not.

        Returns
        -------
        None

        Notes
        -----
        The trained model will be saved on the object.

        Raises
        ------
        ValueError
            This is raised if the training data is not present on the object.
        """
        if not hasattr(self, "train_data_amp"):
            raise ValueError("The `load_data` method must be called before training.")
        if not hasattr(self, "model"):
            # make a new model
            input_shape = self.train_data_amp.shape[1:]
            model = keras_model.amp_phs_model(
                input_shape,
                batch_normalize=batch_normalize,
                dropout_rate=dropout_rate,
                alpha=alpha,
                pool_size=pool_size,
                pool_stride=pool_stride,
            )
            self.model = model

        if verbose:
            print(self.model.summary())

        # compile model
        model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

        if tb_callback:
            tb = keras.callbacks.TensorBoard(log_dir="./logs")
            callbacks = [tb]
        else:
            callbacks = []

        # fit model
        model.fit(
            [self.train_data_amp, self.train_data_phs],
            [self.train_flag],
            validation_data=([self.test_data_amp, self.test_data_phs],
                             [self.test_flag]),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
        )

        # save trained model on the object
        self.model = model

        return

    def save_model(self, model_fn):
        """Save a trained Keras model to disk.

        Parameters
        ----------
        model_fn : str
            The full path to the location to save the file.

        Returns
        -------
        None

        Raises
        ------
        ValueError
            This is raised if a model attribute is not present on the object.
        """
        if not hasattr(self, "model"):
            raise ValueError("The `train_model` method must be called before "
                             "saving the model.")
        print("Saving {}...".format(model_fn))
        self.model.save(model_fn)
        return

    def load_model(self, model_fn, overwrite_model=False):
        """Load a trained Keras model from disk.

        Parameters
        ----------
        model_fn : str
            The full path to the location of the model file.

        Returns
        -------
        None

        Raises
        ------
        ValueError
            This error is raised if the `model` attribute already exists and
            thus would be replaced by the model on disk, but the user has not
            explicitly allowed for the model to be overwritten.
        """
        if hasattr(self, "model") and not overwrite_model:
            raise ValueError("The object already has a `model` defined, and "
                             "reading in from disk would overwrite this model. "
                             "Run with overwrite_model=True to replace the model "
                             "on the object.")
        self.model = load_model(model_fn)

        return

    def make_prediction(self, input_data):
        """Make a prediction on new data.

        Parameters
        ----------
        input_data : ndarray
            The data to run prediction on, with shape (Nwaterfalls, Ntimes, Nfreqs).
            and complex dtype. It should have the same time and frequency
            dimensionality used to train the model.

        Returns
        -------
        predicted_flags : ndarray
            The predicted flags for the input data, of size (Nwaterfalls, Ntimes,
            Nfreqs) and boolean dtype.
        """
        if not hasattr(self, "model"):
            raise ValueError("A model must be trained or loaded before prediction "
                             "can be done.")
        input_shape = input_data.shape
        input_data_amp, input_data_phs = keras_convert_wf(input_data)
        predicted_flags = model.predict([input_data_amp, input_data_phs])
        return predicted_flags.astype(np.bool).reshape(input_shape)
