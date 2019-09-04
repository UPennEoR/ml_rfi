from __future__ import division, print_function, absolute_import

import os
import sys
from time import time
from glob import glob
from copy import copy

import numpy as np
import h5py
import matplotlib
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.metrics import roc_curve
from sklearn.metrics import confusion_matrix

import ml_rfi.helper_functions as hf
from ml_rfi.AmpModel import AmpFCN
from ml_rfi.AmpPhsModel import AmpPhsFCN

# Run on a single GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
args = sys.argv[1:]

filename = "../zen.2458098.45361.xx.HH.uv"
chtypes = "AmpPhs"
ch_input = 2
FCN_version = "v100"
tdset_type = "v00"
edset_type = "uv"
tdset_version = "v00"
mods = "test"
pad_size = 16  # 68
f_factor = 16
slice_size = 16
model_name = "AmpPhsv9SimRealv13_64BSizeDynamicVis"
model_dir = glob("./" + model_name + "/model_*")

try:
    models2sort = [
        int(model_dir[i].split("/")[2].split(".")[0].split("_")[1])
        for i in range(len(model_dir))
    ]
    model_ind = np.argmax(models2sort)
    model = "model_" + str(models2sort[model_ind]) + ".ckpt"
    print(model)
except OSError:
    print("Cannot find model.")

vis_input = tf.placeholder(
    tf.float32, shape=[None, None, None, ch_input]
)  # 2*(pad_size+2)+60, 2*pad_size+1024/f_factor, ch_input])
mode_bn = tf.placeholder(tf.bool)
d_out = tf.placeholder(tf.float32)
kernel_size = tf.placeholder(tf.int32)

# Initialize Network
if chtypes == "Amp":
    RFI_guess = AmpFCN(vis_input, mode_bn=mode_bn, d_out=d_out)
elif chtypes == "AmpPhs":
    RFI_guess = AmpPhsFCN(vis_input, mode_bn=mode_bn, d_out=d_out)

# Initialize the variables (i.e. assign their default value)
init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())

# Load dataset
dset = hf.RFIDataset()
dset_start_time = time()
dset.load_pyuvdata(filename, chtypes, f_factor, pad_size)
# dset.load(tdset_version,vdset,batch_size,pad_size,chtypes=chtypes)
dset_load_time = (time() - dset_start_time) / dset.get_size()  # per visibility
saver = tf.train.Saver()
with tf.Session() as sess:
    # Run the initializer
    sess.run(init)
    # Check to see if model exists
    if len(model_dir) > 0:
        print("Model exists. Loading last save.")
        saver.restore(sess, "./" + model_name + "/" + model)
        print("Model " + "./" + model_name + "/" + model + " loaded.")
    else:
        raise ValueError("No Model Found. Pipeline killed.")

    time0 = time()
    ind = 0
    print("N=%i number of baselines time: " % 1, time() - time0)
    ct = 0
    batch_x = (
        dset.predict_pyuvdata()
    )  # np.array([dset.predict_pyuvdata() for i in range(1)]).reshape(-1,2*(pad_size+2)+60, 2*pad_size+1024/f_factor,2)
    print(np.shape(batch_x))
    pred_start = time()
    g = sess.run(RFI_guess, feed_dict={vis_input: batch_x, mode_bn: True})
    print("Current Visibility: {0}".format(ct))
    pred_unfold = map(
        hf.unfoldl,
        tf.reshape(
            tf.argmax(g, axis=-1),
            [
                -1,
                int(f_factor),
                int(2 * (pad_size + 2) + 60),
                int(2 * pad_size + 1024 / f_factor),
            ],
        ).eval(),
        [f_factor],
        [pad_size],
    )
    pred_time = time() - pred_start
    if chtypes == "AmpPhs":
        thresh = 0.62  # 0.329 real #0.08 sim
    else:
        thresh = 0.385  # 0.385 real #0.126 sim
    y_pred = np.array(pred_unfold[0]).reshape(-1, 1024)
    data = dset.uv.get_data((1, 11))

    plt.imshow(
        np.log10(np.abs(data) * np.logical_not(y_pred)),
        aspect="auto",
        vmin=-4,
        vmax=0.0,
    )
    plt.colorbar()
    plt.savefig("AdamBMissedRFITest.png")
    ct += 1
    #       y_pred = hf.hard_thresh(pred_unfold[:,64*ci_1:1024-64*ci_2],thresh=thresh).reshape(-1)
    process_time = (time() - time0) / 60.0
    print("Total processing time: {0} mins".format(process_time))
    print("Data throughput: {0} Vis/h/gpu".format(1100.0 / (process_time / 60.0)))
