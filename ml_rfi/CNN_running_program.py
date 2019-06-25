import sys
import numpy as np
import tensorflow as tf
from pyuvdata import UVData
from ml_rfi import helper_functions as hf
from ml_rfi.amp_model import AmpFCN
from ml_rfi.amp_phs_model import AmpPhsFCN
from time import time

class Predictor:

    def __init__(self, filename, CNN_model, ch_input = 2,
                 pad_size = 16, f_factor = 16,
                 chtypes = 'AmpPhs'):
        self.filename = filename
        self.ch_input = ch_input
        self.CNN_model = CNN_model
        self.pad_size = pad_size
        self.f_factor = f_factor
        self.chtypes = chtypes

        # create variables for the input, output, and kernel
        vis_input = tf.placeholder(tf.float32, shape=[None, None, None, ch_input])
        mode_bn = tf.placeholder(tf.bool)
        d_out = tf.placeholder(tf.float32)
        kernel_size = tf.placeholder(tf.int32)

        # Create a PyUVData object to store the input data
        self.uvd = UVData()

    def check_antennas():
        uvd.read_uvh5(self.filename, read_data = False)
        print("Antenna Options: ")
        print(np.unique(uvd.ant_1_array))

    def pick_antennas(antenna_array):
        uvd.read_uvh5(self.filename, antenna_nums=antenna_array)

    def prepare_input(ants, all_antennas = False):

        if (all_antennas):

        else:

            num_comb = len(ants) * (len(ants) - 1)/2
            ant1 = ants.tolist()
            ant2 = ants.tolist()
            wf_data = np.zeros([num_comb, 60, 1024, 4])

            counter = 0
            for i in range(ant1):
                for j in range(ant2):
                    if (ant1[i] == ant2[j]):
                        continue

                    temp = uvd.get_data(ant1[i], ant2[j])
                    wf_data[counter] = temp[:, :, 0].reshape(1, np.shape(temp)[0], np.shape(temp)[1])
                    ant1.remove(ant1[i])
                    counter += 1

            return wf_data


    def make_prediction(wfs):
        RFI_guess = AmpPhsFCN(vis_input, mode_bn=mode_bn, d_out=d_out, reuse=tf.AUTO_REUSE)
        init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        savr = tf.train.Saver()

        with tf.Session() as sess:
            sess.run(init)
            savr.restore(sess, tf_model)
            for i in range(len(wfs)):
                batch_x = (np.array(hf.fold(wfs[i, :, :, :],f_factor,pad_size))[:,:,:,:2]).reshape(-1,2*(pad_size+2)+60,int(2*pad_size+1024/f_factor),2)
                time0 = time()
                ind = 0
                ct = 0
                pred_start = time()
                g = sess.run(RFI_guess, feed_dict={vis_input: batch_x, mode_bn: True})
                pred_unfold = hf.store_iterator(
                    map(hf.unfoldl,tf.reshape(tf.argmax(g,axis=-1),
                    [-1,int(f_factor),int(2*(pad_size+2)+60),
                    int(2*pad_size+1024/f_factor)]).eval(),[f_factor],[pad_size]))

                pred_time = time() - pred_start
                if chtypes == 'AmpPhs':
                    thresh = 0.62
                else:
                    thresh = 0.385

                y_pred = np.array(pred_unfold[0]).reshape(-1,1024)
                data = wfs[i, :, :, :]
                # plt.subplot(1, len(wfs), i+1)
                # plt.imshow(np.log10(np.abs(data[0, :, :])*np.logical_not(y_pred)),aspect='auto',vmin=-4,vmax=0.)
                # plt.colorbar()
