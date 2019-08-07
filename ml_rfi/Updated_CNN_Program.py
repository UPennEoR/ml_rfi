import matplotlib.pyplot as plt
import sys
import numpy as np
import tensorflow as tf
from pyuvdata import UVData
from pyuvdata import UVFlag
from ml_rfi import Updated_helper_functions as hf
from ml_rfi.amp_model import AmpFCN
from ml_rfi.amp_phs_model import AmpPhsFCN
import time

def visualize_output(uvd, num_vis, show_folds = False, in_data = False, font_size = 10, fig_length = 12, fig_height = 360):
        """
        Method for visualizing the output data (with or without the input 
                                                data)
        Input: The PyUVData object with the data and the corresponding flags
               The number of waterfalls to show
               (Optional: whether to show the folds of the data)
               (Optional: whether to show the output data next to the input)
               (Optional: the font and figure dimensions)
               
               
        """
        
        keys = uvd.get_antpairpols()
        plt.rcParams.update({'font.size': font_size})
        plt.figure(figsize=(fig_length, fig_height))
        
        if in_data:
            
            counter0 = 0
            counter1 = 0
            counter2 = 0
            
            for i in range(3*num_vis):
                if (i % 3 == 0):
                    plt.subplot(num_vis, 3, i+1)
                    plt.imshow(np.log10(np.abs(uvd.get_data(keys[counter0], force_copy = True))), aspect="auto")
                    if show_folds:
                        a = [plt.plot([(i+1)*64, (i+1)*64], [0, 60], 'k-', lw=0.7, linestyle = '--') for i in range(16)]
                    plt.xlabel(keys[counter0])
                    plt.colorbar()
                    counter0 += 1
                    
                elif (i % 3 == 1):
                    plt.subplot(num_vis, 3, i+1)
                    plt.imshow(np.angle(uvd.get_data(keys[counter1], force_copy = True)), aspect="auto")
                    if show_folds:
                        a = [plt.plot([(i+1)*64, (i+1)*64], [0, 60], 'k-', lw=0.7, linestyle = '--') for i in range(16)]
                    plt.xlabel(keys[counter1])
                    plt.colorbar()
                    counter1 += 1
                    
                else:
                    plt.subplot(num_vis, 3, i+1)
                    plt.imshow(np.log10(np.abs(uvd.get_data(keys[counter2], force_copy = True))*np.logical_not(uvd.get_flags(keys[counter2], force_copy = True))), aspect='auto',vmin=-4,vmax=0.)
                    if show_folds:
                        a = [plt.plot([(i+1)*64, (i+1)*64], [0, 60], 'k-', lw=0.7, linestyle = '--') for i in range(16)]
                    plt.xlabel(keys[counter2])
                    plt.colorbar()
                    counter2 += 1
                
            
        else:
            for j in range(num_vis):
                plt.subplot(int(num_vis/2), 2, j+1)
                plt.imshow(np.log10(np.abs(uvd.get_data(keys[j], force_copy = True)) *np.logical_not(uvd.get_flags(keys[j], force_copy = True))), aspect='auto',vmin=-4,vmax=0.)
                plt.xlabel(keys[j])
                plt.colorbar()
                
                
def visualize_input(uvd, num_vis, show_folds = False, font_size = 10, fig_length = 12, fig_height = 360):
    """
    Method for making a graph for the input data
    Input: The PyUVData object with the data and the corresponding data
               The number of waterfalls to show
               (Optional: whether to show the folds of the data)
               (Optional: the font and figure dimensions)
    """
    keys = uvd.get_antpairpols()
    plt.rcParams.update({'font.size': font_size})
    plt.figure(figsize=(fig_length, fig_height))
    in_counter = 0
    out_counter = 0
    for i in range(2*num_vis):
        if (i % 2 == 0):
            plt.subplot(num_vis, 2, i+1)
            plt.imshow(np.log10(np.abs(uvd.get_data(keys[in_counter]))), aspect='auto')
            if show_folds:
                a = [plt.plot([(i+1)*64, (i+1)*64], [0, 60], 'k-', lw=0.7, linestyle = '--') for i in range(16)]
            plt.xlabel(keys[in_counter])
            plt.colorbar()
            in_counter += 1

        else:
            plt.subplot(num_vis, 2, i+1)
            plt.imshow(np.angle(uvd.get_data(keys[out_counter])), aspect='auto')
            if show_folds:
                a = [plt.plot([(i+1)*64, (i+1)*64], [0, 60], 'k-', lw=0.7, linestyle = '--') for i in range(16)]
            plt.xlabel(keys[out_counter])
            plt.colorbar()
            out_counter += 1


class Predictor:
    """
    Class for making a prediction given a data filename and a CNN model
    """
    
    def __init__(self, uvd, filename, CNN_model, batch_size = 40, ch_input = 2,
                 pad_size = 16, f_factor = 16,
                 chtypes = 'AmpPhs'):
        self.uvd = uvd
        self.batch_size = batch_size
        self.filename = filename
        self.ch_input = ch_input
        self.CNN_model = CNN_model
        self.pad_size = pad_size
        self.f_factor = f_factor
        self.chtypes = chtypes
        

    def check_antennas(self):
        """
        Method for checking the available antennas
        """
        self.uvd.read_uvh5(self.filename, read_data = False)
        print("Antenna Options: ")
        print(np.unique(self.uvd.ant_1_array))
    
    #@profile
    def make_prediction(self, num_batches, save_flags = False):
        """
        Main method for performing the CNN prediction
        Input (Optional): The number of antennas to pick, where 0 corresponds to all
        Output (optional): A UVFlag object 
        """
        # read in the data
        print("reading {}".format(self.filename))
        self.uvd.read_uvh5(self.filename)
        
        # create variables for the input, output, and kernel
        vis_input = tf.placeholder(tf.float32, shape=[None, None, None, self.ch_input])
        mode_bn = tf.placeholder(tf.bool)
        d_out = tf.placeholder(tf.float32)
        kernel_size = tf.placeholder(tf.int32)
        
        RFI_guess = AmpPhsFCN(vis_input, mode_bn=mode_bn, d_out=d_out, reuse=tf.AUTO_REUSE)
        init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        savr = tf.train.Saver()

        if save_flags:
            # make UVFlag object
            uvf = UVFlag(self.uvd)

        with tf.Session() as sess:
            sess.run(init)
            savr.restore(sess, self.CNN_model)

            # iterate over waterfalls in UVData object
            idx = 0
            batch = np.zeros((self.batch_size, self.uvd.Ntimes, self.uvd.Nfreqs),
                             dtype=self.uvd.data_array.dtype)
            keys = []
            batch_id = 0
            print("starting prediction...")
            t0 = time.time()
            for key, data in self.uvd.antpairpol_iter():
                if batch_id < num_batches:
                    if idx < self.batch_size:
                            # add another waterfall to the batch
                            batch[idx, :, :] = data
                            keys.append(key)
                            idx += 1
                    else:
                            print("batch {:d}".format(batch_id))
                            batch_id += 1
                            # actually make prediction
                            batches = np.array(list(map(hf.fold,                                                        batch, self.batch_size * [self.f_factor],                                                       self.batch_size * [self.pad_size])))[:, :, :, :, :2].reshape(                                                                -1, 2 * (self.pad_size + 2) + 60,                                                               int(2 * self.pad_size + 1024 / self.f_factor),                                                              2)
                            g_s = sess.run(RFI_guess, feed_dict={vis_input: batches, mode_bn: True})

#                             guess = np.reshape(tf.argmax(g_s, axis=-1).eval(),                                                                       (-1, int(self.f_factor),                                                                        int(2 * (self.pad_size + 2) + 60),                                                                        int(2 * self.pad_size + 1024 / self.f_factor)))
                            
#                             sh = np.shape(guess)
#                             temp = guess[:, :, (self.pad_size+2) : (sh[2] - (self.pad_size+2)), self.pad_size : (sh[3] - self.pad_size)]
#                             _, ch_fold, ntimes, dfreqs = np.shape(temp)
#                             pred_unfold = np.lib.stride_tricks.as_strided(temp, (self.batch_size, ntimes, ch_fold*dfreqs), (self.batch_size*ch_fold*dfreqs*8, ch_fold*dfreqs*8, 8))
                            
                            # unfold data
                            pred_unfold = np.array(list(map(hf.unfoldl,                                                           np.reshape(tf.argmax(g_s, axis=-1).eval(),                                                                       (-1, int(self.f_factor),                                                                        int(2 * (self.pad_size + 2) + 60),                                                                        int(2 * self.pad_size + 1024 / self.f_factor))),                                                            self.batch_size*[self.f_factor],self.batch_size*[self.pad_size])))
                            # store flags back in flag_array
                            for j, key in enumerate(keys):
                                    blt1, blt2, pol = self.uvd._key2inds(key)
                                    assert len(blt2) == 0
                                    assert len(blt1) == pred_unfold.shape[1]
                                    assert pred_unfold.shape[2] == self.uvd.Nfreqs
                                    if save_flags:
                                            uvf.flag_array[blt1, 0, :, pol[0]] = pred_unfold[j, :, :]
                                    else:
                                            self.uvd.flag_array[blt1, 0, :, pol[0]] = pred_unfold[j, :, :]

                            # reinitialize
                            idx = 0
                            batch = np.zeros((self.batch_size, self.uvd.Ntimes, self.uvd.Nfreqs),
                                             dtype=self.uvd.data_array.dtype)
                            keys = []
                else:
                    break
            if idx > 0:
                    # make one final prediction
                    batch = batch[:idx, :, :]
                    batches = np.array(list(map(hf.fold,
                                       batch,
                                       self.batch_size * [self.f_factor],
                                       self.batch_size * [self.pad_size])))[:, :, :, :, :2].reshape(
                                               -1, 2 * (self.pad_size + 2) + 60,
                                               int(2 * self.pad_size + 1024 / self.f_factor), 2)
                    g_s = sess.run(RFI_guess, feed_dict={vis_input: batches, mode_bn: True})

                    # unfold data
                    pred_unfold = np.array(list(map(hf.unfoldl,
                                           tf.reshape(tf.argmax(g_s, axis=-1),
                                                      [-1, int(self.f_factor),
                                                       int(2 * (self.pad_size + 2) + 60),
                                                       int(2 * self.pad_size + 1024 / self.f_factor)]).eval(),
                                           self.batch_size*[self.f_factor],self.batch_size*[self.pad_size])))
                    # store flags back in flag_array
                    for j, key in enumerate(keys):
                            blt1, blt2, pol = self.uvd._key2inds(key)
                            assert len(blt2) == 0
                            assert len(blt1) == pred_unfold.shape[1]
                            assert pred_unfold.shape[2] == self.uvd.Nfreqs
                            if save_flags:
                                    uvf.flag_array[blt1, 0, :, pol[0]] = pred_unfold[j, :, :]
                            else:
                                    self.uvd.flag_array[blt1, 0, :, pol[0]] = pred_unfold[j, :, :]

        #Optionally return a UVFlag object
        if save_flags:
            return uvf

        t1 = time.time()
        print("Prediction Complete")
        print("elapsed time: ", t1 - t0)
