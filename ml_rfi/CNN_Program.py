import matplotlib.pyplot as plt
import sys
import numpy as np
import tensorflow as tf
from pyuvdata import UVData
from ml_rfi import helper_functions as hf
from ml_rfi.amp_model import AmpFCN
from ml_rfi.amp_phs_model import AmpPhsFCN
from time import time


def visualize_output(uvd, in_data = False):
        """
        Method for visualizing the output data (with or without the input 
                                                data)
        Input: The PyUVData object with the data and the corresponding flags
               
        """
        
        iterator = uvd.antpairpol_iter()
        wfs = []
        for x in iterator:
            wfs.append(x)
        
        wfs = np.asarray(wfs)
        
        
        if (in_data):
            plt.rcParams.update({'font.size': 36})
            plt.figure(figsize=(6*len(wfs), 19*len(wfs)))
            in_counter = 0
            out_counter = 0
            
            for i in range(2*len(wfs)):
                if (i % 2 == 0):
                    plt.subplot(len(wfs), 2, i+1)
                    plt.imshow(np.log10(np.abs(wfs[in_counter, 1])), aspect="auto")
                    plt.xlabel(wfs[in_counter, 0])
                    plt.colorbar()
                    in_counter += 1
                    
                else:
                    plt.subplot(len(wfs), 2, i+1)
                    plt.imshow(np.log10(np.abs(wfs[out_counter, 1]) *np.logical_not(uvd.get_flags(wfs[out_counter, 0], force_copy = True))), aspect='auto',vmin=-4,vmax=0.)
                    plt.xlabel(wfs[out_counter, 0])
                    plt.colorbar()
                    out_counter += 1
                
            
        else:
            plt.rcParams.update({'font.size': 32})
            plt.figure(figsize=(3*len(wfs), 6*len(wfs)))
            iterator = uvd.antpairpol_iter()
            counter = 0
            for key, data in iterator:
                plt.subplot(len(wfs)/2, 2, counter+1)
                plt.imshow(np.log10(np.abs(data) *np.logical_not(uvd.get_flags(key, force_copy = True))), aspect='auto',vmin=-4,vmax=0.)
                plt.xlabel(key)
                plt.colorbar()
                counter += 1
                
                
                
def visualize_input(uvd):
    """
    Method for making a graph for the input data
    Input: The PyUVData object with the input data
    """
    num_of_ants = 0
    iterator = uvd.antpairpol_iter()
    for x in iterator:
        num_of_ants += 1
        
    plt.rcParams.update({'font.size': 30})
    counter = 0
    plt.figure(figsize=(3*num_of_ants, 6*num_of_ants))
    iterator = uvd.antpairpol_iter()
    for key, data in iterator:
        plt.subplot(num_of_ants/2, 2, counter+1)
        plt.imshow(np.log10(np.abs(data)), aspect="auto")
        plt.colorbar()
        plt.xlabel(key)
        counter += 1


class Predictor:
    """
    Class for making a prediction given a data filename and a CNN model
    """
    
    def __init__(self, uvd, filename, CNN_model, ch_input = 2,
                 pad_size = 16, f_factor = 16,
                 chtypes = 'AmpPhs'):
        self.uvd = uvd
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

    def pick_antennas(self):
        """
        Method for loading the data in a PyUVData object
        """
        self.uvd.read_uvh5(self.filename, read_data = False)
        self.uvd.read_uvh5(self.filename, np.unique(self.uvd.ant_1_array))
        iterator = self.uvd.antpairpol_iter()
        wfs = []
        counter = 0
        for x in iterator:
            wfs.append(x)
            counter += 1
        
        self.numants = counter
        return np.asarray(wfs)

    def make_prediction(self):
        """
        Main method for performing the CNN prediction
        """
        wfs = self.pick_antennas()
        
        # make a list to contain the visibilities for each of the 4 
        # polarizations
        prev_pred0 = []
        prev_pred1 = []
        prev_pred2 = []
        prev_pred3 = []
        
        # create variables for the input, output, and kernel
        vis_input = tf.placeholder(tf.float32, shape=[None, None, None, self.ch_input])
        mode_bn = tf.placeholder(tf.bool)
        d_out = tf.placeholder(tf.float32)
        kernel_size = tf.placeholder(tf.int32)
        
        RFI_guess = AmpPhsFCN(vis_input, mode_bn=mode_bn, d_out=d_out, reuse=tf.AUTO_REUSE)
        init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        savr = tf.train.Saver()

        with tf.Session() as sess:
            sess.run(init)
            savr.restore(sess, self.CNN_model)
            for i in range(len(wfs)):
                sh = wfs[i][1].shape
                temp = wfs[i][1].reshape(1,sh[0],sh[1])
                
                #fold the input data
                batch_x = (np.array(hf.fold(temp,self.f_factor, self.pad_size))[:,:,:,:2]).reshape(-1,2*(self.pad_size+2)+60,int(2*self.pad_size+1024/self.f_factor),2)
                time0 = time()
                ind = 0
                ct = 0
                pred_start = time()
                
                #perform the actual prediction
                g = sess.run(RFI_guess, feed_dict={vis_input: batch_x, mode_bn: True})
                #unfold the prediction data
                pred_unfold = hf.store_iterator(
                    map(hf.unfoldl,tf.reshape(tf.argmax(g,axis=-1),
                    [-1,int(self.f_factor),int(2*(self.pad_size+2)+60),
                    int(2*self.pad_size+1024/self.f_factor)]).eval(),[self.f_factor],[self.pad_size]))

                pred_time = time() - pred_start
                if self.chtypes == 'AmpPhs':
                    thresh = 0.62
                else:
                    thresh = 0.385
                
                #store the prediction data
                y_pred = np.array(pred_unfold[0]).reshape(-1,1024)
                
                # categorize the predictions in each polarization
                if i%4 == 0:
                    if len(prev_pred0) == 0:
                        prev_pred0 = y_pred
                    else:
                        prev_pred0 = np.concatenate((prev_pred0, y_pred), axis = 0)
                
                elif i%4 == 1:
                    if len(prev_pred1) == 0:
                        prev_pred1 = y_pred
                    else:
                        prev_pred1 = np.concatenate((prev_pred1, y_pred), axis = 0)
                        
                elif i%4 == 2:
                    if len(prev_pred2) == 0:
                        prev_pred2 = y_pred
                    else:
                        prev_pred2 = np.concatenate((prev_pred2, y_pred), axis = 0)
                    
                elif i%4 == 3:
                    if len(prev_pred3) == 0:
                        prev_pred3 = y_pred
                    else:
                        prev_pred3 = np.concatenate((prev_pred3, y_pred), axis = 0)
        
        # update the flags of the PyUVData object input
        for n in range(4):
            if n == 0:
                for j in range(1024):
                    self.uvd.flag_array[:, 0, j, n] = prev_pred0[:, j]
                    
            elif n == 1:
                for j in range(1024):
                    self.uvd.flag_array[:, 0, j, n] = prev_pred1[:, j]
                    
            elif n == 2:
                for j in range(1024):
                    self.uvd.flag_array[:, 0, j, n] = prev_pred2[:, j]
                    
            elif n == 3:
                for j in range(1024):
                    self.uvd.flag_array[:, 0, j, n] = prev_pred3[:, j]
        
        print("Complete")