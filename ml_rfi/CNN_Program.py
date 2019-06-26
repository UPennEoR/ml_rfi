import matplotlib.pyplot as plt
import sys
import numpy as np
import tensorflow as tf
from pyuvdata import UVData
from ml_rfi import helper_functions as hf
from ml_rfi.amp_model import AmpFCN
from ml_rfi.amp_phs_model import AmpPhsFCN
from time import time

class Predictor:
    """
    Class for making a prediction given a data filename and a CNN model
    """
    
    def __init__(self, filename, CNN_model, ch_input = 2,
                 pad_size = 16, f_factor = 16,
                 chtypes = 'AmpPhs'):
        self.filename = filename
        self.ch_input = ch_input
        self.CNN_model = CNN_model
        self.pad_size = pad_size
        self.f_factor = f_factor
        self.chtypes = chtypes

        # Create a PyUVData object to store the input data
        self.uvd = UVData()

    def check_antennas(self):
        """
        Method for checking the available antennas
        """
        self.uvd.read_uvh5(self.filename, read_data = False)
        print("Antenna Options: ")
        print(np.unique(self.uvd.ant_1_array))

    def pick_antennas(self, antenna_array = [], pick_all = False):
        """
        Method for loading the data from the desired antennas
        Input: Array with the desired antenna numbers 
               (not neccessary if pick_all = True)
        """
        if (pick_all):
            self.uvd.read_uvh5(self.filename, read_data = False)
            self.selected_ants = np.unique(self.uvd.ant_1_array)
        else:
            self.selected_ants = antenna_array
            
        self.uvd.read_uvh5(self.filename, antenna_nums=self.selected_ants)

    def prepare_input(self, ants = [], all_antennas = False):
        """
        Method for processing the data before feeding them to the CNN
        Input: Array with the desired antenna numbers 
               (not neccessary if all_antennas = True)
        """
        if (all_antennas):
            ants = self.selected_ants
        
        ant1 = ants.tolist()
        ant2 = ants.tolist()
        num_comb = len(ants) * (len(ants) - 1)/2
        stored_data = []
        wf_data = np.zeros([int(num_comb),1 , 60, 1024])

        counter = 0
        for i in range(len(ants)):
            for j in range(len(ant2)):
                if (ant1[0] == ant2[j]):
                    continue
                
                if ((ant1[0], ant2[j]) in stored_data):
                    continue
                
                temp = self.uvd.get_data(ant1[0], ant2[j])[:, :, 0]
                stored_data.append((ant1[0], ant2[j]))
                stored_data.append((ant2[j], ant1[0]))
                
                wf_data[counter] = temp.reshape(1,temp.shape[0],temp.shape[1])
                counter += 1
            ant1.remove(ant1[0])

        return np.asarray(wf_data)
    
    def visualize_input(self, wfs):
        """
        Method for making a graph for the input data
        Input: The visibility data array
        """
        for i in range(len(wfs)):
            plt.subplot(len(wfs), 1, i+1)
            plt.imshow(np.log10(np.abs(wfs[i, 0, :, :])), aspect="auto")

    def make_prediction(self, wfs):
        """
        Main method for performing the CNN prediction
        Input: The visibility data array
        Output: The flag data array
        """
        output = []
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
                #fold the input data
                batch_x = (np.array(hf.fold(wfs[i, :, :, :],self.f_factor, self.pad_size))[:,:,:,:2]).reshape(-1,2*(self.pad_size+2)+60,int(2*self.pad_size+1024/self.f_factor),2)
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
                output.append(y_pred)
                
        return np.asarray(output)
        
    def visualize_output(self, flags, wfs = [], in_data = False):
        """
        Method for visualizing the output data (with or without the input data)
        Input: The predicted flag data
               The visibility data (not necessary if in_data = False)
               
        """
        if (in_data):
            in_counter = 0
            out_counter = 0
            for i in range(len(flags) * 2):
                plt.subplot(len(wfs), 2, i+1)
                if (i % 2 == 0):
                    plt.imshow(np.log10(np.abs(wfs[in_counter, 0, :, :])), 
                               aspect="auto")
                    in_counter += 1
                else:
                    plt.imshow(np.log10(np.abs(wfs[out_counter, 0, :, :]) 
                                        *np.logical_not(flags[out_counter])), 
                               aspect='auto',vmin=-4,vmax=0.)
                    plt.colorbar()
                    out_counter += 1
        else:
            for i in range(len(flags)):
                data = wfs[i, :, :, :]
                plt.subplot(len(wfs), 1, i+1)
                plt.imshow(np.log10(np.abs(data[0, :,:]) *np.logical_not(flags[i])), aspect='auto',vmin=-4,vmax=0.)
                plt.colorbar()