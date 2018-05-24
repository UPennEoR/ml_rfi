import numpy as np
import tensorflow as tf
import h5py
import matplotlib
matplotlib.use("AGG")
import matplotlib.pyplot as plt
from pyuvdata import UVData
from fcn import cnn, preprocess
from time import time
from hera_qm import xrfi

plot = True

def load_data(uv):
    bsls = uv.get_antpairs()
    num_vis = len(bsls)
    dset = np.zeros((num_vis,60,1024),dtype=complex)
    ct = 0
    for b in bsls:
        dset[ct,:,:] = uv.get_data(b)
        ct+=1
    return dset

def main(args):
    uv = UVData()
    uv.read_miriad('zen.2457555.40356.xx.HH.uvcT')
    bsls = uv.get_antpairs()
    data_test = load_data(uv)#uv.get_data(bsls[24]).reshape(1,60,1024)
    real_data = preprocess(data_test).astype(np.float32)
    real_labels = xrfi.xrfi(np.abs(uv.get_data(bsls[24])))#uv.get_flags(bsls[24])
    print np.shape(real_data)
    test_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x":real_data},
        shuffle=False
    )

    rfiCNN = tf.estimator.Estimator(model_fn=cnn,model_dir='./checkpoint/')

    time_start = time()
    rfiPredict = rfiCNN.predict(input_fn=test_input_fn)
    #print 'It took '+str(time() - time_start)+' seconds to predict.'

    for i,predicts in enumerate(rfiPredict):
        print i
        real_labels = xrfi.xrfi(np.abs(uv.get_data(bsls[i])))
        if plot:
            plt.subplot(411)
            plt.imshow(predicts['classes'].reshape(-1,1024),aspect='auto')
            plt.title('Predicted Flags')
            plt.colorbar()

            plt.subplot(412)
            plt.imshow(real_labels.reshape(-1,1024),aspect='auto')
            plt.title('XRFI Flags')
            plt.colorbar()

            plt.subplot(413)
            plt.imshow(real_data[i,:,:,0],aspect='auto')
            plt.colorbar()
            plt.title('Vis. Log Normalized Amp.')

            plt.subplot(414)
            plt.imshow(real_data[i,:,:,1],aspect='auto')
            plt.colorbar()
            plt.title('Vis. Phs.')
        
            plt.savefig('test_'+str(i)+'.png')
            plt.clf()
if __name__ == "__main__":
    tf.app.run()
