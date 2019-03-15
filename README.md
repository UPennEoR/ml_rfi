# ml_rfi
A Deep Fully Convolutional Neural Net implementation in Tensorflow as applied to RFI flagging in waterfall visibilities.

#Dependencies:
*`numpy`
*`scipy`
*`h5py`
*`keras`
*`tensorflow`
*`sklearn`
*`pyuvdata`

**run_analysis.sh** - Bash script for running runDFCN.py

**runDFCN.py** - Neural net training and evaluation script.

**pipelineDFCN.py** - Script for running strictly for prediction, mostly on pyuvdata datasets.

**AmpMode.py** - Tensorflow model for amplitude input DFCN.

**AmpPhsMode.py** - Tensorflow model for amplitude-phase input DFCN.

**helper_functions.py** - Functions used for loading datasets and defining tensorflow objects used in the models.

# Training Datasets

## Simulated Data

*SimVisRFI_15_120_v3.h5* - Contains 1000 simulated waterfall visibilities over baseline lengths of 15m to 120m. RFI includes
                         stations and more random events (timescales >= 30). Deprecated!!!
                         
*SimVisRFI_v2.h5* - Contains 1000 simulated waterfall visibilities for one baseline type. RFI includes
                  stations and speckled RFI. Deprecated!!!
                  
*SimVisRFI_15_120_NoStations.h5* - Contains 300 simulated waterfall visibilities over baseline lengths of 15m to 120m. No RFI
                                 stations (e.g. ORBCOM), all RFI is randomly placed across all times & frequencies.
                                 Addittionaly includes randomized burst events where RFI is placed at all frequencies for a
                                 time sample. Deprecated!!!

*SimVis_3000_v13.h5* - Amalgamation of several previous simulated training datasets, and the most recently used for training.
                                 
## Real Data
Ground truth is NOT 100% certain for this dataset. It's based entirely on what XRFI perceives as RFI and is prone to
it's biases.

*RealVisRFI_v3.h5* - Contains 3584 real HERA (37-ish) waterfall visibilities all flagged using XRFI. !!! The majority of 
                   visibilities up to 900 look decent enough to train/evaluate on, however everything after this looks like 
                   it's been very poorly flagged by XRFI, i.e. missing ORBCOM !!!

*IDR21TrainingData_Raw_vX.h5* - Small hand flagged dataset which should have a very well identified ground truth.


# Training - Evaluation Strategies

Training is performed using entirely simulated data from HERAsim across a wide range of baseline types, mock sky scenarios, and
RFI instances.

Evaluation is performed in two ways:
1. Evaluate on 30% of unused training data to check for overfitting.
2. Evaluate on real data. (e.g. IDR21TrainingData_Raw_vX.h5)

### Current strategy:

Training dataset: 

*SimVis_3000_v13.h5*

Evaluation dataset:

*IDR21TrainingData_Raw_vX.h5*
