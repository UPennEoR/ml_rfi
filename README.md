# ml_rfi
A Deep Fully Convolutional Neural Net implementation in Tensorflow as applied to RFI flagging in waterfall visibilities.

# Training Datasets

## Simulated Data

*SimVisRFI_15_120_v3.h5* - Contains 1000 simulated waterfall visibilities over baseline lengths of 15m to 120m. RFI includes
                         stations and more spurious events (timescales >= 30).
                         
*SimVisRFI_v2.h5* - Contains 1000 simulated waterfall visibilities for one baseline type. RFI includes
                  stations and speckled RFI.
                  
*SimVisRFI_15_120_NoStations.h5* - Contains 300 simulated waterfall visibilities over baseline lengths of 15m to 120m. No RFI
                                 stations (e.g. ORBCOM), all RFI is randomly placed across all times & frequencies.
                                 Addittionaly includes randomized burst events where RFI is placed at all frequencies for a
                                 time sample.
                                 
## Real Data
Ground truth is NOT 100% certain for this dataset. It's based entirely on what XRFI perceives as RFI and is prone to
it's biases.

*RealVisRFI_v3.h5* - Contains 3584 real HERA (37-ish) waterfall visibilities all flagged using XRFI. !!! The majority of 
                   visibilities up to 900 look decent enough to train/evaluate on, however everything after this looks like 
                   it's been very poorly flagged by XRFI, i.e. missing ORBCOM !!!


# Training - Evaluation Strategies

Ideally we want to train entirely on simulated data as we know with 100% certainty what the ground truth is, but 
there may be the issue that our simulated data doesn't quite accurately reflect our real data (e.g. passband, RFI statistics)

### Current strategy:

Training dataset: 

*SimVisRFI_15_120_v3.h5*

Evaluation dataset:

*RealVisRFI_v3.h5*


# Accessing HPCs for Tensorboard
## Bridges
## Intrepid
