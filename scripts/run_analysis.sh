# Arguments order Tdset_vers FCN_vers batch size ch_input mode hybrid chtypes dropouts ksize steps datatype
# list of anyalyses to run

analysis='tversion' # runs model training/eval for Amp & Amp-Phs

if [ $analysis = "tversion" ];then
    d_version='v9' # training data version
    n_version0='v7' # Amplitude-Phase neural net version
    n_version1='v9' # Amplitude Only neural net version
    echo "Running amplitude-phase training, evaluation, and analysis."
#    python runDFCN.py $d_version $n_version1 100 2 'traineval' '' 'AmpPhs' 0.5 3 1000 ''
     python runDFCN.py $d_version $n_version1 100 2 '' False 'AmpPhs' 0.5 3 100 'uv'
#    python runDFCN.py $d_version $n_version1 32 2 '' False 'AmpPhs' 0.8 3 100 '' 

    echo "Running amplitude training, evaluation, and analysis."
#    python runDFCN.py $d_version $n_version0 512 1 'traineval' '' 'Amp' 0.8 3 500000 '' 
#    python runDFCN.py $d_version $n_version0 64 1 '' False 'Amp' 0.8 3 100 'uv'
#     python runDFCN.py $d_version $n_version0 64 1 '' False 'Amp' 0.8 3 100 ''
fi
