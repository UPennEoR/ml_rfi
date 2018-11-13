# Arguments order Tdset_vers FCN_vers batch size ch_input mode hybrid chtypes dropouts ksize steps datatype
# list of anyalyses to run

analysis='tversion' # runs model training/eval for Amp & Amp-Phs

if [ $analysis = "tversion" ];then
    d_version='v13' # training data version
    n_version0='v7' # neural net version
    n_version1='v9'
    echo "Running amplitude-phase training, evaluation, and analysis."
#    python runDFCN.py $d_version $n_version1 32 2 'traineval' '' 'AmpPhs' 0.8 3 500000 ''
    python runDFCN.py $d_version $n_version1 32 2 '' True 'AmpPhs' 0.8 3 100 'uv'

    echo "Running amplitude training, evaluation, and analysis."
#    python runDFCN.py $d_version $n_version0 64 1 'traineval' '' 'Amp' 0.8 3 800000 '' 
#    python runDFCN.py $d_version $n_version0 64 1 '' True 'Amp' 0.8 3 100 'uv'
fi
