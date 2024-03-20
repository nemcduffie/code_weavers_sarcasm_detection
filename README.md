# code_weavers_sarcasm_detection

To install requirements:

    pip install -r requirements.txt

To run data preprocessing:

    python prep.py --dataset train
    python prep.py --dataset test

To run the individual training scripts:

    python train.py --train_ffnn
    python train.py --train_svm
    python train.py --train_lstm
    python train.py --train_lstm_attention

Arguments can be combined to run training on multiple models like the following:

    # Runs training on FFNN and LSTM
    python train.py --train_ffnn --train_lstm
    # Runs training on all four models
    python train.py --train_ffnn --train_svm --train_lstm --train_lstm_attention
