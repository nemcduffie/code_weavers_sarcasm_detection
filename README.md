# code_weavers_sarcasm_detection

### To run in docker:

To run all four models in a docker conatiner:

    docker-compos build && docker-compose up

Same commands but with DEBUG enabled:

    docker-compose --log-level DEBUG build && docker-compose --log-level DEBUG up

To run individual models set the following environment variables to True:


    # FFNN, LSTM, LSTM_ATTENTION, SVM

    # Run build if you haven't already done so
    docker-compos build
    # Run container with certain parameters set
    docker run -p 8000:8000 -e LSTM=True FFNN=True code_weavers_sarcasm_detection-app:latest python download_embeddings.py


You can also set `SAVE_WORD_INDEX` to True if you would like the word indexed saved to a json file like so:

    docker run -p 8000:8000 -e SAVE_WORD_INDEX=True FFNN=True code_weavers_sarcasm_detection-app:latest python download_embeddings.py


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
    python train.py --train lstm_attention

To have the word index saved to a json file, use `--save_word_index` like the following:

    python train.py --save_word_index

Arguments can be combined to run training on multiple models like the following:

    # Runs training on FFNN and LSTM and saves the word index
    python train.py --train_ffnn --train_lstm --save_word_index
    # Runs training on all four models
    python train.py --train_ffnn --train_svm --train_lstm --train_lstm_attention

