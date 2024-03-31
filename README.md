# code_weavers_sarcasm_detection

### To run docker container:

To build and run all four models in a docker conatiner:

    docker-compose up --build

Same command but with `DEBUG` enabled:

    docker-compose --log-level DEBUG up --build

To run individual models set the following environment variables to 1 like so:


    # model options: FFNN, LSTM, LSTM_ATTENTION, SVM
    LSTM=1 FFNN=1 docker-compose up --build



You can also set `SAVE_WORD_INDEX` to True if you would like the word indexed saved to a json file like so:


You can also set `SAVE_WORD_INDEX` to 1 if you would like the word indexed saved to a json file like so:

    SAVE_WORD_INDEX=1 FFNN=1 docker run up


### To run the models loacally (ideallpy with Python 3.11):

To install requirements:

    pip install -r requirements.txt

To run data preprocessing:

    python prep.py --dataset train
    python prep.py --dataset test

To run the individual training scripts:

    python train.py --train ffnn
    python train.py --train svm
    python train.py --train lstm
    python train.py --train lstm_attention

To have the word index saved to a json file, use `--save_word_index` like the following:

    python train.py --save_word_index

Arguments can be combined to run training on multiple models like the following:

    # Runs training on FFNN and LSTM and saves the word index
    python train.py --train_ffnn --train_lstm --save_word_index
    # Runs training on all four models
    python train.py --train_ffnn --train_svm --train_lstm --train_lstm_attention
