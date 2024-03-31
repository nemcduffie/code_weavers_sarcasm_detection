# **Sarcasm Detection**

This repository tackles the task from the [iSarcasmEval](https://sites.google.com/view/semeval2022-isarcasmeval#h.t53li2ejhrh8) shared task, Task 6, Subtask A, at SemEval 2022.
Subtask A consists of determining whether a given text is sarcastic or non-sarcastic. The selected language in this project is English. 

### Quickstart

    docker-compose up --build


### Approach:

To identify sarcasm in tweets, four different models (FFNN, LSTM, LSTM with Attention, and SVM) are developed. The effectiveness of each model is compared.

### Metrics:

To evaluate and compare the performance of the models, F1-score is used. sklearn.metrics is used to calculate precision, recall, and F1-score.

### Dataset:

The dataset used is the one created for SemEval 2022 Task 6. The overview paper [SemEval-2022 Task 6: iSarcasmEval, Intended Sarcasm Detection in English and Arabic (Abu Farha et al., SemEval 2022)](https://aclanthology.org/2022.semeval-1.111/) contains detailed information.

### Data preprocessing:

Emojis and other common elements present in tweets, such as abbreviations or hashtags, have to be addressed as part of the preprocessing of the data.

### Emojis:

Emojis are mapped into text for further embedding into vectors using a list of Unicode emoji characters and sequences. More information about the emoji dataset used can be found [here](https://www.unicode.org/emoji/charts-14.0/full-emoji-list.html).

### Abbreviations

Abbreviations are converted into full words using a dictionary compiled from common abbreviations. 

### Word embedding:

[GloVe: Global Vectors for Word Representation (Pennington et al., EMNLP 2014)](https://aclanthology.org/D14-1162/) was used to embed words into vectors. 
The pre-trained word vector files can be downloaded [here](https://nlp.stanford.edu/projects/glove/).
In this project, the *glove.twitter.27B.200d.txt* file with 200-dimension vectors was used.  

------------------------------------------------------------
------------------------------------------------------------
------------------------------------------------------------ 

# To run docker container:

To build and run all four models in a docker conatiner:

    docker-compose up --build

Same command but with `DEBUG` enabled:

    docker-compose --log-level DEBUG up --build

To run individual models set the following environment variables to 1 like so:

    # model options: FFNN, LSTM, LSTM_ATTENTION, SVM
    LSTM=1 FFNN=1 docker-compose up --build

To include optional extra dataset (adaptated from test data [task_C_En_test.csv](data/test/task_C_En_test.csv) intended for another task) in training data: 

    TRAIN_EXTRA=1 docker-compose up --build

You can also set `SAVE_WORD_INDEX` to 1 if you would like the word indexed saved to a json file like so:

    SAVE_WORD_INDEX=1 FFNN=1 docker run up

## To run the models loacally (ideallpy with Python 3.11):

To install requirements:

    pip install -r requirements.txt

To run data preprocessing:

    python prep.py --dataset train
    python prep.py --dataset test

To include optional extra dataset (adaptated from test data [task_C_En_test.csv](data/test/task_C_En_test.csv) intended for another task) in training data: 

    python prep.py --dataset train_extra

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
