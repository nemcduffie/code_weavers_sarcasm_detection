import numpy as np
import tensorflow as tf
import chakin
import os

from prep import main as main_prep
from train import main as main_train


SAVE_WORD_INDEX = os.getenv('SAVE_WORD_INDEX', '0') == '1'
MODELS = []
for model in ['ffnn', 'lstm', 'lstm_attention', 'svm']:
    if os.getenv(model.upper(), '0') == '1':
        MODELS.append(model)


def main():
    main_prep('test')
    main_prep('train_extra' if TRAIN_EXTRA else 'train')
    main_train(MODELS, SAVE_WORD_INDEX)


if __name__ == '__main__':
    main()
