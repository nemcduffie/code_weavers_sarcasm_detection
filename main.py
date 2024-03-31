import os
import tensorflow as tf

from prep import main as main_prep
from train import main as main_train


SAVE_WORD_INDEX = os.getenv('SAVE_WORD_INDEX', '0') == '1'
MODELS = []
for model in ['ffnn', 'lstm', 'lstm_attention', 'svm']:
    if os.getenv(model.upper(), '0') == '1':
        MODELS.append(model)
TRAIN_EXTRA = os.getenv('TRAIN_EXTRA', '0') == '1'


def main():
    print('\nPreparing test data...')
    main_prep('test')

    print('\nPreparing train data...')
    main_prep('train_extra' if TRAIN_EXTRA else 'train')

    print(f'\nTraining and testing {MODELS or "all"} models\n')
    main_train(MODELS or None, SAVE_WORD_INDEX)


if __name__ == '__main__':
    main()
