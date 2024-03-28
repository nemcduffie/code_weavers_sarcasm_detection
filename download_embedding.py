import numpy as np
import tensorflow as tf
import chakin
from prep import main as main_prep
from train import main as main_train

import os


DATA_DIR = './data'

CHAKIN_INDEX = 17
NUMBER_OF_DIMENSIONS = 200
SUBDIR_NAME = 'glove.twitter.27B'

ZIP_FILE = os.path.join(DATA_DIR, f'{SUBDIR_NAME}.zip')
UNZIP_DIR = os.path.join(DATA_DIR, SUBDIR_NAME)

if SUBDIR_NAME[-1] == 'd':
    GLOVE_FILENAME = os.path.join(UNZIP_DIR, f'{SUBDIR_NAME}.txt')
else:
    GLOVE_FILENAME = os.path.join(
        UNZIP_DIR, f'{SUBDIR_NAME}.{NUMBER_OF_DIMENSIONS}d.txt'
    )

if not os.path.exists(ZIP_FILE) and not os.path.exists(UNZIP_DIR):
    # GloVe by Stanford is licensed Apache 2.0:
    #     https://github.com/stanfordnlp/GloVe/blob/master/LICENSE
    #     http://nlp.stanford.edu/data/glove.twitter.27B.zip
    #     Copyright 2014 The Board of Trustees of The Leland Stanford Junior University
    print(f'Downloading embeddings to {ZIP_FILE}')
    chakin.download(number=CHAKIN_INDEX, save_dir=DATA_DIR)
else:
    print('Embeddings already downloaded.')

if not os.path.exists(UNZIP_DIR):
    import zipfile

    with zipfile.ZipFile(ZIP_FILE, 'r') as zip_ref:
        print(f'Extracting embeddings to {UNZIP_DIR}')
        zip_ref.extractall(UNZIP_DIR)

    if os.path.exists(GLOVE_FILENAME):
        for file in [
            'glove.twitter.27B.25d.txt',
            'glove.twitter.27B.50d.txt',
            'glove.twitter.27B.100d.txt',
        ]:
            os.remove(os.path.join(UNZIP_DIR, file))
        os.remove(ZIP_FILE)
else:
    print('Embeddings already extracted.')

main_prep('train')
main_prep('test')
main_train()

