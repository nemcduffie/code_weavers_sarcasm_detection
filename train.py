import argparse
import torch
import torch.nn as nn
import unidecode
import string
import time
import numpy as np
import json

from tensorflow import keras
from keras.preprocessing.text import Tokenizer
from keras.utils import pad_sequences
from keras.initializers import Constant
from keras.optimizers import Adam
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from models.lstm import LSTM
from models.ffnn import FFNN


EMBEDDING_PATH = './glove.twitter.27B/glove.twitter.27B.200d.txt'
TRAIN_DATA_PATH = './prep_train.json'
TEST_DATA_PATH = './prep_test.json'
EMBEDDING_DIM = 200


# Maximun length function
def max_tweet_length(preprocessed_text):
    max_len = 0
    for tweet in preprocessed_text:
        tweet_len = len(tweet)
        if tweet_len > max_len:
            max_len = tweet_len
    return max_len


# Padding data function to have the same length in all inputs
def padding_data(sentences, max_len, tokenizer):
    sequences = tokenizer.texts_to_sequences(
        sentences
    )  # Convert tokens to integers sequences
    padded_data = pad_sequences(
        sequences, maxlen=max_len, truncating='post', padding='post'
    )  # Pad sequences
    return padded_data


# Saving word index
def save_word_index(tokenizer, filename):
    word_index = tokenizer.word_index
    with open(filename, 'w') as f:
        json.dump(word_index, f, indent=4)


def main():
    parser = argparse.ArgumentParser(description='Train Models')
    parser.add_argument(
        '--train_ffnn',
        dest='train_ffnn',
        help='Train FFNN model',
        action='store_true',
    )
    parser.add_argument(
        '--train_svm',
        dest='lstm_train',
        help='Train LSTM model',
        action='store_true',
    )
    parser.add_argument(
        '--train_lstm',
        dest='lstm_train',
        help='Train LSTM model',
        action='store_true',
    )
    parser.add_argument(
        '--train_lstm_attention',
        dest='train_lstm_attention',
        help='Train LSTM model with attention',
        action='store_true',
    )
    args = parser.parse_args()

    # Load train and test files
    with open(TRAIN_DATA_PATH, 'r', encoding='utf-8') as train_file:
        train_data = json.load(train_file)

    with open(TEST_DATA_PATH, 'r', encoding='utf-8') as test_file:
        test_data = json.load(test_file)

    # Extract data from files
    train_sentences = [
        entry['text_with_emojis'] for entry in train_data
    ]  # Tokenized and preprocessed
    train_labels = [entry['sarcastic'] for entry in train_data]

    test_sentences = [
        entry['text_with_emojis'] for entry in test_data
    ]  # Tokenized and preprocessed
    test_labels = [entry['sarcastic'] for entry in test_data]

    all_sentences = train_sentences + test_sentences

    # Convert labels to numpy arrays
    train_labels = np.array(train_labels)
    test_labels = np.array(test_labels)

    max_len = max_tweet_length(all_sentences)

    # Define the tokenizer
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(all_sentences)

    train_padded_data = padding_data(train_sentences, max_len, tokenizer)
    test_padded_data = padding_data(test_sentences, max_len, tokenizer)

    # save_word_index(tokenizer, 'tokenizer_word_index.json')

    # Create embedding dictionary as numpy array
    embedding_dict = {}
    with open(EMBEDDING_PATH, 'r', encoding='utf8') as f:
        for line in f:
            values = line.split()
            word = values[0]
            vectors = np.asarray(values[1:], dtype=float)
            embedding_dict[word] = vectors
    f.close()

    # Create embedding matrix
    word_index = tokenizer.word_index
    num_words = len(word_index) + 1
    embedding_matrix = np.zeros((num_words, EMBEDDING_DIM))

    for word, i in word_index.items():
        if i < num_words:
            emb_vec = embedding_dict.get(word)
            if emb_vec is not None:
                embedding_matrix[i] = emb_vec

    if args.train_ffnn:
        # Create model
        model = FFNN(num_words, max_len, EMBEDDING_DIM, embedding_matrix)

        # Train model
        history = model.train_model(
            train_padded_data,
            train_labels,
            test_padded_data,
            test_labels,
        )
        f1score, report = model.evaluate_model(test_padded_data, test_labels)

        print("F1-score:", f1score)
        print("Classification Report:")
        print(report)

    if args.lstm_train:
        model = LSTM(
            num_words,
            max_len,
            EMBEDDING_DIM,
            embedding_matrix,
            drop_rate=0.5,
            learning_rate=0.2,
        )

        # Train model
        history = model.train_model(
            train_padded_data,
            train_labels,
            test_padded_data,
            test_labels,
        )
        f1score, report = model.evaluate_model(test_padded_data, test_labels)

        print("F1-score:", f1score)
        print("Classification Report:")
        print(report)


if __name__ == "__main__":
    main()
