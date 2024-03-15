import argparse
import torch
import torch.nn as nn
import unidecode
import string
import time

from utils import (
    load_dataset,
    char_tensor,
    random_training_set,
    time_since,
    random_chunk,
    CHUNK_LEN,
)

# from models.lstm import LSTM
from models.ffnn import FFNN

import numpy as np
import json
import torch.nn as nn

# from tensorflow import keras
from keras.preprocessing.text import Tokenizer

# from keras.layers import TextVectorization
from keras.utils import pad_sequences
from keras.initializers import Constant
from keras.optimizers import Adam
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split


EMBEDDING_PATH = './glove.twitter.27B/glove.twitter.27B.200d.txt'
TRAIN_DATA_PATH = './prep_train.json'
TEST_DATA_PATH = './prep_test.json'


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
    embedding_matrix = np.zeros((num_words, 200))

    for word, i in word_index.items():
        if i < num_words:
            emb_vec = embedding_dict.get(word)
            if emb_vec is not None:
                embedding_matrix[i] = emb_vec

    if args.train_ffnn:

        # Create model
        model = FFNN.create_model(num_words, max_len, embedding_matrix)

        # Train model
        history = FFNN.train_model(
            model,
            train_padded_data,
            train_labels,
            test_padded_data,
            test_labels,
        )
        f1score, report = FFNN.evaluate_model(
            model, test_padded_data, test_labels
        )

        print("F1-score:", f1score)
        print("Classification Report:")
        print(report)

    if args.lstm_train:
        n_epochs = 3000
        print_every = 100
        plot_every = 10
        hidden_size = 128
        n_layers = 2

        lr = 0.005
        # decoder = LSTM(num_words, hidden_size, num_words, n_layers)
        # decoder_optimizer = torch.optim.Adam(decoder.parameters(), lr=lr)

        start = time.time()
        all_losses = []
        loss_avg = 0

        # for epoch in range(1, n_epochs + 1):
        #     loss = train(decoder, decoder_optimizer, *random_training_set())
        #     loss_avg += loss

        #     if epoch % print_every == 0:
        #         print(
        #             '[{} ({} {}%) {:.4f}]'.format(
        #                 time_since(start), epoch, epoch / n_epochs * 100, loss
        #             )
        #         )
        #         print(generate(decoder, 'A', 100), '\n')

        #     if epoch % plot_every == 0:
        #         all_losses.append(loss_avg / plot_every)
        #         loss_avg = 0


if __name__ == "__main__":
    main()

