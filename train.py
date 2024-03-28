import argparse
import numpy as np
import json
import logging

from keras.preprocessing.text import Tokenizer
from keras.utils import pad_sequences
from models.lstm import LSTM, LSTMWithAttention
from models.ffnn import FFNN
from models.svm import SVM

logger = logging.getLogger()
logger.setLevel(logging.INFO)
logging.basicConfig(filename='train.log', level=logging.INFO)

EMBEDDING_PATH = './data/glove.twitter.27B/glove.twitter.27B.200d.txt'
TRAIN_DATA_PATH = './data/prep_train.json'
TEST_DATA_PATH = './data/prep_test.json'
EMBEDDING_DIM = 200
EPOCHS = 25
VERBOSE = 0


# Function to load and organise data
def load_data(path):
    data = []
    # Load train or test file
    with open(path, 'r', encoding='utf-8') as file:
        data = json.load(file)

    # Extract data from files
    sentences = [
        entry['text_with_emojis'] for entry in data
    ]  # Tokenized and preprocessed
    labels = [entry['sarcastic'] for entry in data]

    return sentences, labels


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
def save_word_index_file(tokenizer, filename):
    word_index = tokenizer.word_index
    with open(filename, 'w') as f:
        json.dump(word_index, f, indent=4)


# Function to generate embedding matrix
def get_embedding_matrix(tokenizer):

    # Create embedding dictionary containing numpy arrays
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

    return num_words, embedding_matrix


def train_and_evaluate_model(
    model,
    train_padded_data,
    train_labels,
    test_padded_data,
    test_labels,
    epochs,
):
    # Train model
    model.train_model(
        train_padded_data,
        train_labels,
        test_padded_data,
        test_labels,
        epochs=epochs,
        verbose=VERBOSE,
    )

    # Evaluate model and collect f1score and classification report
    f1score, report = model.evaluate_model(test_padded_data, test_labels)

    output = '\n'.join(
        [
            f'{"-"*40}{model.__name__}{"-"*40}',
            f'Epochs: {epochs}',
            f'F1-score: {f1score}',
            f'Classification Report:\n{report}',
        ]
    )
    # Print results
    print(output)
    # Save evaluation results to log file
    logger.info(output)


def main(train=None, save_word_index=False):
    # Load and organise data into train and test
    train_sentences, train_labels = load_data(TRAIN_DATA_PATH)
    test_sentences, test_labels = load_data(TEST_DATA_PATH)
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

    # Save word index to file if requested by user
    if save_word_index:
        save_word_index_file(tokenizer, 'tokenizer_word_index.json')

    # Create embedding matrix for models
    num_words, embedding_matrix = get_embedding_matrix(tokenizer)

    models = []
    if train == None or 'ffnn' in train:
        # Create FFNN model
        models.append(FFNN(num_words, max_len, EMBEDDING_DIM, embedding_matrix))

    if train == None or 'lstm' in train:
        # Create LSTM model
        models.append(
            LSTM(
                num_words,
                max_len,
                EMBEDDING_DIM,
                embedding_matrix,
            )
        )

    if train == None or 'lstm_attention' in train:
        # Create LSTM model with attention
        models.append(
            LSTMWithAttention(
                num_words,
                max_len,
                EMBEDDING_DIM,
                embedding_matrix,
                drop_rate=0.2,
            )
        )

    for model in models:
        # Train and evaluate the collected models (SVM handled separately below)
        train_and_evaluate_model(
            model,
            train_padded_data,
            train_labels,
            test_padded_data,
            test_labels,
            epochs=EPOCHS,
        )

    if train == None or 'svm' in train:
        # Create SVM model
        svm_model = SVM(TRAIN_DATA_PATH, TEST_DATA_PATH)
        # Tain and evaluate model
        output = svm_model.train_and_evaluate()
        # Print results
        print(output)
        # Save evaluation results to log file
        logger.info(output)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Models')

    parser.add_argument(
        '--train',
        dest='train',
        help='Model to train',
        choices=['ffnn', 'lstm', 'lstm_attention', 'svm'],
        action='append',
    )

    # Option to save word indx to a file
    parser.add_argument(
        '--save_word_index',
        dest='save_word_index',
        help='Flag to save generated word index to file',
        action='store_true',
    )

    args = parser.parse_args()
    main(args.train, args.save_word_index)
