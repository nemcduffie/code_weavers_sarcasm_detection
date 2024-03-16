import numpy as np
import pandas as pd
import seaborn as sns
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    f1_score,
)
from sklearn.model_selection import train_test_split

from tensorflow import keras

from keras.optimizers import Adam

# from keras.optimizers.legacy import Adam
from keras.models import Sequential
from keras.layers import LSTM as KerasLSTM
from keras.layers import Dense, Embedding, Dropout, Bidirectional
from keras.initializers import Constant


class LSTM(nn.Module):
    def __init__(
        self,
        num_words,
        max_len,
        embedding_dim,
        embedding_matrix,
        drop_rate=0.5,
        learning_rate=0.2,
    ):
        self.model = Sequential()
        self.model.add(
            Embedding(
                input_dim=num_words,
                output_dim=embedding_dim,
                input_length=max_len,
                embeddings_initializer=Constant(embedding_matrix),
                trainable=False,
            )
        )
        self.model.add(
            Bidirectional(
                KerasLSTM(
                    units=embedding_dim,
                    recurrent_dropout=0.5,
                    dropout=drop_rate,
                )
            )
        )
        self.model.add(Dense(40, activation='relu'))
        self.model.add(Dropout(drop_rate))
        self.model.add(Dense(20))
        self.model.add(Dense(1, activation='sigmoid'))
        self.model.compile(
            optimizer=Adam(learning_rate=learning_rate),
            loss='binary_crossentropy',
            metrics=['accuracy'],
        )

    # Train model function
    def train_model(
        self,
        train_data,
        train_labels,
        val_data,
        val_labels,
        epochs=20,
        verbose=1,
    ):
        return self.model.fit(
            train_data,
            train_labels,
            epochs=epochs,
            validation_data=(val_data, val_labels),
            verbose=verbose,
        )

    # Evaluation function
    def evaluate_model(self, test_data, test_labels):
        # Test the model
        predictions = self.model.predict(test_data)
        predictions = (predictions > 0.5).astype(
            int
        )  # Convert the probabilities into binary predictions

        # Calculate F1-score
        f1score = f1_score(test_labels, predictions)

        # Classification report
        report = classification_report(test_labels, predictions)

        return f1score, report
