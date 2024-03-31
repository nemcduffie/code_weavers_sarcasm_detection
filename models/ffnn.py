import numpy as np
import torch.nn as nn

from tensorflow import keras
from keras.initializers import Constant
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report


class FFNN(nn.Module):
    __name__ = 'FFNN'

    def __init__(self, num_words, max_len, embedding_dim, embedding_matrix):
        self.model = keras.Sequential()
        self.model.add(
            keras.layers.Embedding(
                input_dim=num_words,
                output_dim=embedding_dim,
                input_length=max_len,
                embeddings_initializer=Constant(embedding_matrix),
                trainable=False,
            )
        )
        self.model.add(keras.layers.GlobalAveragePooling1D())
        self.model.add(
            keras.layers.Dense(63, activation='relu')
        )  # First hidden layer
        self.model.add(
            keras.layers.Dense(32, activation='tanh')
        )  # Second hidden layer
        self.model.add(
            keras.layers.Dense(1, activation='sigmoid')
        )  # Output layer
        self.model.compile(
            optimizer='adam', loss='binary_crossentropy', metrics=['accuracy']
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
