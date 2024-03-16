import numpy as np
import json
import torch.nn as nn

from tensorflow import keras
from keras.utils import pad_sequences
from keras.initializers import Constant
from keras.optimizers import Adam
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split


class FFNN(nn.Module):
    # Create model function
    def create_model(num_words, max_len, embedding_matrix):
        model = keras.Sequential()
        model.add(
            keras.layers.Embedding(
                input_dim=num_words,
                output_dim=200,
                input_length=max_len,
                embeddings_initializer=Constant(embedding_matrix),
                trainable=False,
            )
        )
        model.add(keras.layers.GlobalAveragePooling1D())
        model.add(
            keras.layers.Dense(63, activation='relu')
        )  # First hidden layer
        model.add(
            keras.layers.Dense(32, activation='tanh')
        )  # Second hidden layer
        model.add(keras.layers.Dense(1, activation='sigmoid'))  # Output layer
        model.summary()
        model.compile(
            optimizer='adam', loss='binary_crossentropy', metrics=['accuracy']
        )
        return model

    # Train model function
    def train_model(
        model,
        train_data,
        train_labels,
        val_data,
        val_labels,
        epochs=20,
        verbose=1,
    ):
        history = model.fit(
            train_data,
            train_labels,
            epochs=epochs,
            validation_data=(val_data, val_labels),
            verbose=verbose,
        )
        return history

    # Evaluation function
    def evaluate_model(model, test_data, test_labels):
        # Test the model
        predictions = model.predict(test_data)
        predictions = (predictions > 0.5).astype(
            int
        )  # Convert the probabilities into binary predictions

        # Calculate F1-score
        f1score = f1_score(test_labels, predictions)

        # Classification report
        report = classification_report(test_labels, predictions)

        return f1score, report
