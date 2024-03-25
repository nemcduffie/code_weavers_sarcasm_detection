import torch.nn as nn
from sklearn.metrics import (
    classification_report,
    f1_score,
)

from keras.models import Sequential
from keras.layers import LSTM as KerasLSTM
from keras.layers import Dense, Embedding, Dropout, Bidirectional, Attention
from keras.initializers import Constant
import keras.backend as kb


class LSTM(nn.Module):
    __name__ = 'LSTM'

    def __init__(
        self,
        num_words,
        max_len,
        embedding_dim,
        embedding_matrix,
        drop_rate=0.25,
    ):
        # Define initial model
        self.model = Sequential()
        # Add embedding layer with predefined matrix
        self.model.add(
            Embedding(
                input_dim=num_words,
                output_dim=embedding_dim,
                input_length=max_len,
                embeddings_initializer=Constant(embedding_matrix),
                trainable=False,
            )
        )
        # Add Bidirectional LSTM
        self.model.add(
            Bidirectional(
                KerasLSTM(
                    units=embedding_dim,
                    dropout=drop_rate,
                    use_bias=True,
                )
            )
        )
        self.model.add(Dense(128, activation='relu'))
        self.model.add(Dropout(drop_rate))
        self.model.add(Dense(64))
        self.model.add(Dropout(drop_rate))
        self.model.add(Dense(1, activation='sigmoid'))
        self.model.compile(
            optimizer='adam',
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
        epochs=25,
        verbose=1,
        class_weight={0: 1, 1: 2},
    ):
        return self.model.fit(
            train_data,
            train_labels,
            epochs=epochs,
            class_weight=class_weight,
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


class LSTMWithAttention(LSTM):
    __name__ = 'LSTMWithAttention'

    def __init__(
        self,
        num_words,
        max_len,
        embedding_dim,
        embedding_matrix,
        drop_rate=0.3,
    ):
        super().__init__(
            num_words,
            max_len,
            embedding_dim,
            embedding_matrix,
            drop_rate,
        )

        # Define attention mechanism
        self.attention = Attention(use_scale=True)

        # Add dropout layer
        self.dropout = Dropout(drop_rate)

    # Forward pass function
    def forward(self, x):
        # Pass input through embedding layer
        embedded = self.model.layers[0](x)

        # Pass embedded input through bidirectional LSTM layer
        lstm_output = self.model.layers[1](embedded)

        # Apply dropout
        lstm_output = self.dropout(lstm_output)

        # Apply attention mechanism
        attended_output = self.attention([lstm_output, lstm_output])

        # Flatten attended output
        attended_output = kb.squeeze(attended_output, axis=1)

        # Return output after passing through fully connected layers
        return self.model.layers[2](attended_output)

    # Train model function
    def train_model(
        self,
        train_data,
        train_labels,
        val_data,
        val_labels,
        epochs=25,
        verbose=1,
    ):
        return super().train_model(
            train_data,
            train_labels,
            val_data,
            val_labels,
            epochs=epochs,
            verbose=verbose,
        )

    # Evaluation function
    def evaluate_model(self, test_data, test_labels):
        return super().evaluate_model(test_data, test_labels)
