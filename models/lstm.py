import numpy as np
import pandas as pd
import seaborn as sns
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
)
from sklearn.model_selection import train_test_split

from tensorflow import keras
from keras.optimizers import Adam
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM, Dropout, Bidirectional
from keras.initializers import Constant


class LSTM(nn.Module):
    def __init__(
        self,
        num_words,
        hidden_size,
        max_len,
        embedding_matrix,
        n_layers=2,  # TODO
        drop_rate=0.5,
        batch_size=1,
    ):
        self.model = Sequential()
        self.model.add(
            Embedding(
                input_dim=num_words,
                output_dim=200,
                input_length=max_len,
                embeddings_initializer=Constant(embedding_matrix),
                trainable=False,
            )
        )
        self.model.add(
            Bidirectional(
                LSTM(
                    units=hidden_size, recurrent_dropout=0.5, dropout=drop_rate
                )
            )
        )
        # TODO: try without bidirectional, adjust dim and dropouts?

        self.model.add(Dense(40, activation='relu'))
        self.model.add(Dropout(drop_rate))
        self.model.add(Dense(20))
        self.model.add(Dense(1, activation='sigmoid'))
        # self.model.compile(
        #     optimizer=Adam(lr=0.01),
        #     loss='binary_crossentropy',
        #     metrics=['accuracy'],
        # )
        self.model.compile(
            optimizer='adam', loss='binary_crossentropy', metrics=['accuracy']
        )

    def metrics(self, X_train, y_train, X_test, y_test):
        self.model.summary()
        self.history = self.model.fit(
            X_train,
            y_train,
            batch_size=self.batch_size,
            validation_data=(X_test, y_test),
            epochs=2,
        )
        # self.model.fit(
        #     encodings_train,
        #     data_train['is_sarcastic'],
        #     epochs=8,
        #     batch_size=64,
        #     validation_data=(encodings_test, data_test['is_sarcastic']),
        # )

        pred = self.model.predict_classes(X_test)
        print(
            classification_report(
                y_test, pred, target_names=['Not Sarcastic', 'Sarcastic']
            )
        )
        cm = confusion_matrix(y_test, pred)
        cm = pd.DataFrame(
            cm,
            index=['Not Sarcastic', 'Sarcastic'],
            columns=['Not Sarcastic', 'Sarcastic'],
        )
        plt.figure(figsize=(10, 10))
        sns.heatmap(
            cm,
            cmap="Blues",
            linecolor='black',
            linewidth=1,
            annot=True,
            fmt='',
            xticklabels=['Not Sarcastic', 'Sarcastic'],
            yticklabels=['Not Sarcastic', 'Sarcastic'],
        )
