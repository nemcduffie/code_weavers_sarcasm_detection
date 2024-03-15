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
from keras.optimizers import Adam
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM, Dropout, Bidirectional, GRU


class LSTM(nn.Module):
    def __init__(
        self,
        input_size,
        hidden_size,
        embedding_size,
        embedding_matrix,
        n_layers=2,
        drop_rate=0.5,
        batch_size=1,
    ):
        self.model = Sequential()
        self.batch_size = batch_size
        self.n_layers = n_layers
        self.model.add(
            Embedding(
                input_size,
                output_dim=hidden_size,
                weights=[embedding_matrix],
                input_length=embedding_size,
                trainable=True,
            )
        )
        # self.model.add(Embedding(input_size, 300, weights=[embedding_matrix],
        #                     input_length=254,  trainable=False))

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
        self.model.compile(
            optimizer=Adam(lr=0.01),
            loss='binary_crossentropy',
            metrics=['accuracy'],
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

    # def __init__(
    #     self,
    #     input_size,
    #     hidden_size,
    #     embedding_dim,
    #     n_layers=2,
    #     drop_rate=None,
    #     batch_size=1,
    # ):
    #     super(LSTM, self).__init__()

    #     self.input_size = input_size  # total number of all possible chars
    #     self.embedding_dim = embedding_dim  # dimention of the embedding layer
    #     self.hidden_size = hidden_size  # dimention of the hidden layer
    #     self.n_layers = n_layers  # number of layers in the hidden layer

    #     # add the embedding layer
    #     self.embedding = nn.Embedding(input_size, embedding_dim, padding_idx=0)
    #     # add the LSTM layer
    #     self.rnn = nn.LSTM(
    #         input_size=input_size,
    #         hidden_size=hidden_size,
    #         num_layers=n_layers,
    #     )
    #     # add a dropout layer if drop_rate is included in params
    #     self.dropout = nn.Dropout(drop_rate) if drop_rate != None else None
    #     # add the decoder layer
    #     self.linear = nn.Linear(hidden_size, input_size)

    # def forward(self, x, state):
    #     x = self.embedding(x)
    #     # ensure x is the correct shape for the rest of the steps
    #     x = x.flatten().view([1, self.embedding_dim])
    #     # if a dropout layer has been defined run x through it
    #     if self.dropout != None:
    #         x = self.dropout(x)
    #     # run x and our previous hidden and cell states through the lstm layer,
    #     # returning the next predicted output along with the current hidden and
    #     # cell states
    #     output, (hidden, cell) = self.rnn(x, state)
    #     # run the predicted output through our final decoder layer
    #     output = self.linear(output)
    #     return output, (hidden.detach(), cell.detach())

    # def init_hidden(self):
    #     """Initialise hidden layers"""
    #     hidden = torch.zeros(self.n_layers, self.hidden_size)
    #     cell = torch.zeros(self.n_layers, self.hidden_size)
    #     return hidden, cell
