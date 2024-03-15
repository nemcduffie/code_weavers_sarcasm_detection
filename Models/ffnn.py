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


embedding_file_path =  './glove.twitter.27B/glove.twitter.27B.200d.txt'
train_file_path = './prep_train.json'
test_file_path = './prep_test.json'


# Load train and test files
with open(train_file_path, 'r', encoding='utf-8') as train_file:
    train_data = json.load(train_file)

with open(test_file_path, 'r', encoding='utf-8') as test_file:
    test_data = json.load(test_file)

# Extract data from files
train_sentences = [entry['text_with_emojis'] for entry in train_data] # Tokenized and preprocessed
train_labels = [entry['sarcastic'] for entry in train_data]

test_sentences = [entry['text_with_emojis'] for entry in test_data] # Tokenized and preprocessed
test_labels = [entry['sarcastic'] for entry in test_data]

all_sentences = train_sentences + test_sentences

# Convert labels to numpy arrays
train_labels = np.array(train_labels)
test_labels = np.array(test_labels)

# Maximun lenght function 
def max_tweet_length(preprocessed_text):
    max_len = 0
    for tweet in preprocessed_text:
        tweet_len = len(tweet)
        if tweet_len > max_len:
            max_len = tweet_len
    return max_len

max_len = max_tweet_length(all_sentences)


# Define the tokenizer
tokenizer = Tokenizer()
tokenizer.fit_on_texts(all_sentences)

# Padding data function to have the same lenght in all inputs
def padding_data(sentences, max_len, tokenizer):
    sequences = tokenizer.texts_to_sequences(sentences) # Convert tokens to integers sequences
    padded_data = pad_sequences(sequences, maxlen=max_len, truncating='post', padding='post')  # Pad sequences
    return padded_data

train_padded_data = padding_data(train_sentences, max_len, tokenizer)
test_padded_data = padding_data(test_sentences, max_len, tokenizer)

# # Saving word index 
# def save_word_index(tokenizer, filename):
#     word_index = tokenizer.word_index
#     with open(filename, 'w') as f:
#         json.dump(word_index, f, indent=4)

# save_word_index(tokenizer, 'tokenizer_word_index.json')
        

# Create embedding dictionary as numpy array
embedding_dict = {}
with open(embedding_file_path, 'r', encoding='utf8') as f:
    for line in f:
        values = line.split()
        word = values [0]
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


# Create model function
def create_model():
    model = keras.Sequential()
    model.add(
        keras.layers.Embedding(
        input_dim=num_words,
        output_dim=200,
        input_length=max_len,
        embeddings_initializer=Constant(embedding_matrix),
        trainable=False
        )
    )
    model.add(keras.layers.GlobalAveragePooling1D())
    model.add(keras.layers.Dense(63, activation='relu'))  # First hidden layer
    model.add(keras.layers.Dense(32, activation='tanh'))  # Second hidden layer
    model.add(keras.layers.Dense(1, activation='sigmoid'))  # Output layerd
    model.summary()
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Train model function
def train_model(model, train_data, train_labels, val_data, val_labels, epochs=20, verbose=1):
    history = model.fit(
        train_data,
        train_labels,
        epochs=epochs,
        validation_data=(val_data, val_labels),
        verbose=verbose
    )
    return history

# Evaluation function
def evaluate_model(model, test_data, test_labels):
    # Test the model
    predictions = model.predict(test_data)
    predictions = (predictions > 0.5).astype(int)  # Convert the probabilities into binary predictions
    
    # Calculate F1-score
    f1score = f1_score(test_labels, predictions)
    
    # Classification report
    report = classification_report(test_labels, predictions)
    
    return f1score, report


# Create model
model = create_model()

# Train model
history = train_model(model, train_padded_data, train_labels, test_padded_data, test_labels)
f1score, report = evaluate_model(model, test_padded_data, test_labels)

print("F1-score:", f1score)
print("Classification Report:")
print(report)
    
