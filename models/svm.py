import pandas as pd
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import f1_score, accuracy_score, classification_report


# Determine the path to the 'data' folder within the parent directory
current_dir = os.getcwd()
data_folder = os.path.join(os.path.dirname(current_dir), 'data')

# Load training data in JSON format
train_data_path = os.path.join(data_folder, 'prep_train.json')
train_data = pd.read_json(train_data_path)

# Load testing data in JSON format
test_data_path = os.path.join(data_folder, 'prep_test.json')
test_data = pd.read_json(test_data_path)

# Extract features and labels for training data
X_train = train_data['text_with_embeddings'].astype(str)
y_train = train_data['sarcastic']

# Extract features and labels for testing data
X_test = test_data['text_with_embeddings'].astype(str)
# Assuming the testing data has a column named 'sarcastic' to compare predictions
y_test = test_data['sarcastic']

# Convert text data to feature vectors using TfidfVectorizer
vectorizer = TfidfVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Initialize and train SVM model
svm_model = SVC(kernel='linear')  # You can experiment with different kernels
svm_model.fit(X_train_vec, y_train)

# Make predictions on the test set
y_pred = svm_model.predict(X_test_vec)

# Evaluate the model using F1-score
f1 = f1_score(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)

print(f"F1-Score: {f1}")
print(f"Accuracy: {accuracy}")

# Additional metrics and classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))