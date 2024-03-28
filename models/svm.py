import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import f1_score, accuracy_score, classification_report


class SVM:
    def __init__(self, train_data_path, test_data_path):
        # Load training data in JSON format
        train_data = pd.read_json(train_data_path)

        # Load testing data in JSON format
        test_data = pd.read_json(test_data_path)

        # Extract features and labels for training data
        self.X_train = train_data['text_with_embeddings'].astype(str)
        self.y_train = train_data['sarcastic']

        # Extract features and labels for testing data
        self.X_test = test_data['text_with_embeddings'].astype(str)
        self.y_test = test_data['sarcastic']

        # Convert text data to feature vectors using TfidfVectorizer
        vectorizer = TfidfVectorizer()
        self.X_train_vec = vectorizer.fit_transform(self.X_train)
        self.X_test_vec = vectorizer.transform(self.X_test)

    def train_and_evaluate(self):
        # Initialize and train SVM model
        self.model = SVC(
            kernel='linear'
        )  # You can experiment with different kernels
        self.model.fit(self.X_train_vec, self.y_train)

        # Make predictions on the test set
        y_pred = self.model.predict(self.X_test_vec)

        # Evaluate the model using F1-score
        f1 = f1_score(self.y_test, y_pred)
        accuracy = accuracy_score(self.y_test, y_pred)

        line = '----------------------------------------'
        print(f'{line}SVM{line}')

        print(f"F1-Score: {f1}")
        print(f"Accuracy: {accuracy}")

        # Additional metrics and classification report
        print("\nClassification Report:")
        print(classification_report(self.y_test, y_pred))
