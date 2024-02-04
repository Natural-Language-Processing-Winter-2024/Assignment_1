import os
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score

# Load the training data
with open('corpus.txt', 'r', encoding='utf-8') as file:
    train_texts = file.readlines()

# Load the labels for each sentence
with open('labels.txt', 'r', encoding='utf-8') as file:
    train_labels = file.readlines()

# Remove newline characters from the loaded training labels
for i in range(len(train_texts)):
    train_texts[i] = train_texts[i].strip()
    train_labels[i] = train_labels[i].strip()


# Define emotions
emotions = ['sadness', 'joy', 'love', 'anger', 'fear', 'surprise']

# Function to load testing data
def load_testing_data(emotion):
    file_path = f'sent_bigram_{emotion}.txt'
    with open(file_path, 'r', encoding='utf-8') as file:
        data = file.readlines()
    labels = [emotion] * len(data)
    texts = [' '.join(line.split()) for line in data]
    return labels, texts


# Create the SVM model with TF-IDF vectorizer
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer()),
    ('svc', SVC())
])

# Parameters for Grid Search
parameters = {
    'tfidf__max_df': (0.25, 0.5, 0.75),
    'tfidf__ngram_range': [(1, 1), (1, 2)],
    'svc__C': [1, 10, 100, 1000],
    'svc__kernel': ['linear', 'rbf']
}

# Perform Grid Search
grid_search = GridSearchCV(pipeline, parameters, n_jobs=-1, verbose=1)
grid_search.fit(train_texts, train_labels)

# Print best parameters from Grid Search
print("Best Parameters:", grid_search.best_params_)

# Use the best model for testing
best_model = grid_search.best_estimator_

# Evaluate the model for each emotion
for emotion in emotions:
    test_labels, test_texts = load_testing_data(emotion)
    predictions = best_model.predict(test_texts)

    # Calculate and print the accuracy and classification report
    accuracy = accuracy_score(test_labels, predictions)
    report = classification_report(test_labels, predictions)
    
    print(f"Emotion: {emotion}")
    print(f"Accuracy: {accuracy}")
    print("Classification Report:")
    print(report)
    print("\n")