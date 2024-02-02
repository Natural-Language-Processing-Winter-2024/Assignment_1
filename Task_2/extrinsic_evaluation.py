import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, accuracy_score

# Load the original corpus and labels
with open('../data/corpus.txt', 'r', encoding='utf-8') as file:
    original_corpus = file.readlines()

with open('../data/labels.txt', 'r', encoding='utf-8') as file:
    original_labels = file.readlines()

# Load the generated samples and labels for testing
with open('../output/generated.txt', 'r', encoding='utf-8') as file:
    generated_samples = file.readlines()

with open('../output/generated_labels.txt', 'r', encoding='utf-8') as file:
    generated_labels = file.readlines()

# Vectorize the text using TF-IDF
vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2), stop_words='english')

# Transform the original corpus
X_train_vectorized = vectorizer.fit_transform(original_corpus)

# Transform the generated samples
X_test_vectorized = vectorizer.transform(generated_samples)

# Define the Support Vector Classifier (SVC) and perform Grid Search
param_grid = {'C': [0.1, 1, 10, 100], 'kernel': ['linear', 'rbf', 'poly']}
svc = SVC()

grid_search = GridSearchCV(svc, param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train_vectorized, original_labels)

# Get the best model from Grid Search
best_svc = grid_search.best_estimator_

# Print the best parameters
print("Best Parameters:", grid_search.best_params_)

# Evaluate the model on the generated samples
y_pred = best_svc.predict(X_test_vectorized)

# Print classification report and accuracy
print("Classification Report:\n", classification_report(generated_labels, y_pred))
print("Accuracy:", accuracy_score(generated_labels, y_pred))

# Save the model
import pickle

with open('svc_model.pkl', 'wb') as file:
    pickle.dump(best_svc, file)

# Load the model
with open('svc_model.pkl', 'rb') as file:
    model = pickle.load(file)

# Evaluate the model on the generated samples
y_pred = model.predict(X_test_vectorized)

