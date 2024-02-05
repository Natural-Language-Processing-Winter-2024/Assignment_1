import numpy as np
import pandas as pd
import random
from tqdm import tqdm
from utils import emotion_scores
import pickle
import nltk
nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet

lemmatizer = WordNetLemmatizer()


class BigramLM:
    def __init__(self, emotions):
        self.vocab = None
        self.vocab_size = None
        self.bigram_counts = None
        self.unigram_counts = None
        self.smoothed_bigram_probs = None
        self.emotion_tables = {emotion: None for emotion in emotions}

    def preprocess_corpus(self, corpus):
        processed_corpus = [['<s>'] + sentence.split() + ['</s>'] for sentence in corpus]
        # Lemmatize the words
        # processed_corpus = [[lemmatizer.lemmatize(word) for word in sentence] for sentence in processed_corpus]
        # Remove punctuations
        # processed_corpus = [[word for word in sentence if word.isalnum()] for sentence in processed_corpus]

        self.vocab = sorted(list(set([word.lower() for sentence in processed_corpus for word in sentence])))
        self.vocab_size = len(self.vocab)
        return processed_corpus

    def learn_model(self, processed_corpus):
        self.bigram_counts = np.zeros((self.vocab_size, self.vocab_size))
        self.unigram_counts = np.zeros(self.vocab_size)

        for sentence in processed_corpus:
            for i in range(len(sentence) - 1):
                current_word_idx = self.vocab.index(sentence[i])
                next_word_idx = self.vocab.index(sentence[i + 1])
                self.bigram_counts[current_word_idx, next_word_idx] += 1
                self.unigram_counts[current_word_idx] += 1

        # Calculate smoothed bigram probabilities
        self.smoothed_bigram_probs = self.laplace_smoothing()

        # self.calculate_emotional_probabilities()

    def laplace_smoothing(self, alpha=1):
        return (self.bigram_counts + alpha) / (self.unigram_counts[:, np.newaxis] + alpha * self.vocab_size)

    def kneser_ney_smoothing(self, discount=0.75):
        lambda_k = discount / self.unigram_counts[:, np.newaxis].sum(axis=1)
        p_continuation = np.maximum(self.bigram_counts - discount, 0) / self.unigram_counts[:, np.newaxis]
        p_discount = discount * lambda_k * self.unigram_counts[:, np.newaxis].sum(axis=1) / self.unigram_counts[:, np.newaxis]
        p_discount /= self.unigram_counts[:, np.newaxis]

        return p_continuation + p_discount

    def calculate_emotional_probabilities(self, processed_corpus):
        emotions = ['sadness', 'joy', 'love', 'anger', 'fear', 'surprise']
        self.emotion_tables = {emotion: np.zeros((self.vocab_size, self.vocab_size)) for emotion in emotions}
        print("Calculating emotional probabilities...")
        for _,i in enumerate(tqdm(range(len(processed_corpus)))):
            # bigram = f"{self.vocab[i]} {self.vocab[j]}"
            # print(bigram)
            # sentence = ' '.join(processed_corpus[i][1:-1])
            # emotion_score_sentence = emotion_scores(sentence)
            # print("\n---------------------------------------------------------------Bigram Emotion:" + bigram)
            # print(emotion_score)
            for j1 in range(1,len(processed_corpus[i]) - 2):
                j2 = j1 + 1
                current_word = processed_corpus[i][j1]
                next_word = processed_corpus[i][j2]
                bigram__ = current_word + ' ' + next_word
                # print(bigram__)
                emotion_score = emotion_scores(bigram__)

                current_word_idx = self.vocab.index(current_word)
                next_word_idx = self.vocab.index(next_word)
                for k, label_score in enumerate(emotion_score):
                    self.emotion_tables[label_score['label']][current_word_idx, next_word_idx] += label_score['score']

    def normalize_table(self, table):
        row_sums = table.sum(axis=1, keepdims=True)
        col_sums = table.sum(axis=0, keepdims=True)
        normalized_table = table / (row_sums + 1e-10)  # Avoid division by zero
        normalized_table = normalized_table / (col_sums + 1e-10)  # Avoid division by zero
        return normalized_table

    def get_smoothed_probabilities_with_emotion(self, smoothing_type='laplace'):
        if smoothing_type == 'laplace':
            smoothed_probs = self.laplace_smoothing()
        elif smoothing_type == 'kneser-ney':
            smoothed_probs = self.kneser_ney_smoothing()
        else:
            raise ValueError("Invalid smoothing type. Choose 'laplace' or 'kneser-ney'.")

        return smoothed_probs

    def generate_sentence_with_emotion(self, emotion, max_length=10):
        if self.smoothed_bigram_probs is None or emotion not in self.emotion_tables:
            raise ValueError("Model not trained or invalid emotion provided. Use learn_model() method first.")

        sentence = ['<s>']
        current_word = '<s>'

        for _ in range(max_length):
            next_word_probs = self.smoothed_bigram_probs[self.vocab.index(current_word), :]
            emotion_table = self.emotion_tables[emotion][self.vocab.index(current_word), :]
            combined_probs = next_word_probs + emotion_table

            combined_probs = combined_probs/np.sum(combined_probs)

            # print('Shape: ', np.shape(next_word_probs))
            # print('Shape: ', np.shape(combined_probs))
            # print('Original prob: ', next_word_probs)
            # print('+Emotion prob: ', combined_probs)
            # print('Sum: ', np.sum(next_word_probs))
            # print('Sum: ', np.sum(combined_probs))

            next_word = np.random.choice(self.vocab, p=combined_probs)

            if next_word == '</s>':
                break

            sentence.append(next_word)
            current_word = next_word

        return ' '.join(sentence[1:])

    def get_emotion_tables(self):
        return self.emotion_tables
    

# Specify the path to your corpus file
corpus_file_path = "corpus.txt"

# Read the corpus file and create a list of sentences
with open(corpus_file_path, 'r', encoding='utf-8') as file:
    corpus = [line.strip() for line in file]

emotions = ['sadness', 'joy', 'love', 'anger', 'fear', 'surprise']

# Create an instance of the BigramLM class
bigram_model = BigramLM(emotions)
processed_corpus = bigram_model.preprocess_corpus(corpus)

# Train the bigram language model
bigram_model.learn_model(processed_corpus)

# Calculate emotional probabilities
bigram_model.calculate_emotional_probabilities(processed_corpus)