import numpy as np
from collections import defaultdict

from Task_2.utils import classifier


class BigramLM:
    def __init__(self):
        self.bigram_counts = defaultdict(lambda: defaultdict(int))
        self.unigram_counts = defaultdict(int)
        self.vocab_size = 0

    def learn_model(self, corpus):
        for sentence in corpus:
            tokens = sentence.split()
            for i in range(len(tokens) - 1):
                self.bigram_counts[tokens[i]][tokens[i + 1]] += 1
                self.unigram_counts[tokens[i]] += 1
            # Handling last token
            self.unigram_counts[tokens[-1]] += 1

        self.vocab_size = len(self.unigram_counts)

    def laplace_smoothing(self, word, prev_word):
        count_bigram = self.bigram_counts[prev_word].get(word, 0)
        count_unigram = self.unigram_counts[prev_word]
        return (count_bigram + 1) / (count_unigram + self.vocab_size)

    def kneser_ney_smoothing(self, word, prev_word):
        lambda_val = 0.75  # Kneser-Ney smoothing parameter
        count_bigram = self.bigram_counts[prev_word].get(word, 0)
        count_prev_word = sum(self.bigram_counts[prev_word].values())
        prob_cont = max(count_bigram - lambda_val, 0) / count_prev_word
        prob_disc = lambda_val * (len(self.bigram_counts[prev_word]) / count_prev_word)
        return prob_cont + prob_disc * self.unigram_counts[word] / self.vocab_size

    def get_probability(self, word, prev_word, smoothing="laplace"):
        if smoothing == "laplace":
            return self.laplace_smoothing(word, prev_word)
        elif smoothing == "kneser-ney":
            return self.kneser_ney_smoothing(word, prev_word)
        else:
            raise ValueError("Invalid smoothing method")

    def generate_next_word(self, prev_word, smoothing="laplace"):
        if prev_word in self.bigram_counts:
            next_words = list(self.bigram_counts[prev_word].keys())
            probabilities = [self.get_probability(w, prev_word, smoothing) for w in next_words]
            return np.random.choice(next_words, p=probabilities)
        else:
            return None


# Example usage:
corpus_file_path = '../data/Corpus.txt'
with open(corpus_file_path, 'r', encoding='utf-8') as file:
    corpus = file.read().splitlines()
lm = BigramLM()
lm.learn_model(corpus)

print("Laplace Smoothing:")
for prev_word in lm.bigram_counts:
    for word in lm.bigram_counts[prev_word]:
        prob = lm.get_probability(word, prev_word, "laplace")
        print(f"P({word}|{prev_word}) = {prob:.4f}")

print("\nKneser-Ney Smoothing:")
for prev_word in lm.bigram_counts:
    for word in lm.bigram_counts[prev_word]:
        prob = lm.get_probability(word, prev_word, "kneser-ney")
        print(f"P({word}|{prev_word}) = {prob:.4f}")



class BigramLM:
    def __init__(self):
        self.bigram_counts = defaultdict(lambda: defaultdict(int))
        self.unigram_counts = defaultdict(int)
        self.vocab_size = 0

    def learn_model(self, corpus):
        for sentence in corpus:
            tokens = sentence.split()
            for i in range(len(tokens) - 1):
                self.bigram_counts[tokens[i]][tokens[i + 1]] += 1
                self.unigram_counts[tokens[i]] += 1
            # Handling last token
            self.unigram_counts[tokens[-1]] += 1

        self.vocab_size = len(self.unigram_counts)

    def laplace_smoothing(self, word, prev_word):
        count_bigram = self.bigram_counts[prev_word].get(word, 0)
        count_unigram = self.unigram_counts[prev_word]
        return (count_bigram + 1) / (count_unigram + self.vocab_size)

    def kneser_ney_smoothing(self, word, prev_word):
        lambda_val = 0.75  # Kneser-Ney smoothing parameter
        count_bigram = self.bigram_counts[prev_word].get(word, 0)
        count_prev_word = sum(self.bigram_counts[prev_word].values())
        prob_cont = max(count_bigram - lambda_val, 0) / count_prev_word
        prob_disc = lambda_val * (len(self.bigram_counts[prev_word]) / count_prev_word)
        return prob_cont + prob_disc * self.unigram_counts[word] / self.vocab_size

    def get_probability(self, word, prev_word, emotion_beta, smoothing="laplace"):
        if smoothing == "laplace":
            prob = self.laplace_smoothing(word, prev_word)
        elif smoothing == "kneser-nay":
            prob = self.kneser_ney_smoothing(word, prev_word)
        else:
            raise ValueError("Invalid smoothing method")

        # Include emotion component
        emotion_scores = classifier(prev_word + " " + word)
        emotion_beta = emotion_scores['score'] * emotion_beta  # Assuming 'score' corresponds to the emotion intensity
        return prob + emotion_beta

    def generate_next_word(self, prev_word, emotion_beta=0.1, smoothing="laplace"):
        if prev_word in self.bigram_counts:
            next_words = list(self.bigram_counts[prev_word].keys())
            probabilities = [self.get_probability(w, prev_word, emotion_beta, smoothing) for w in next_words]
            return np.random.choice(next_words, p=probabilities)
        else:
            return None

    def generate_emotion_oriented_sample(self, start_word, target_emotion, max_length=10, emotion_beta=0.):
        generated_sentence = [start_word]
        prev_word = start_word
        for _ in range(max_length):
            next_word = self.generate_next_word(prev_word, emotion_beta)
            if next_word:
                generated_sentence.append(next_word)
                prev_word = next_word
                if next_word == '<EOS>':  # Assume '<EOS>' is the end of sentence token
                    break
            else:
                break
        return ' '.join(generated_sentence)


# # Example usage:
# corpus = ["the cat sat on the mat", "the dog barked"]
lm = BigramLM()
lm.learn_model(corpus)

start_word = "car"
target_emotion = "sad"  # Example emotion
emotion_beta = 0.1  # Adjust the weight of emotion component

emotion_oriented_sample = lm.generate_emotion_oriented_sample(start_word, target_emotion, emotion_beta=emotion_beta)
print("Emotion-Oriented Sample:", emotion_oriented_sample)