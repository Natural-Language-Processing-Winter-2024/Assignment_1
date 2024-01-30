from collections import defaultdict
import re
import os


class Tokenizer:
    """
    Tokenizer class
    : initialized a dictionary for vocabulary
    : initialized a list for merge rules
    """
    def __init__(self):
        self.vocab = defaultdict(int)
        self.merge_rules = set()

    def learn_vocabulary(self, corpus, num_merges):
        """
        :param corpus:
        :param num_merges:
        :return: void

        It learns the vocabulary from the provided corpus and merge rule (given number of merges)
        """
        for _ in range(num_merges):
            pairs = defaultdict(int)
            for sentence in corpus:
                words = sentence.split()
                for i in range(len(words) - 1):
                    pairs[words[i], words[i + 1]] += 1

            if not pairs:
                break

            best = max(pairs, key=pairs.get)
            self.merge_rules.add(best)

            # Merge the tokens in the vocabulary
            new_vocab = defaultdict(int)
            for word in self.vocab:
                new_word = re.sub(' '.join(best), ''.join(best), word)
                new_vocab[new_word] = self.vocab[word]
            self.vocab = new_vocab

    def tokenize(self, text):
        """
        :param text:
        :return: tokenized text

        It tokenizes the text given in the parameter based on the vocabulary and merge rule (given number of merges)
        and returns the tokenized text
        """
        tokens = []
        for word in text.split():
            word_tokens = word.split(' ')
            for token in word_tokens:
                if token in self.vocab:
                    tokens.extend(token.split(' '))
                else:
                    tokens.append(token)
        return tokens

    def get_vocabulary_tokens(self, file_name):
        """
        :param file_name:
        :return: a file where each line corresponds to 1 token

        It adds each token in vocabulary into each line of the file
        """
        with open(os.path.join("../output", file_name), 'w', encoding='utf-8') as file:
            for token in self.vocab.keys():
                file.write(token + '\n')

    def get_merge_rules(self, file_name):
        """
        :param file_name:
        :return: a file where each line corresponds to a learned merge rule

        It adds a merge rule into each line of the file
        """
        with open(os.path.join("../output", file_name), 'w', encoding='utf-8') as file:
            for rule in self.merge_rules:
                file.write(','.join(rule) + '\n')

    def tokenize_samples(self, samples, file_name):
        tokenized_samples = []
        for sample in samples:
            tokens = self.tokenize(sample)
            tokenized_samples.append(','.join(tokens))

        with open(os.path.join("../output", file_name), 'w', encoding='utf-8') as file:
            for sample in tokenized_samples:
                file.write(sample + '\n')



"""
Reading the Corpus from .txt file from given in the data directory. It reads the lines and split them into a list of strings.
We also define the number of merges which can be changed but here we have defined to be 10 (random)
"""

corpus_file_path = '../data/Corpus.txt'
num_merges = 10

with open(corpus_file_path, 'r', encoding='utf-8') as file:
    corpus = file.read().splitlines()

tokenizer = Tokenizer()
for sentence in corpus:
    for word in sentence.split():
        tokenizer.vocab[word] += 1

tokenizer.learn_vocabulary(corpus, num_merges)

# sample_text = "low high high low low high"
# tokens = tokenizer.tokenize(sample_text)
# print(tokens)

tokenizer.get_vocabulary_tokens("tokens.txt")

tokenizer.get_merge_rules("merge_rules.txt")

test_sample_file = "../data/test.txt"
with open(test_sample_file, 'r', encoding='utf-8') as file:
    test_sample = file.read().splitlines()

tokenizer.tokenize_samples(test_sample, "tokenized_samples.txt")
