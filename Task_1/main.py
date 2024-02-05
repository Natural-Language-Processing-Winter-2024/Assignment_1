import re
import collections


class Tokenizer:
    def __init__(self, text, num_merges):
        self.num_merges = num_merges
        self.text = text.split()
        self.word_freq_dict = collections.defaultdict(int)
        self.char_freq_dict = collections.defaultdict(int)
        self.merge_rules = []

    def learn_vocabulary(self):
        # Get word frequency
        for word in self.text:
            self.word_freq_dict[' '.join(word) + ' $'] += 1

        # Get character frequency
        for word, freq in self.word_freq_dict.items():
            chars = word.split()
            for char in chars:
                self.char_freq_dict[char] += freq

        # Byte pair encoding
        for _ in range(self.num_merges):
            pairs = collections.defaultdict(int)
            for word, freq in self.word_freq_dict.items():
                chars = word.split()
                for i in range(len(chars) - 1):
                    pairs[chars[i], chars[i + 1]] += freq

            best_pair = max(pairs, key=pairs.get)
            self.merge_rules.append(best_pair)
            bigram = re.escape(' '.join(best_pair))
            p = re.compile(r'(?<!\S)' + bigram + r'(?!\S)')

            merged_dict = {}
            for word in self.word_freq_dict:
                w_out = p.sub(''.join(best_pair), word)
                merged_dict[w_out] = self.word_freq_dict[word]

            self.word_freq_dict = merged_dict

        # Get subword tokens
        for word, freq in self.word_freq_dict.items():
            chars = word.split()
            for char in chars:
                self.char_freq_dict[char] += freq

    def tokenize(self, sentence):
        tokenized_sentence = []
        for word in sentence.split():
            word = word + '$'
            if word not in self.word_freq_dict:
                while len(word) != 0:
                    subwords = [subword for subword in self.word_freq_dict if word.startswith(subword)]
                    if len(subwords) == 0:
                        tokenized_sentence.append(word[:-1])
                        break
                    max_len_subword = max(subwords, key=len)
                    if max_len_subword[-1] == "$":
                        max_len_subword = max_len_subword[:-1]
                    tokenized_sentence.append(max_len_subword)
                    word = word[len(max_len_subword):]
            else:
                tokenized_sentence.append(word[:-1])
                continue
        return tokenized_sentence

    def get_tokens(self, filename):
        with open(f'../output/{filename}', 'w') as f:
            for token in self.char_freq_dict.keys():
                f.write(token + '$' + '\n')

    def get_merge_rules(self, filename):
        with open(f'../output/{filename}', 'w') as f:
            for rule in self.merge_rules:
                f.write(', '.join(rule) + '\n')

    def get_tokenized_sentences(self, samples, filename):
        with open(f'../output/{filename}', 'w') as f:
            for sentence in samples:
                sentence = self.tokenize(sentence)
                f.write(', '.join(words for words in sentence) + '\n')


filename = "../data/corpus.txt"
num_merges = 100

text = open(filename, 'r').read()
tokenizer = Tokenizer(text, num_merges)
tokenizer.learn_vocabulary()

text_file = "../data/test.txt"

with open(text_file, 'r') as f:
    text = f.readlines()

tokenizer.get_tokens('tokens.txt')
tokenizer.get_merge_rules('merge_rules.txt')
tokenizer.get_tokenized_sentences(text, 'tokenized_samples.txt')
