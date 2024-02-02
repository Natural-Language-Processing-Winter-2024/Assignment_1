import re
from collections import defaultdict


def get_stats(vocab):
    """
	Given a vocabulary (dictionary mapping words to frequency counts), returns a
	dictionary of tuples representing the frequency count of pairs of characters
	in the vocabulary.
	"""
    pairs = defaultdict(int)
    for word, freq in vocab.items():
        symbols = word.split()
        for i in range(len(symbols) - 1):
            pairs[symbols[i], symbols[i + 1]] += freq
    return pairs


def merge_vocab(pair, v_in):
    """
	Given a pair of characters and a vocabulary, returns a new vocabulary with the
	pair of characters merged together wherever they appear.
	"""
    v_out = {}
    bigram = re.escape(' '.join(pair))
    p = re.compile(r'(?<!\S)' + bigram + r'(?!\S)')
    for word in v_in:
        w_out = p.sub(''.join(pair), word)
        v_out[w_out] = v_in[word]
    return v_out


def get_vocab(data):
    """
	Given a list of strings, returns a dictionary of words mapping to their frequency
	count in the data.
	"""
    vocab = defaultdict(int)
    for line in data:
        for word in line.split():
            vocab[' '.join(list(word)) + ' $'] += 1
    return vocab


def byte_pair_encoding(data, n):
    """
	Given a list of strings and an integer n, returns a list of n merged pairs
	of characters found in the vocabulary of the input data.
	"""
    vocab = get_vocab(data)
    for i in range(n):
        pairs = get_stats(vocab)
        if len(pairs) == 0:
            return vocab
        best = max(pairs, key=pairs.get)
        vocab = merge_vocab(best, vocab)
    return vocab


# Example usage:
with open("../data/corpus.txt", "r", encoding="utf-8") as file:
    corpus = file.read().splitlines()

n = 10000
bpe_pairs = byte_pair_encoding(corpus, n)

print(bpe_pairs)
