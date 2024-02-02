from collections import defaultdict, Counter


class BPE:
    def __init__(
            self,
            corpus: list[str],
    ):
        self.corpus = corpus
        self.vocab = []
        self.word_freq = Counter()
        self.splits = {}  # e.g. highest: [high, est$]
        self.merges = {}  # e.g. [high, est$]: highest

    def train(self):
        """
        Train a BPE Tokenizer
        """
        # count the word frequency
        for document in self.corpus:
            # split each document in corpus by whitespace
            words = document.split()
            self.word_freq += Counter(words)

        # initialize the self.splits
        for word in self.word_freq:
            self.splits[word] = list(word) + ["$"]

        alphabet = set()
        for word in self.word_freq:
            alphabet |= set(list(word))
        alphabet.add("$")

        self.vocab = list(alphabet)
        self.vocab.sort()

        prev_pair_freq = None
        while True:

            # find the most frequent pair
            pair_freq = self.get_pairs_freq()

            if pair_freq == prev_pair_freq:
                break

            prev_pair_freq = pair_freq

            if len(pair_freq) == 0:
                break

            pair = max(pair_freq, key=pair_freq.get)

            self.update_splits(pair[0], pair[1])

            self.merges[pair] = pair[0] + pair[1]

            self.vocab.append(pair[0] + pair[1])

            self.vocab = list(set(self.vocab))

            self.vocab.sort()

    def update_splits(self, lhs: str, rhs: str):
        """
        If we see lhs and rhs appear consecutively, we merge them
        """
        for word, word_split in self.splits.items():
            new_split = []
            cursor = 0
            while cursor < len(word_split):
                if (
                        word_split[cursor] == lhs
                        and cursor + 1 < len(word_split)
                        and word_split[cursor + 1] == rhs
                ):
                    new_split.append(lhs + rhs)
                    cursor += 2
                else:
                    new_split.append(word_split[cursor])
                    cursor += 1
            self.splits[word] = new_split

    def get_pairs_freq(self) -> dict:
        """
        Compute the pair frequency
        """
        pairs_freq = defaultdict(int)
        for word, freq in self.word_freq.items():
            split = self.splits[word]
            for i in range(len(split)):
                if i + 1 < len(split):
                    pairs_freq[(split[i], split[i + 1])] += freq

        return pairs_freq

    def tokenize(self, s: str) -> list[str]:
        splits = [list(t) + ["$"] for t in s.split()]

        for lhs, rhs in self.merges:
            for idx, split in enumerate(splits):
                new_split = []
                cursor = 0
                while cursor < len(split):
                    if (
                            cursor + 1 < len(split)
                            and split[cursor] == lhs
                            and split[cursor + 1] == rhs
                    ):
                        new_split.append(lhs + rhs)
                        cursor += 2
                    else:
                        new_split.append(split[cursor])
                        cursor += 1
                assert "".join(new_split) == "".join(split)
                splits[idx] = new_split

        return sum(splits, [])


corpus_file_path = "../data/Corpus.txt"
num_merges = 100

with open(corpus_file_path, "r", encoding="utf-8") as file:
    corpus = file.read().splitlines()

tokenizer = BPE(corpus)

tokenizer.train()

print(tokenizer.get_pairs_freq().values())
