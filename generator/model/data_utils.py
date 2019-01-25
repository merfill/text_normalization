import numpy as np
import os
import csv
import nltk


# shared global variables to be imported from model also
GO = '<go>'
EOS = '<eos>'
PAD = 0


# special error message
class MyIOError(Exception):
    def __init__(self, filename):
        # custom error message
        message = """
ERROR: Unable to locate file {}.

FIX: Have you tried running python build_data.py first?
This will build vocab file from your train, test and dev sets and
trimm your clause vectors.
""".format(filename)
        super(MyIOError, self).__init__(message)


class Dataset(object):
    def __init__(self, filename, labels=None, processing_chars=None, processing_words=None, maxiter=None):
        self.filename = filename
        self.processing_chars = processing_chars
        self.processing_words = processing_words
        self.length = None
        if labels is None:
            self.labels = None
        else:
            self.labels = set()
            self.labels.update(labels)
        if maxiter is None:
            self.maxiter = None
        else:
            self.maxiter = maxiter


    def __iter__(self):
        with open(self.filename, 'rb') as f:
            reader = csv.DictReader(f, delimiter=',', quotechar='"')
            n = 0
            for row in reader:
                if self.labels is not None and row['class'] not in self.labels:
                    continue
                before, after = row['before'], row['after']
                if self.processing_chars is not None:
                    before = self.processing_chars(before)
                if self.processing_words is not None:
                    after = self.processing_words(after)
                yield before, after
                if self.maxiter is not None:
                    n += 1
                    if n >= self.maxiter:
                        break


    def __len__(self):
        if self.length is None:
            self.length = 0
            for _ in self:
                self.length += 1
        return self.length


def load_vocab(filename):
    try:
        d = dict()
        v = dict()
        with open(filename) as f:
            for idx, clause in enumerate(f):
                clause = clause.rstrip('\n')
                # 0 is reserved for unknown token
                d[clause] = idx + 1
                v[idx + 1] = clause
    except IOError:
        raise MyIOError(filename)
    return d, v


def write_vocab(vocab, filename):
    print('Writing vocab to {0}...'.format(filename))
    with open(filename, "w") as f:
        for i, word in enumerate(vocab):
            if i != len(vocab) - 1:
                f.write("{}\n".format(word))
            else:
                f.write(word)
    print("- done. {} tokens".format(len(vocab)))


def get_word_vocab(datasets):
    print("Building word vocab...")
    vocab_words = set()
    for dataset in datasets:
        for _, words in dataset:
            vocab_words.update(nltk.word_tokenize(words))
    print("- done. {} words".format(len(vocab_words)))
    return vocab_words


def get_char_vocab(datasets):
    print("Building char vocab...")
    vocab_chars = set()
    for dataset in datasets:
        for chars, _ in dataset:
            vocab_chars.update(chars)
    print("- done. {} chars".format(len(vocab_chars)))
    return vocab_chars


def get_processing_chars(vocab):
    def f(clause):
        char_ids = []
        for char in clause:
            # ignore chars out of vocabulary
            if char in vocab:
                char_ids += [vocab[char]]
            else:
                raise Exception('Char "{}" is not in the vocabulary'.format(char))
        return char_ids
    return f


def get_processing_words(vocab):
    def f(clause):
        word_ids = []
        for word in nltk.word_tokenize(clause):
            if word in vocab:
                word_ids += [vocab[word]]
            else:
                raise Exception('Word {} is not in the vocabulary'.format(word))
        return word_ids
    return f


def _pad_sequences(sequences, pad_tok, max_length, go=None, eos=None):
    sequence_padded, sequence_length = [], []

    for seq in sequences:
        seq_ = []
        if go is not None:
            seq_ += [go]
        seq_ += list(seq)
        if eos is not None:
            seq_ += [eos]
        res_seq = seq_[:max_length] + [pad_tok] * max(max_length - len(seq_), 0)
        sequence_padded +=  [res_seq]
        sequence_length += [min(len(seq_), max_length)]

    return sequence_padded, sequence_length


def pad_sequences(sequences, pad_tok, go=None, eos=None):
    max_length = max(map(lambda x : len(x), sequences))
    max_length += 2
    sequence_padded, sequence_length = _pad_sequences(sequences, pad_tok, max_length, go, eos)
    return sequence_padded, sequence_length


def minibatches(data, minibatch_size):
    x_batch, y_batch = [], []
    for (x, y) in data:
        if len(x_batch) == minibatch_size:
            yield x_batch, y_batch
            x_batch, y_batch = [], []

        if type(x[0]) == tuple:
            x = zip(*x)
        x_batch += [x]
        y_batch += [y]

    if len(x_batch) != 0:
        yield x_batch, y_batch

