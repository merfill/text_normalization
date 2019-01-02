import numpy as np
mport os
import csv


# shared global variables to be imported from model also
UNK = "$UNK$"
NUM = "$NUM$"
NONE = "PLAIN"


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
    """Class that iterates over CoNLL Dataset

    __iter__ method yields a tuple (clauses, tags)
        clauses: list of raw clauses
        tags: list of raw tags

    If processing_clause and processing_tag are not None, optional preprocessing is appplied

    Example:
        ```python
        data = Dataset(filename)
        for clauses, tags in data:
            pass
        ```

    """
    def __init__(self, filename, processing_clause=None, processing_tag=None, max_iter=None):
        """
        Args:
            filename: path to the file
            processing_clauses: (optional) function that takes a clause as input
            processing_tags: (optional) function that takes a tag as input
            max_iter: (optional) max number of sentences to yield

        """
        self.filename = filename
        self.processing_clause = processing_clause
        self.processing_tag = processing_tag
        self.max_iter = max_iter
        self.length = None


    def __iter__(self):
        niter = 0
        with open(self.filename, 'rb') as f:
            reader = csv.reader(f, delimiter=',', quotechar='"')
            clauses, tags = [], []
            sent_id = -1
            for row in reader:
                if sent_id != row[0]:
                    if sent_id != -1:
                        niter += 1
                        if self.max_iter is not None and niter > self.max_iter:
                            break
                        if len(clauses) > 0:
                            yield clauses, tags
                        clauses, tags = [], []
                    sent_id = row[0]
                clause, tag = row[3], row[2]
                if self.processing_clause is not None:
                    clause = self.processing_clause(clause)
                if self.processing_tag is not None:
                    tag = self.processing_tag(tag)
                clauses += [clause]
                tags += [tag]
            if len(clauses) > 0:
                yield clauses, tags


    def __len__(self):
        """Iterates once over the corpus to set and store length"""
        if self.length is None:
            self.length = 0
            for _ in self:
                self.length += 1

        return self.length


def get_char_vocab(dataset):
    """Build char vocabulary from an iterable of datasets objects

    Args:
        dataset: a iterator yielding tuples (sentence, tags)

    Returns:
        a set of all the characters in the dataset

    """
    vocab_char = set()
    for clauses, _ in dataset:
        for clause in clauses:
            vocab_char.update(clause)

    return vocab_char


def load_vocab(filename):
    """Loads vocab from a file

    Args:
        filename: (string) the format of the file must be one clause per line.

    Returns:
        d: dict[clause] = index

    """
    try:
        d = dict()
        with open(filename) as f:
            for idx, clause in enumerate(f):
                clause = clause.strip()
                d[clause] = idx

    except IOError:
        raise MyIOError(filename)
    return d


def write_vocab(vocab, filename):
    """Writes a vocab to a file

    Writes one word per line.

    Args:
        vocab: iterable that yields word
        filename: path to vocab file

    Returns:
        write a word per line

    """
    print("Writing vocab...")
    with open(filename, "w") as f:
        for i, word in enumerate(vocab):
            if i != len(vocab) - 1:
                f.write("{}\n".format(word))
            else:
                f.write(word)
    print("- done. {} tokens".format(len(vocab)))


def get_tag_vocab(datasets):
    """Build vocabulary from an iterable of datasets objects

    Args:
        datasets: a list of dataset objects

    Returns:
        a set of all the words in the dataset

    """
    print("Building vocab...")
    vocab_tags = set()
    for dataset in datasets:
        for _, tags in dataset:
            vocab_tags.update(tags)
    print("- done. {} tags".format(len(vocab_tags)))
    return vocab_tags


def get_char_vocab(dataset):
    """Build char vocabulary from an iterable of datasets objects

    Args:
        dataset: a iterator yielding tuples (sentence, tags)

    Returns:
        a set of all the characters in the dataset

    """
    vocab_chars = set()
    for clauses, _ in dataset:
        for clause in clauses:
            vocab_chars.update(clause)
    print("- done. {} chars".format(len(vocab_chars)))
    return vocab_chars


def get_processing_clause(vocab):
    """Return lambda function that transform a clause (string) into list, of int corresponding to the ids of the clause and
    its corresponding characters.

    Args:
        vocab: dict[clause] = idx

    Returns:
        f("4 March 2014") = list of char ids

    """
    def f(clause):
        char_ids = []
        for char in clause:
            # ignore chars out of vocabulary
            if char in vocab:
                char_ids += [vocab[char]]
        return char_ids

    return f


def get_processing_tag(vocab):
    def f(tag):
        if tag in vocab:
            tag_id = vocab[tag]
        else:
            raise Exception('Tag {} is basent in the vocabulary'.format(tag))
        return tag_id

    return f


def _pad_sequences(sequences, pad_tok, max_length):
    """
    Args:
        sequences: a generator of list or tuple
        pad_tok: the char to pad with

    Returns:
        a list of list where each sublist has same length
    """
    sequence_padded, sequence_length = [], []

    for seq in sequences:
        seq = list(seq)
        seq_ = seq[:max_length] + [pad_tok] * max(max_length - len(seq), 0)
        sequence_padded +=  [seq_]
        sequence_length += [min(len(seq), max_length)]

    return sequence_padded, sequence_length


def pad_sequences(sequences, pad_tok, nlevels=1):
    """
    Args:
        sequences: a generator of list or tuple
        pad_tok: the char to pad with
        nlevels: "depth" of padding, for the case where we have characters ids

    Returns:
        a list of list where each sublist has same length

    """
    if nlevels == 1:
        max_length = max(map(lambda x : len(x), sequences))
        sequence_padded, sequence_length = _pad_sequences(sequences, pad_tok, max_length)
    elif nlevels == 2:
        max_length_clause = max([max(map(lambda x: len(x), seq)) for seq in sequences])
        sequence_padded, sequence_length = [], []
        for seq in sequences:
            # all clauses are same length now
            sp, sl = _pad_sequences(seq, pad_tok, max_length_clause)
            sequence_padded += [sp]
            sequence_length += [sl]

        max_length_sentence = max(map(lambda x : len(x), sequences))
        sequence_padded, _ = _pad_sequences(sequence_padded, [pad_tok]*max_length_clause, max_length_sentence)
        sequence_length, _ = _pad_sequences(sequence_length, 0, max_length_sentence)

    return sequence_padded, sequence_length


def minibatches(data, minibatch_size):
    """
    Args:
        data: generator of (sentence, tags) tuples
        minibatch_size: (int)

    Yields:
        list of tuples

    """
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


def get_chunk_type(tok, idx_to_tag):
    """
    Args:
        tok: id of token, ex 4
        idx_to_tag: dictionary {4: "B-PER", ...}

    Returns:
        tuple: "B", "PER"

    """
    tag_name = idx_to_tag[tok]
    tag_class = tag_name.split('-')[0]
    tag_type = tag_name.split('-')[-1]
    return tag_class, tag_type


def get_chunks(seq, tags):
    """Given a sequence of tags, group entities and their position

    Args:
        seq: [4, 4, 0, 0, ...] sequence of labels
        tags: dict["O"] = 4

    Returns:
        list of (chunk_type, chunk_start, chunk_end)

    Example:
        seq = [4, 5, 0, 3]
        tags = {"B-PER": 4, "I-PER": 5, "B-LOC": 3}
        result = [("PER", 0, 2), ("LOC", 3, 4)]

    """
    default = tags[NONE]
    idx_to_tag = {idx: tag for tag, idx in tags.items()}
    chunks = []
    chunk_type, chunk_start = None, None
    for i, tok in enumerate(seq):
        # End of a chunk 1
        if tok == default and chunk_type is not None:
            # Add a chunk.
            chunk = (chunk_type, chunk_start, i)
            chunks.append(chunk)
            chunk_type, chunk_start = None, None

        # End of a chunk + start of a chunk!
        elif tok != default:
            tok_chunk_class, tok_chunk_type = get_chunk_type(tok, idx_to_tag)
            if chunk_type is None:
                chunk_type, chunk_start = tok_chunk_type, i
            elif tok_chunk_type != chunk_type or tok_chunk_class == "B":
                chunk = (chunk_type, chunk_start, i)
                chunks.append(chunk)
                chunk_type, chunk_start = tok_chunk_type, i
        else:
            pass

    # end condition
    if chunk_type is not None:
        chunk = (chunk_type, chunk_start, len(seq))
        chunks.append(chunk)

    return chunks
