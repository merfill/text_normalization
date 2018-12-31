
from model.config import Config
from model.data_utils import *


def main():
    """Procedure to build data

    You MUST RUN this procedure. It iterates over the whole dataset (train,
    dev and test) and extract the vocabularies in terms of clauses, tags, and
    characters. Having built the vocabularies it writes them in a file. The
    writing of vocabulary in a file assigns an id (the line #) to each clause.
    It then extract the relevant GloVe vectors and stores them in a np array
    such that the i-th entry corresponds to the i-th clause in the vocabulary.


    Args:
        config: (instance of Config) has attributes like hyper-params...

    """
    # get config and processing of clauses
    config = Config(load=False)

    # Generators
    dev   = Dataset(config.filename_dev)
    test  = Dataset(config.filename_test)
    train = Dataset(config.filename_train)

    # Build tags vocab
    vocab_tags = get_tag_vocab([train, dev, test])
    vocab_tags.add(UNK)

    # Save vocab
    write_vocab(vocab_tags, config.filename_tags)


    # Build and save char vocab
    train = Dataset(config.filename_train)
    vocab_chars = get_char_vocab(train)
    write_vocab(vocab_chars, config.filename_chars)


if __name__ == "__main__":
    main()
