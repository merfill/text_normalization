
from model.config import Config
from model.data_utils import *


def main():
    # get config and processing of clauses
    model_name = 'DATE'
    config = Config(name=model_name, load=False)

    # Generators
    labels = ['DATE']
    dev   = Dataset(config.filename_dev, labels=labels)
    test  = Dataset(config.filename_test, labels=labels)
    train = Dataset(config.filename_train, labels=labels)

    # Build and save words vocab
    vocab_words = get_word_vocab([train, dev, test])
    vocab_words.add(GO)
    vocab_words.add(EOS)
    write_vocab(vocab_words, config.filename_words)


    # Build and save char vocab
    vocab_chars = get_char_vocab([train, dev, test])
    write_vocab(vocab_chars, config.filename_chars)

if __name__ == "__main__":
    main()
