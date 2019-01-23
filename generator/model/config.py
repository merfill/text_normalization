import os


from .general_utils import get_logger
from .data_utils import *


class Config():
    def __init__(self, name, load=True):
        """Initialize hyperparameters and load vocabs

        Args:
            load_embeddings: (bool) if True, load embeddings into
                np array, else None

        """
        self.name = name

        # general config
        self.dir_output = self.name + '/results/test/'
        self.dir_model  = self.dir_output + "model.weights/"
        self.path_log   = self.dir_output + "log.txt"

        # vocab (created from dataset with build_data.py)
        self.filename_words = 'data/{0}_words.txt'.format(self.name)
        self.filename_chars = 'data/{0}_chars.txt'.format(self.name)

        # directory for training outputs
        if not os.path.exists(self.dir_output):
            os.makedirs(self.dir_output)

        # create instance of logger
        self.logger = get_logger(self.path_log)

        # load if requested (default)
        if load:
            self.load()


    def load(self):
        """Loads vocabulary, processing functions and embeddings

        Supposes that build_data.py has been run successfully and that
        the corresponding files have been created (vocab and trimmed GloVe
        vectors)

        """
        # 1. vocabulary
        self.vocab_words, self.words_vocab = load_vocab(self.filename_words)
        self.vocab_chars, self.chars_vocab = load_vocab(self.filename_chars)

        # +1 for pad symbol with index 0
        self.char_vocab_size = len(self.vocab_chars) + 1
        self.word_vocab_size = len(self.vocab_words) + 1

        # 2. get processing functions that map str -> id
        self.processing_chars = get_processing_chars(self.vocab_chars)
        self.processing_words = get_processing_words(self.vocab_words)


    # embeddings
    word_embedding_size = 100
    char_embedding_size = 100

    beam_width = 10

    # dataset
    filename_dev = 'data/dev.csv'
    filename_test = 'data/test.csv'
    filename_train = 'data/train.csv'

    # training
    train_embeddings = False
    nepochs          = 15
    dropout          = 0.5
    batch_size       = 50
    lr_method        = "adam"
    lr               = 0.001
    lr_decay         = 0.9
    clip             = 5.
    nepoch_no_imprv  = 3

    # model hyperparameters
    num_encoder_layers = 3
    num_hidden = 128
    use_attention = True
    max_iterations = 20

