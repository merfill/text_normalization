import os


from .general_utils import get_logger
from .data_utils import load_vocab, get_processing_clause, get_processing_tag


class Config():
    def __init__(self, load=True):
        """Initialize hyperparameters and load vocabs

        Args:
            load_embeddings: (bool) if True, load embeddings into
                np array, else None

        """
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
        self.vocab_tags  = load_vocab(self.filename_tags)
        self.vocab_chars = load_vocab(self.filename_chars, chars=True)

        self.nchars = len(self.vocab_chars) + 1
        self.ntags = len(self.vocab_tags)

        print 'nchars: ', self.nchars
        print 'ntags: ', self.ntags

        # 2. get processing functions that map str -> id
        self.processing_clause = get_processing_clause(self.vocab_chars)
        self.processing_tag = get_processing_tag(self.vocab_tags)


    # general config
    dir_output = "results/test/"
    dir_model  = dir_output + "model.weights/"
    path_log   = dir_output + "log.txt"

    # embeddings
    dim_clause = 100
    dim_char = 100

    # dataset
    filename_dev = 'data/dev.csv'
    filename_test = 'data/test.csv'
    filename_train = 'data/train.csv'

    max_iter = None # if not None, max number of examples in Dataset

    # vocab (created from dataset with build_data.py)
    filename_tags = "data/tags.txt"
    filename_chars = "data/chars.txt"

    # training
    train_embeddings = False
    nepochs          = 15
    dropout          = 0.5
    batch_size       = 50
    lr_method        = "adam"
    lr               = 0.001
    lr_decay         = 0.9
    clip             = -1 # if negative, no clipping
    nepoch_no_imprv  = 3

    # model hyperparameters
    hidden_size_char = 128 # lstm on chars
    hidden_size_lstm = 128 # lstm on clause embeddings
    num_encoder_layers = 3
    max_length_clause = 50

    # NOTE: if both chars and crf, only 1.6x slower on GPU
    use_crf = True # if crf, training is 1.7x slower on CPU
    use_chars = True # if char embedding, training is 3.5x slower on CPU

