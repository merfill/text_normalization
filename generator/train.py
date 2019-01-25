from model.data_utils import Dataset
from model.model import GenModel
from model.config import Config


def main():
    # create instance of config
    config = Config(name='DATE')

    # build model
    model = GenModel(config)
    model.build()

    # create datasets
    dev = Dataset(config.filename_dev, labels=['DATE'], processing_chars=config.processing_chars, processing_words=config.processing_words)
    print 'load dev: {}'.format(len(dev))
    train = Dataset(config.filename_train, labels=['DATE'], processing_chars=config.processing_chars, processing_words=config.processing_words)
    print 'load train: {}'.format(len(train))

    # train model
    model.train(train, dev)

if __name__ == "__main__":
    main()
