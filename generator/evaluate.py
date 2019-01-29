
from model.data_utils import Dataset
from model.model import GenModel
from model.config import Config


def main():
    # create instance of config
    config = Config(name='DATE', use_beamsearch=True)

    # build model
    model = GenModel(config)
    model.build()
    model.restore_session(config.dir_model)

    # create datasets
    test = Dataset(config.filename_test, labels=['DATE'], processing_chars=config.processing_chars, processing_words=config.processing_words)
    print 'load test: {}'.format(len(test))

    # train model
    res = model.run_evaluate(test)
    print res['acc']
    for error in res['errors']:
        print error

if __name__ == "__main__":
    main()

