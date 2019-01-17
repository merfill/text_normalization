from model.data_utils import Dataset
from model.classification_model import ClassificationModel
from model.config import Config


def main():
    # create instance of config
    config = Config()

    # build model
    model = ClassificationModel(config)
    model.build()
    # model.restore_session("results/crf/model.weights/") # optional, restore weights
    # model.reinitialize_weights("proj")

    # create datasets
    dev   = Dataset(config.filename_dev, config.processing_clause, config.processing_tag, config.max_iter)
    train = Dataset(config.filename_train, config.processing_clause, config.processing_tag, config.max_iter)

    # train model
    model.train(train, dev)

if __name__ == "__main__":
    main()
