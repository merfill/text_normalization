
from model.data_utils import Dataset
from model.classification_model import ClassificationModel
from model.config import Config


def main():
    # create instance of config
    config = Config()

    # build model
    model = ClassificationModel(config)
    model.build()
    model.restore_session(config.dir_model)

    # create dataset
    test  = Dataset(config.filename_test, config.processing_clause, config.processing_tag, config.max_iter)

    # evaluate and interact
    model.evaluate(test)


if __name__ == "__main__":
    main()
