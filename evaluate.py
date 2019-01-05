
import csv

from model.data_utils import Dataset
from model.classification_model import ClassificationModel
from model.config import Config

def load_vocab(filename):
    try:
        d = dict()
        with open(filename) as f:
            for idx, clause in enumerate(f):
                clause = clause.strip()
                d[idx] = clause

    except IOError:
        raise MyIOError(filename)
    return d


def get_tag(tag, tag_d):
    return tag_d[tag]

def get_text(clause, char_d):
    r = []
    for c in clause:
        r.append(char_d[c])
    return r

def main():
    # create instance of config
    config = Config()

    # build model
    model = ClassificationModel(config)
    model.build()
    model.restore_session(config.dir_model)

    # create dataset
    test  = Dataset(config.filename_test, config.processing_clause, config.processing_tag, config.max_iter)

    # prepare vocabulary
    tags_d = load_vocab(config.filename_tags)
    char_d = load_vocab(config.filename_chars)

    # evaluate
    res = model.evaluate(test)
    print 'acc: ', res['acc']

    # print errors
    with open('erros.csv', "w") as f:
        fieldnames = ['label', 'class', 'text']
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        writer.writeheader()
        for l,p,c in res['errors']:
            row = {}
            row['label'] = get_tag(l, tags_d)
            row['class'] = get_tag(p, tags_d)
            row['text'] = ''.join(get_text(c, char_d))
            writer.writerow(row)

if __name__ == "__main__":
    main()
