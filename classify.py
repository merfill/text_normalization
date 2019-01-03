
import csv
from model.classification_model import ClassificationModel
from model.config import Config


def main():
    # create instance of config
    config = Config()

    # build model
    model = ClassificationModel(config)
    model.build()
    model.restore_session(config.dir_model)

    # read sentences
    sents = []
    data = []
    print 'start reading en_test.csv into memory...'
    with open('data/en_test.csv', 'rb') as f:
        reader = csv.DictReader(f, delimiter=',', quotechar='"')
        rows = []
        sent = []
        sent_id = -1
        for row in reader:
            if sent_id != row['sentence_id']:
                if sent_id != -1:
                    data.append(rows)
                    sents.append(sent)
                sent_id = row['sentence_id']
                rows = []
                sent = []
            rows += [row]
            sent += [row['before']]
        data.append(rows)
        sents.append(sent)
    print 'end reading en_test.csv, read {} sentences'.format(len(sents))

    # predict clause labels
    print 'start prediction...'
    labels = []
    for sent in sents:
        labels += [model.predict(sent)]
    print 'end prediction'

    # write labels within data
    filename = 'data/en_test_with_labels.csv'
    print 'write data to {} file'.format(filename)
    with open(filename, 'wb') as f:
        fieldnames = ['sentence_id', 'token_id', 'class', 'before']
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        writer.writeheader()
        nrows = 0
        for i, rows in enumerate(data):
            for j, row in enumerate(rows):
                row['class'] = labels[i][j]
                writer.writerow(row)
                nrows += 1
        print 'end writing of {} rows'.format(nrows)


if __name__ == "__main__":
    main()
