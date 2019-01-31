
import os
import csv
import nltk

source = set()
target = set()

def filter_and_write(input_file_path, output_file_path):
    with open(input_file_path, 'rb') as f:
        rows = []
        nrows = 0
        reader = csv.DictReader(f, delimiter=',', quotechar='"')
        for row in reader:
            if row['class'] == 'DATE':
                rows += [{'before': row['before'], 'after': row['after']}]
                nrows += 1
                source.update(row['before'])
                target.update(nltk.word_tokenize(row['after']))
    print 'read {} rows'.format(len(rows))

    with open(output_file_path, 'wb') as f:
        fieldnames = ['before', 'after']
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        writer.writeheader()
        nrows = 0
        for row in rows:
            writer.writerow(row)
            nrows += 1
    print 'write {} rows'.format(nrows)

filter_and_write('../classifier/data/train.csv', 'data/train.csv')
filter_and_write('../classifier/data/test.csv', 'data/test.csv')
filter_and_write('../classifier/data/dev.csv', 'data/dev.csv')

def write(file_name, collection):
    with open(file_name, 'w') as f:
        for item in collection:
            f.write('{}\n'.format(item))

write('data/vocab.source.txt', source)
write('data/vocab.target.txt', target)
