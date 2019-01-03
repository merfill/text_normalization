
import os
import csv
import random
from model.config import Config

data = []

# read data into memory
print 'start reading data into memory...'
with open('data/en_train.csv', 'rb') as f:
    reader = csv.DictReader(f, delimiter=',', quotechar='"')
    rows = []
    sent_id = -1
    for row in reader:
        if sent_id != row['sentence_id']:
            if sent_id != -1:
                data.append(rows)
            sent_id = row['sentence_id']
            rows = []
        rows += [row]
    data.append(rows)

# select data for dev, test and train and write
data_len = len(data)
print 'end data reading, read {} of sentences'.format(data_len)
sel_size = int(data_len * .1)
print 'select dev and test {} as 10% of data and write to files...'.format(sel_size)

data_ids = range(0, data_len)
dev_ids = {}
dev_data = []
for id in sorted(random.sample(data_ids, sel_size)):
    dev_ids[id] = 1
    dev_data.append(data[id])
data_ids = [x for x in data_ids if x not in dev_ids]

test_ids = {}
test_data = []
for id in sorted(random.sample(data_ids, sel_size)):
    test_ids[id] = 1
    test_data.append(data[id])
data_ids = [x for x in data_ids if x not in test_ids]

train_data = []
for id in data_ids:
    train_data.append(data[id])

small_train_data = []
for id in sorted(random.sample(data_ids, 10000)):
    small_train_data.append(data[id])

def write_to_file(filename, data):
    print 'write data to {} file'.format(filename)
    with open(filename, 'wb') as f:
        fieldnames = ['sentence_id', 'token_id', 'class', 'before', 'after']
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        writer.writeheader()
        nrows = 0
        for rows in data:
            for row in rows:
                writer.writerow(row)
                nrows += 1
        print 'end writing of {} rows'.format(nrows)

write_to_file(Config.filename_dev, dev_data)
write_to_file(Config.filename_test, test_data)
write_to_file(Config.filename_train, train_data)
write_to_file('data/small_train.csv', small_train_data)
