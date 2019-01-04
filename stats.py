
import os
import csv

# read data into memory
classes = {}
print 'start reading data into memory...'
with open('data/en_train.csv', 'rb') as f:
    reader = csv.DictReader(f, delimiter=',', quotechar='"')
    for row in reader:
        if row['class'] not in classes:
            classes[row['class']] = 0
        classes[row['class']] += 1

for cl in classes.keys():
    print '{0}: {1}'.format(cl, classes[cl])
