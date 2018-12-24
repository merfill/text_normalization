
import numpy as np
import math
import json
import codecs
from pprint import pprint
import nltk


class CharExtractor:

    def __init__(self, embedding_size, char_to_id, embeddings, start_id, end_id, label_one_hot, id_to_label, label_to_id):
        self._embedding_size = embedding_size
        self._char_to_id = char_to_id
        self._embeddings = embeddings
        self._start_id = start_id
        self._end_id = end_id
        self._label_one_hot = label_one_hot
        self._id_to_label = id_to_label
        self._label_to_id = label_to_id


    @classmethod
    def create_from_data(cls, data, label_list, embedding_size=32):
        # Initialize labels
        label_one_hot = {}
        id_to_label = {}
        label_to_id = {}
        label_one_hot = np.eye(len(label_list))
        for i in range(len(label_list)):
            label_to_id[label_list[i]] = i
            id_to_label[i] = label_list[i]

        # Initialize character data
        char_to_id = {}
        for text in data:
            for c in text:
                if c not in char_to_id:
                    char_id = len(char_to_id)
                    char_to_id[c] = char_id
        vocab_size = len(char_to_id) + 2
        embeddings = np.random.random([vocab_size, embedding_size])
        start_id = vocab_size - 2
        end_id = vocab_size - 1

        return cls(embedding_size, char_to_id, embeddings, start_id, end_id, label_one_hot, id_to_label, label_to_id)


    @classmethod
    def load_from_file(cls, file_path):
        with codecs.open(file_path, encoding='utf-8') as f:    
            data = json.load(f)

            embedding_size = data['embedding_size']
            char_to_id = data['char_to_id']
            embeddings = data['embeddings']
            start_id = data['start_id']
            end_id = data['end_id']
            label_one_hot = data['label_one_hot']

            id_to_label = {}
            for char_id in data['id_to_label']:
                id_to_label[int(char_id)] = data['id_to_label'][char_id]
            label_to_id = data['label_to_id']

        return cls(embedding_size, char_to_id, embeddings, start_id, end_id, label_one_hot, id_to_label, label_to_id)


    def save(self, file_path):
        data = {}
        data['embedding_size'] = self._embedding_size
        data['char_to_id'] = self._char_to_id
        data['embeddings'] = self._embeddings.tolist()
        data['start_id'] = self._start_id
        data['end_id'] = self._end_id
        data['label_one_hot'] = self._label_one_hot.tolist()
        data['id_to_label'] = self._id_to_label
        data['label_to_id'] = self._label_to_id

        with codecs.open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f)

