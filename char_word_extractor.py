
import numpy as np
import math
import json
import codecs
from pprint import pprint
import nltk


class CharWordExtractor:

    def __init__(self, char_embedding_size, char_to_id, char_embeddings, start_char_id, end_char_id,
                  word_embedding_size, word_embeddings, word_one_hots, word_to_id, id_to_word,
                  start_word_id, end_word_id):
        self._char_embedding_size = char_embedding_size
        self._char_to_id = char_to_id
        self._char_embeddings = char_embeddings
        self._start_char_id = start_char_id
        self._end_char_id = end_char_id
        self._word_embedding_size = word_embedding_size
        self._word_embeddings = word_embeddings
        self._word_one_hots = word_one_hots
        self._word_to_id = word_to_id
        self._id_to_word = id_to_word
        self._start_word_id = start_word_id
        self._end_word_id = end_word_id


    @classmethod
    def create_from_data(cls, char_data, word_data, char_embedding_size=32, word_embedding_size=32):
        # Initialize character data
        char_to_id = {}
        for text in char_data:
            for c in text:
                if c not in char_to_id:
                    id = len(char_to_id)
                    char_to_id[c] = id
        vocab_size = len(char_to_id) + 2
        char_embeddings = np.random.random([vocab_size, char_embedding_size])
        start_char_id = vocab_size - 2
        end_char_id = vocab_size - 1

        # Initialize word data
        word_to_id = {}
        id_to_word = {}
        for text in word_data:
            for word in nltk.word_tokenize(text):
                if word not in word_to_id:
                    id = len(word_to_id)
                    word_to_id[word] = id
                    id_to_word[id] = word
        vocab_size = len(word_to_id) + 2 # Add two extra 'pseudo words': start and end
        word_one_hots = np.eye(vocab_size)
        word_embeddings = np.random.random([vocab_size, word_embedding_size])
        start_word_id = vocab_size - 2
        end_word_id = vocab_size - 1

        return cls(char_embedding_size, char_to_id, char_embeddings, start_char_id, end_char_id, 
                   word_embedding_size, word_embeddings, word_one_hots, word_to_id, id_to_word,
                   start_word_id, end_word_id)


    @classmethod
    def load_from_file(cls, file_path):
        with codecs.open(file_path, encoding='utf-8') as f:    
            data = json.load(f)

            char_embedding_size = data['char_embedding_size']
            char_to_id = data['char_to_id']
            char_embeddings = data['char_embeddings']
            start_char_id = data['start_char_id']
            end_char_id = data['end_char_id']
            word_embedding_size = data['word_embedding_size']
            word_embeddings = data['word_embeddings']
            word_one_hots = data['word_one_hots']
            word_to_id = data['word_to_id']

            id_to_word = {}
            for word_id in data['id_to_word']:
                id_to_word[int(word_id)] = data['id_to_word'][word_id]

            start_word_id = data['start_word_id']
            end_word_id = data['end_word_id']

        return cls(char_embedding_size, char_to_id, char_embeddings, start_char_id, end_char_id,
                   word_embedding_size, word_embeddings, word_one_hots, word_to_id, id_to_word,
                   start_word_id, end_word_id)


    def save(self, file_path):
        data = {}
        data['char_embedding_size'] = self._char_embedding_size
        data['char_to_id'] = self._char_to_id
        data['char_embeddings'] = self._char_embeddings.tolist()
        data['start_char_id'] = self._start_char_id
        data['end_char_id'] = self._end_char_id
        data['word_embedding_size'] = self._word_embedding_size
        data['word_embeddings'] = self._word_embeddings.tolist()
        data['word_one_hots'] = self._word_one_hots.tolist()
        data['word_to_id'] = self._word_to_id
        data['id_to_word'] = self._id_to_word
        data['start_word_id'] = self._start_word_id
        data['end_word_id'] = self._end_word_id

        with codecs.open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f)

