
from __future__ import absolute_import, division, print_function, unicode_literals

import unicodedata
import re
import io
import re
import pickle

class TranslationDataPreprocessor:

    def __init__(self, text_files, destination_file, num_examples = 40000):
        self.text_files = text_files
        self.destination_file = destination_file
        self.num_examples = num_examples
        self.start_token = "_"
        self.end_token = "|"

    def preprocess(self, **kwargs):
        en, ch = self._create_dataset(self.text_files, self.num_examples)
        datasets = {"en":en, "ch":ch}
        with open(self.destination_file, 'wb+') as f:
            pickle.dump(datasets, f, pickle.HIGHEST_PROTOCOL)

    # Converts the unicode file to ascii
    def _unicode_to_ascii(self, s):
        return ''.join(c for c in unicodedata.normalize('NFD', s)
                       if unicodedata.category(c) != 'Mn')

    def _preprocess_english(self, w):
        w = self._unicode_to_ascii(w.lower().strip())

        # creating a space between a word and the punctuation following it
        # eg: "he is a boy." => "he is a boy ."
        # Reference:- https://stackoverflow.com/questions/3645931/python-padding-punctuation-with-white-spaces-keeping-punctuation
        w = re.sub(r"([?.!,¿])", r" \1 ", w)
        w = re.sub(r'[" "]+', " ", w)

        # replacing everything with space except (a-z, A-Z, ".", "?", "!", ",")
        w = re.sub(r"[^a-zA-Z?.!]+", " ", w)

        w = w.rstrip().strip()

        # adding a start and an end token to the sentence
        # so that the model know when to start and stop predicting.
        w = self.start_token + ' ' + w + ' ' + self.end_token
        return w

    def _preprocess_chinese(self, w):
        # remove numbers and start or end characters
        w = re.sub(r"[0-9_|]+", "", w)

        w = self.start_token + w + self.end_token
        return w

    def _create_dataset(self, path, num_examples):
        lines = io.open(path, encoding='UTF-8').read().strip().split('\n')

        word_pairs = [[self._preprocess_english(l.split('\t')[0]), self._preprocess_chinese(l.split('\t')[1])] for l in
                      lines[:num_examples]]

        return zip(*word_pairs)
