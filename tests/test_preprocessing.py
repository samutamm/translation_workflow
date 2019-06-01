from .context import translation_operators

import unittest
import tempfile
import os.path

import pickle

class PreprocessorTestCase(unittest.TestCase):

    def test_data_preprocessed(self):
        test_data_dir = "tests/test_data/cmn.txt"
        with tempfile.NamedTemporaryFile() as tmp:
            assert os.stat(tmp.name).st_size == 0, "File should be empty."
            preprocessor = translation_operators.TranslationDataPreprocessor(
                test_data_dir, tmp.name
            )
            preprocessor.preprocess()

            with open(tmp.name, 'rb') as f:
                data = pickle.load(f)
                assert "en" in data
                assert "ch" in data


if __name__ == '__main__':
    unittest.main()
