
from .context import translation_operators

import unittest
import tempfile
import os.path
import tempfile

class DataFetcherTestCase(unittest.TestCase):

    @unittest.skip("Polling public source is not good idea and mocking it is too much effort")
    def test_data_is_stored(self):
        with tempfile.NamedTemporaryFile() as tmp:
            assert os.stat(tmp.name).st_size == 0, "File should be empty."
            fetcher = translation_operators.TranslationDataFetcher(
                "http://www.manythings.org/anki/cmn-eng.zip",
                tmp.name
            )
            fetcher.fetch()

            assert os.stat(tmp.name).st_size > 0, "Now there should be content."


if __name__ == '__main__':
    unittest.main()
