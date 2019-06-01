
import requests
import zipfile
import tempfile
from io import StringIO

class TranslationDataFetcher:

    def __init__(self, data_url, destination_dir):
        self.data_url = data_url
        self.destination_dir = destination_dir

    def fetch(self):
        r = requests.get(self.data_url, stream=True)
        z = zipfile.ZipFile(StringIO.StringIO(r.content))
        z.extractall(self.destination_dir)
