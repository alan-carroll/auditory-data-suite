import tinydb
import tinymongo

class TinyMongoClient(tinymongo.TinyMongoClient):
    @property
    def _storage(self):
        return tinydb.storages.JSONStorage
