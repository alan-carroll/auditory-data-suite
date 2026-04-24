import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from db_adapter import JSONStore


class JSONStoreUpdateTests(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.db = JSONStore(Path(self.tmpdir.name) / "subject.json")
        self.coll = self.db.densetc_analysis

    def tearDown(self):
        self.db.close()
        self.tmpdir.cleanup()

    def test_update_doc_uses_tinydb_doc_id(self):
        self.coll.insert_many([
            {"_id": "a", "number": 1, "marked": False},
            {"_id": "b", "number": 2, "marked": False},
        ])
        doc = self.coll.find_one({"_id": "b"})

        self.coll.update_doc(doc.doc_id, {"$set": {"marked": True}})

        self.assertFalse(self.coll.find_one({"_id": "a"})["marked"])
        self.assertTrue(self.coll.find_one({"_id": "b"})["marked"])

    def test_update_many_by_doc_ids_applies_distinct_updates_once(self):
        self.coll.insert_many([
            {"_id": "a", "number": 1, "marked": False},
            {"_id": "b", "number": 2, "field_assignment": ""},
        ])
        first = self.coll.find_one({"_id": "a"})
        second = self.coll.find_one({"_id": "b"})
        update_table = self.coll._table._update_table
        self.coll._table._update_table = Mock(wraps=update_table)

        updated_ids = self.coll.update_many_by_doc_ids([
            (first.doc_id, {"marked": True}),
            (second.doc_id, {"field_assignment": "A1"}),
        ])

        self.assertEqual(updated_ids, [first.doc_id, second.doc_id])
        self.assertEqual(self.coll._table._update_table.call_count, 1)
        self.assertTrue(self.coll.find_one({"_id": "a"})["marked"])
        self.assertEqual(
            self.coll.find_one({"_id": "b"})["field_assignment"], "A1")

    def test_update_many_by_doc_ids_merges_duplicate_doc_ids(self):
        self.coll.insert_one({"_id": "a", "marked": False})
        doc = self.coll.find_one({"_id": "a"})

        self.coll.update_many_by_doc_ids([
            (doc.doc_id, {"marked": True}),
            (doc.doc_id, {"field_assignment": "AAF"}),
        ])

        updated = self.coll.find_one({"_id": "a"})
        self.assertTrue(updated["marked"])
        self.assertEqual(updated["field_assignment"], "AAF")


if __name__ == "__main__":
    unittest.main()
