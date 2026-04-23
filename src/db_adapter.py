# db_adapter.py
"""
Drop-in replacement for the subset of tinymongo this app uses.

Tinymongo was a thin wrapper over TinyDB, so the on-disk format is just a
TinyDB database with one table per collection. We use TinyDB directly and
recreate only the mongo-ish surface area we actually call. Existing JSON
files open unmodified; writes go to the same format.

Supported:
    db = JSONStore("/path/to/subject.json")
    coll = db.sites                                    # or db.collection("sites")
    coll.find({})                                      # all docs
    coll.find({"analysis_id": x, "number": 3})         # equality filter(s), AND-ed
    coll.find_one({...})                               # first match or None
    coll.get_only()                                    # exactly one document
    coll.clone_by("analysis_id", old_id, new_id)       # duplicate matching docs
    coll.update_one({"_id": x}, {"$set": {...}})
    get_project_config(db)                             # read-through migration

Not supported: delete, $gt/$lt/$in/$ne, nested paths, projections, sort.
Add them if something new needs them.
"""
from functools import reduce
from tinydb import TinyDB, Query
from uuid import uuid4
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class InsertOneResult:
    inserted_id: str


@dataclass(frozen=True)
class InsertManyResult:
    inserted_ids: list[str]


class JSONStore:
    def __init__(self, path):
        # Unlike TinyMongoClient, which took a directory and derived the
        # filename from the "database" name, we just take the full path.
        # Adds `.json` extension if it's missing, since this is just a
        # simple adapter layer for old tinymongo json files
        path = Path(path)
        if not path.suffix:
            path = path.with_suffix(".json")
        self._db = TinyDB(str(path))

    def collection(self, name):
        return Collection(self._db.table(name))

    def __getattr__(self, name):
        # `db.sites` → collection("sites"), same as tinymongo's attr access.
        # __getattr__ only fires for missing attrs so .close() etc. aren't
        # shadowed.
        return self.collection(name)

    def close(self):
        self._db.close()


def get_project_config(db):
    """
    Return the project-level stimulus configuration for a subject DB.

    Newer databases store it on the single metadata document.
    Older databases stored it on the frozen auto-analysis metadata doc;
    when found there, write it forward so future reads use the new home.
    """
    meta = db.metadata.get_only()
    if "project_configuration" in meta:
        return meta["project_configuration"]

    legacy = db.analysis_metadata.find_one({"configuration": {"$exists": True}})
    if legacy is None:
        raise ValueError(
            "No project configuration found in db.metadata or legacy "
            "analysis_metadata."
        )

    cfg = legacy["configuration"]
    db.metadata.update_one({"_id": meta["_id"]},
                           {"$set": {"project_configuration": cfg}})
    return cfg


class Collection:
    def __init__(self, table):
        self._table = table

    def find(self, query=None):
        if not query:
            return self._table.all()
        return self._table.search(_build_query(query))

    def find_one(self, query=None):
        if not query:
            raise ValueError("find_one() requires a non-empty filter")
        return self._table.get(_build_query(query))
    
    def get_only(self):
        docs = self._table.all()
        if len(docs) != 1:
            raise ValueError(f"Expected exactly 1 document, found {len(docs)}")
        return docs[0]

    def clone_by(self, field, old_value, new_value):
        """
        Duplicate every document where `field == old_value`, rewriting that
        field to `new_value` and dropping `_id` so fresh IDs are assigned.
        """
        clones = []
        for doc in self.find({field: old_value}):
            clone = dict(doc)
            clone[field] = new_value
            clone.pop("_id", None)
            clones.append(clone)
        return self.insert_many(clones)
    
    def insert_one(self, document):
        if not isinstance(document, dict):
            raise TypeError("insert_one() requires a dict document")

        doc = deepcopy(document)
        doc.setdefault("_id", str(uuid4())) # preserve tinymongo behavior
        self._table.insert(doc)
        return InsertOneResult(doc["_id"])
    
    def insert_many(self, documents):
        docs = list(documents)
        if not all(isinstance(doc, dict) for doc in docs):
            raise TypeError("insert_many() requires an iterable of dict documents")

        inserted_ids = []
        to_insert = []
        for document in docs:
            doc = deepcopy(document)
            doc.setdefault("_id", str(uuid4())) # preserve tinymongo behavior
            inserted_ids.append(doc["_id"])
            to_insert.append(doc)

        if to_insert:
            self._table.insert_multiple(to_insert)

        return InsertManyResult(inserted_ids)

    def update_one(self, filter_, update):
        if set(update) != {"$set"}:
            raise NotImplementedError(
                f"Only $set is supported, got {set(update)}")
        if not filter_:
            raise ValueError("update_one() requires a non-empty filter")
        # Update at most one document even if the filter matches several. 
        doc = self._table.get(_build_query(filter_))
        if doc is not None:
            self._table.update(update["$set"], doc_ids=[doc.doc_id])


def _build_query(mongo_query):
    """
    Translate a flat mongo-style filter to a TinyDB Query. Fields AND together.
    """
    if not mongo_query:
        raise ValueError("_build_query() requires a non-empty filter")
    Q = Query()
    clauses = []
    for field, val in mongo_query.items():
        if "." in field:
            raise NotImplementedError(
                f"Nested field paths are not supported: {field!r}"
            )
        if isinstance(val, dict):
            if len(val) != 1:
                raise NotImplementedError(
                    f"Only one operator per field is supported, got {val!r}"
                )
            op, arg = next(iter(val.items()))
            if op == "$exists":
                clauses.append(Q[field].exists() if arg
                               else ~Q[field].exists())
            else:
                raise NotImplementedError(f"Operator {op}")
        else:
            clauses.append(Q[field] == val)
    return reduce(lambda a, b: a & b, clauses)
