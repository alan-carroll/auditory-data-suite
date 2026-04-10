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
    coll.find_one({"configuration": {"$exists": True}})
    coll.update_one({"_id": x}, {"$set": {...}})

Not supported: insert, delete, $gt/$lt/$in/$ne, nested paths, projections, sort.
Add them if something new needs them.
"""
from functools import reduce
from tinydb import TinyDB, Query


class JSONStore:
    def __init__(self, path):
        # Unlike TinyMongoClient, which took a directory and derived the
        # filename from the "database" name, we just take the full path.
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
