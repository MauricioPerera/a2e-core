import os
import json
import uuid
import copy
from .query_engine import (
    match_filter, 
    _get_nested_value, 
    _set_nested_value, 
    _delete_nested_value
)

class Collection:
    def __init__(self, name, db_dir):
        self.name = name
        self.dir = os.path.join(db_dir, name)
        os.makedirs(self.dir, exist_ok=True)
        self._cache = {} # id -> doc
        self._load_all()

    def _load_all(self):
        for filename in os.listdir(self.dir):
            if filename.endswith(".json"):
                path = os.path.join(self.dir, filename)
                try:
                    with open(path, 'r', encoding='utf-8') as f:
                        doc = json.load(f)
                        self._cache[doc["_id"]] = doc
                except Exception:
                    pass

    def _persist(self, doc):
        doc_id = doc["_id"]
        path = os.path.join(self.dir, f"{doc_id}.json")
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(doc, f, indent=2, ensure_ascii=False)

    def _delete_file(self, doc_id):
        path = os.path.join(self.dir, f"{doc_id}.json")
        if os.path.exists(path):
            os.remove(path)

    def insert(self, doc):
        doc = copy.deepcopy(doc)
        if "_id" not in doc:
            doc["_id"] = str(uuid.uuid4())
            
        self._cache[doc["_id"]] = doc
        self._persist(doc)
        return doc

    def find(self, query_filter=None):
        results = []
        for doc in self._cache.values():
            if match_filter(doc, query_filter):
                results.append(copy.deepcopy(doc))
        return results

    def findOne(self, query_filter=None):
        for doc in self._cache.values():
            if match_filter(doc, query_filter):
                return copy.deepcopy(doc)
        return None

    def count(self, query_filter=None):
        c = 0
        for doc in self._cache.values():
            if match_filter(doc, query_filter):
                c += 1
        return c

    def delete(self, query_filter=None):
        to_delete = []
        for doc_id, doc in list(self._cache.items()):
            if match_filter(doc, query_filter):
                to_delete.append(doc_id)
                
        for doc_id in to_delete:
            del self._cache[doc_id]
            self._delete_file(doc_id)
            
        return len(to_delete)

    def update(self, query_filter, update_spec):
        """
        Updates documents matching query_filter using MongoDB-like operators:
        $set, $unset, $inc, $push, $pull, $rename
        """
        matched_docs = []
        for doc_id, doc in list(self._cache.items()):
            if match_filter(doc, query_filter):
                matched_docs.append(doc_id)
                
        count = 0
        for doc_id in matched_docs:
            doc = self._cache[doc_id]
            updated_doc = self._apply_update(doc, update_spec)
            self._cache[doc_id] = updated_doc
            self._persist(updated_doc)
            count += 1
            
        return count

    def _apply_update(self, doc, update_spec):
        result = copy.deepcopy(doc)
        for op, fields in update_spec.items():
            if op == "$set":
                for k, v in fields.items():
                    _set_nested_value(result, k, v)
            elif op == "$unset":
                for k in fields.keys():
                    _delete_nested_value(result, k)
            elif op == "$inc":
                for k, v in fields.items():
                    cur = _get_nested_value(result, k) or 0
                    _set_nested_value(result, k, cur + v)
            elif op == "$push":
                for k, v in fields.items():
                    arr = _get_nested_value(result, k)
                    if isinstance(arr, list):
                        arr.append(v)
                    else:
                        _set_nested_value(result, k, [v])
            elif op == "$pull":
                for k, v in fields.items():
                    arr = _get_nested_value(result, k)
                    if isinstance(arr, list) and v in arr:
                        arr.remove(v)
            elif op == "$rename":
                for old_key, new_key in fields.items():
                    val = _get_nested_value(result, old_key)
                    if val is not None:
                        _set_nested_value(result, new_key, val)
                        _delete_nested_value(result, old_key)
        return result

class LocalDocStore:
    def __init__(self, db_dir=r"D:\.lmstudio\a2e_db"):
        self.db_dir = db_dir
        os.makedirs(db_dir, exist_ok=True)
        self.collections = {}

    def collection(self, name):
        if name not in self.collections:
            self.collections[name] = Collection(name, self.db_dir)
        return self.collections[name]
