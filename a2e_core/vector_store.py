import os
import json
import uuid
import struct
import math
import heapq
from array import array
from .query_engine import match_filter

# --- 1. TOP-K HEAP (Min-Heap based on Mauricio's JS design) ---

class TopKHeap:
    def __init__(self, k):
        self.k = k
        self.data = [] # Stores tuples: (score, counter, item)
        self.counter = 0

    def push(self, item, score):
        self.counter += 1
        if len(self.data) < self.k:
            heapq.heappush(self.data, (score, self.counter, item))
        elif score > self.data[0][0]:
            heapq.heappushpop(self.data, (score, self.counter, item))

    def sorted(self):
        # Sort in descending order of scores
        sorted_data = sorted(self.data, key=lambda x: x[0], reverse=True)
        return [{"score": score, **item} for score, counter, item in sorted_data]

# --- 2. MATH UTILITIES ---

def normalize(v):
    norm = math.sqrt(sum(x * x for x in v))
    if norm == 0:
        return list(v)
    return [x / norm for x in v]

def cosine_sim(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    na = sum(x * x for x in a)
    nb = sum(y * y for y in b)
    denom = math.sqrt(na * nb)
    return dot / denom if denom > 0 else 0.0

def euclidean_dist(a, b):
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))

def dot_product(a, b):
    return sum(x * y for x, y in zip(a, b))

def manhattan_dist(a, b):
    return sum(abs(x - y) for x, y in zip(a, b))

def compute_score(a, b, metric):
    if metric == 'cosine':
        return cosine_sim(a, b)
    elif metric == 'dotProduct':
        return dot_product(a, b)
    elif metric == 'euclidean':
        return 1.0 / (1.0 + euclidean_dist(a, b))
    elif metric == 'manhattan':
        return 1.0 / (1.0 + manhattan_dist(a, b))
    return cosine_sim(a, b)

# --- 3. VECTOR STORE (Float32 with File-Persistence) ---

class LocalVectorStore:
    def __init__(self, db_dir=r"D:\.lmstudio\a2e_vector_db", dim=768, model=None):
        self.db_dir = db_dir
        self.dim = dim
        self.default_model = model
        os.makedirs(db_dir, exist_ok=True)
        self._collections = {} # name -> entry

    def _bin_file(self, col_name):
        return os.path.join(self.db_dir, col_name, f"{col_name}.bin")

    def _json_file(self, col_name):
        return os.path.join(self.db_dir, col_name, f"{col_name}.json")

    def _load(self, col_name):
        if col_name in self._collections:
            return self._collections[col_name]

        col_dir = os.path.join(self.db_dir, col_name)
        os.makedirs(col_dir, exist_ok=True)

        json_path = self._json_file(col_name)
        bin_path = self._bin_file(col_name)

        ids = []
        meta = []
        model = self.default_model

        if os.path.exists(json_path):
            try:
                with open(json_path, 'r', encoding='utf-8') as f:
                    manifest = json.load(f)
                    ids = manifest.get("ids", [])
                    meta = manifest.get("meta", [])
                    model = manifest.get("model", model)
            except Exception:
                pass

        # Load Float32 array from binary file if it exists
        vectors_list = []
        if os.path.exists(bin_path) and len(ids) > 0:
            try:
                with open(bin_path, 'rb') as f:
                    arr = array('f')
                    arr.fromfile(f, len(ids) * self.dim)
                    for i in range(len(ids)):
                        vectors_list.append(list(arr[i * self.dim : (i + 1) * self.dim]))
            except Exception:
                pass

        # Fill missing vectors if any binary read error occurred
        while len(vectors_list) < len(ids):
            vectors_list.append([0.0] * self.dim)

        id_map = {doc_id: i for i, doc_id in enumerate(ids)}
        entry = {
            "ids": ids,
            "meta": meta,
            "vectors": vectors_list,
            "id_map": id_map,
            "model": model,
            "dirty": False
        }
        self._collections[col_name] = entry
        return entry

    def set(self, col_name, id, vector, metadata=None):
        if metadata is None:
            metadata = {}
        entry = self._load(col_name)
        vector = list(vector)[:self.dim]
        if len(vector) < self.dim:
            vector += [0.0] * (self.dim - len(vector))

        if id in entry["id_map"]:
            idx = entry["id_map"][id]
            entry["vectors"][idx] = vector
            entry["meta"][idx] = metadata
        else:
            idx = len(entry["ids"])
            entry["ids"].append(id)
            entry["meta"].append(metadata)
            entry["vectors"].append(vector)
            entry["id_map"][id] = idx

        entry["dirty"] = True

    def remove(self, col_name, id):
        entry = self._load(col_name)
        if id not in entry["id_map"]:
            return False
            
        idx = entry["id_map"][id]
        entry["ids"].pop(idx)
        entry["meta"].pop(idx)
        entry["vectors"].pop(idx)
        
        # Re-build ID map
        entry["id_map"] = {doc_id: i for i, doc_id in enumerate(entry["ids"])}
        entry["dirty"] = True
        return True

    def drop(self, col_name):
        col_dir = os.path.join(self.db_dir, col_name)
        if os.path.exists(col_dir):
            import shutil
            shutil.rmtree(col_dir)
        if col_name in self._collections:
            del self._collections[col_name]

    def flush(self):
        for col_name, entry in self._collections.items():
            if not entry["dirty"]:
                continue
                
            json_path = self._json_file(col_name)
            bin_path = self._bin_file(col_name)

            # Write manifest
            manifest = {
                "ids": entry["ids"],
                "meta": entry["meta"],
                "dim": self.dim,
                "model": entry["model"]
            }
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(manifest, f, indent=2, ensure_ascii=False)

            # Write binary Float32 vectors using array module (extremely efficient)
            arr = array('f')
            for vec in entry["vectors"]:
                arr.extend(vec)
            with open(bin_path, 'wb') as f:
                arr.tofile(f)

            entry["dirty"] = False

    def get(self, col_name, id):
        entry = self._load(col_name)
        if id not in entry["id_map"]:
            return None
        idx = entry["id_map"][id]
        return {
            "id": id,
            "vector": entry["vectors"][idx],
            "metadata": entry["meta"][idx]
        }

    def has(self, col_name, id):
        return id in self._load(col_name)["id_map"]

    def count(self, col_name):
        return len(self._load(col_name)["ids"])

    def list_collections(self):
        return [d for d in os.listdir(self.db_dir) if os.path.isdir(os.path.join(self.db_dir, d))]

    def search(self, col_name, query_vector, limit=10, filter_query=None, metric='cosine'):
        """
        Hybrid vector-metadata search matching js-vector-store.
        Applies metadata filters using match_filter and ranks by similarity score.
        """
        entry = self._load(col_name)
        heap = TopKHeap(limit)
        query_vector = list(query_vector)

        for i, (doc_id, vector, metadata) in enumerate(zip(entry["ids"], entry["vectors"], entry["meta"])):
            # 1. Apply metadata filtering (from our query_engine)
            if filter_query and not match_filter(metadata, filter_query):
                continue
                
            # 2. Compute similarity score
            score = compute_score(query_vector, vector, metric)
            
            # 3. Add to Top-K Heap
            heap.push({
                "id": doc_id,
                "metadata": metadata
            }, score)

        return heap.sorted()

    def search_across(self, collections, query_vector, limit=10, metric='cosine'):
        """
        Cross-collection search with score normalization (port of Mauricio's JS design).
        Normalizes scores to [0,1] within each collection and merges using a global TopKHeap.
        """
        if not collections:
            return []
        if len(collections) == 1:
            return self.search(collections[0], query_vector, limit, metric=metric)

        per_col_results = []
        for col in collections:
            res = self.search(col, query_vector, limit, metric=metric)
            if res:
                per_col_results.append(res)

        global_heap = TopKHeap(limit)
        for res_list in per_col_results:
            scores = [r["score"] for r in res_list]
            min_score = min(scores)
            max_score = max(scores)
            score_range = max_score - min_score

            for item in res_list:
                # Normalize score to [0, 1]
                normalized = (item["score"] - min_score) / score_range if score_range > 0 else 1.0
                # Preserve original score in metadata, set normalized as score
                item["raw_score"] = item["score"]
                global_heap.push({
                    "id": item["id"],
                    "metadata": item["metadata"]
                }, normalized)

        return global_heap.sorted()
