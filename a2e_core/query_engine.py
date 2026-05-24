import re

def _get_nested_value(doc, path):
    """
    Accesses nested dict values using dot notation: 'address.city'
    """
    if not doc:
        return None
    if '.' not in path:
        return doc.get(path)
        
    parts = path.split('.')
    current = doc
    for p in parts:
        if current is None or not isinstance(current, dict):
            return None
        current = current.get(p)
    return current

def _set_nested_value(doc, path, value):
    """
    Sets nested dict values using dot notation.
    """
    if '.' not in path:
        doc[path] = value
        return
        
    parts = path.split('.')
    current = doc
    for p in parts[:-1]:
        if p not in current or current[p] is None:
            current[p] = {}
        current = current[p]
    current[parts[-1]] = value

def _delete_nested_value(doc, path):
    """
    Deletes nested dict values using dot notation.
    """
    if '.' not in path:
        if path in doc:
            del doc[path]
        return
        
    parts = path.split('.')
    current = doc
    for p in parts[:-1]:
        if p not in current or current[p] is None:
            return
        current = current[p]
    if parts[-1] in current:
        del current[parts[-1]]

def match_filter(doc, query_filter):
    """
    Python port of Mauricio Perera's js-doc-store query matching engine.
    Supports MongoDB-like operators:
    $and, $or, $not, $eq, $ne, $gt, $gte, $lt, $lte, $in, $nin, $exists, $regex, $contains, $size
    """
    if not query_filter or not isinstance(query_filter, dict):
        return True
    if not doc:
        doc = {}

    for key, cond in query_filter.items():
        if key == '$and':
            if not isinstance(cond, list):
                return False
            if not all(match_filter(doc, sub) for sub in cond):
                return False
            continue
            
        if key == '$or':
            if not isinstance(cond, list):
                return False
            if not any(match_filter(doc, sub) for sub in cond):
                return False
            continue
            
        if key == '$not':
            if match_filter(doc, cond):
                return False
            continue

        val = _get_nested_value(doc, key)

        # Direct exact match or regex
        if cond is None or not isinstance(cond, dict):
            if isinstance(cond, str) and cond.startswith('/') and cond.endswith('/'):
                # Handle mock regex representation
                regex_str = cond[1:-1]
                if not re.search(regex_str, str(val if val is not None else "")):
                    return False
            elif val != cond:
                return False
            continue

        # Operator evaluation
        for op, target in cond.items():
            if op == '$eq':
                if val != target: return False
            elif op == '$ne':
                if val == target: return False
            elif op == '$gt':
                if val is None or not (val > target): return False
            elif op == '$gte':
                if val is None or not (val >= target): return False
            elif op == '$lt':
                if val is None or not (val < target): return False
            elif op == '$lte':
                if val is None or not (val <= target): return False
            elif op == '$in':
                if not isinstance(target, list) or val not in target: return False
            elif op == '$nin':
                if isinstance(target, list) and val in target: return False
            elif op == '$exists':
                exists = val is not None
                if exists != bool(target): return False
            elif op == '$regex':
                regex_str = target.pattern if hasattr(target, 'pattern') else str(target)
                if not re.search(regex_str, str(val if val is not None else "")): return False
            elif op == '$contains':
                if not isinstance(val, list) or target not in val: return False
            elif op == '$size':
                if not isinstance(val, list) or len(val) != target: return False
            elif op == '$not':
                # Field-level negation
                sub_filter = {key: target}
                if match_filter(doc, sub_filter): return False

    return True
