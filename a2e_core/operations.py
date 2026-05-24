import re
import json

def get_nested_value(state, path):
    """
    Retrieves a nested value from the state dictionary using a dot-separated path.
    Example: path="get_user.posts" -> state["get_user"]["posts"]
    """
    parts = path.split('.')
    current = state
    for part in parts:
        if isinstance(current, dict) and part in current:
            current = current[part]
        elif isinstance(current, list):
            try:
                idx = int(part)
                current = current[idx]
            except (ValueError, IndexError):
                return None
        else:
            return None
    return current

def resolve_templates(val, state):
    """
    Recursively resolves template variables formatted as {{variable_name}} or {{nested.key}}
    using values from the execution state.
    """
    if isinstance(val, str):
        # Check if the string is EXACTLY a single template variable
        # If so, return the actual object (not just stringified)
        exact_match = re.match(r'^\{\{([^}]+)\}\}$', val.strip())
        if exact_match:
            path = exact_match.group(1).strip()
            return get_nested_value(state, path)
            
        # Otherwise, perform standard inline string substitution
        def replace(match):
            path = match.group(1).strip()
            res = get_nested_value(state, path)
            return str(res) if res is not None else ""
            
        return re.sub(r'\{\{([^}]+)\}\}', replace, val)
        
    elif isinstance(val, dict):
        return {k: resolve_templates(v, state) for k, v in val.items()}
    elif isinstance(val, list):
        return [resolve_templates(item, state) for item in val]
    return val

def execute_set(step, state, engine):
    var_name = step.get("var")
    value = resolve_templates(step.get("value"), state)
    state[var_name] = value
    return f"Establecido {var_name} = {value}"

def execute_api_call(step, state, engine):
    import urllib.request
    import urllib.error
    
    url = resolve_templates(step.get("url"), state)
    method = step.get("method", "GET").upper()
    headers = resolve_templates(step.get("headers", {}), state)
    body = resolve_templates(step.get("body"), state)
    var_id = step.get("id")
    
    # Mocking standard test APIs so the offline demo works perfectly
    if "api.test" in url:
        mock_response = handle_mock_api(url, method, body)
        state[var_id] = mock_response
        return f"API_CALL (MOCK) [{method}] {url} -> 200 OK (guardado en '{var_id}')"
        
    # Real HTTP execution (safe and handles content-types)
    try:
        data = None
        if body is not None:
            if isinstance(body, (dict, list)):
                data = json.dumps(body).encode('utf-8')
                if 'Content-Type' not in headers:
                    headers['Content-Type'] = 'application/json'
            else:
                data = str(body).encode('utf-8')
                
        req = urllib.request.Request(url, data=data, headers=headers, method=method)
        with urllib.request.urlopen(req, timeout=5) as response:
            res_body = response.read().decode('utf-8')
            try:
                parsed_res = json.loads(res_body)
            except json.JSONDecodeError:
                parsed_res = res_body
                
            state[var_id] = parsed_res
            return f"API_CALL [{method}] {url} -> {response.status} OK (guardado en '{var_id}')"
            
    except urllib.error.HTTPError as e:
        state[var_id] = {"error": True, "status": e.code, "message": e.read().decode('utf-8')}
        return f"API_CALL [{method}] {url} -> ERROR {e.code}"
    except Exception as e:
        state[var_id] = {"error": True, "message": str(e)}
        return f"API_CALL [{method}] {url} -> EXCEPCION: {e}"

from .query_engine import match_filter

def execute_filter(step, state, engine):
    var_id = step.get("id")
    source_path = step.get("source")
    query = step.get("query", {})
    
    source_data = get_nested_value(state, source_path)
    if not isinstance(source_data, list):
        state[var_id] = []
        return f"FILTER_DATA -> El origen '{source_path}' no es una lista o no existe."
        
    # Resolve templates in the entire MongoDB query structure
    resolved_query = resolve_templates(query, state)
    
    filtered = []
    for item in source_data:
        if not isinstance(item, dict):
            continue
        # Evaluate item matching against the full MongoDB-like query
        if match_filter(item, resolved_query):
            filtered.append(item)
            
    state[var_id] = filtered
    return f"FILTER_DATA -> Filtrados {len(filtered)} elementos de {len(source_data)} (guardado en '{var_id}')"

def execute_conditional(step, state, engine):
    condition = step.get("condition", "")
    then_steps = step.get("then", [])
    else_steps = step.get("else", [])
    
    # Safely evaluate condition using template resolution
    # Instead of eval, we do a simple safe evaluation (like comparison)
    is_true = evaluate_safe_condition(condition, state)
    
    result_msg = f"CONDITIONAL [Condicion: {condition} -> {is_true}]"
    engine.log(result_msg)
    
    steps_to_run = then_steps if is_true else else_steps
    for nested_step in steps_to_run:
        engine.execute_step(nested_step)
        
    return f"CONDITIONAL finalizado."

def execute_loop(step, state, engine):
    source_path = step.get("source")
    iterator_var = step.get("as", "item")
    loop_steps = step.get("steps", [])
    
    source_list = get_nested_value(state, source_path)
    if not isinstance(source_list, list):
        return f"LOOP -> El origen '{source_path}' no es una lista."
        
    engine.log(f"Iniciando LOOP sobre '{source_path}' ({len(source_list)} elementos)")
    
    for idx, item in enumerate(source_list):
        engine.log(f"  [Iteracion {idx+1}/{len(source_list)}]")
        # Set the iterator variable in the state
        state[iterator_var] = item
        for nested_step in loop_steps:
            engine.execute_step(nested_step)
            
    # Clean up iterator variable
    if iterator_var in state:
        del state[iterator_var]
        
    return f"LOOP finalizado."

def detect_collection_dim(db_dir, collection_name, default_dim=4):
    import os
    json_path = os.path.join(db_dir, collection_name, f"{collection_name}.json")
    if os.path.exists(json_path):
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                manifest = json.load(f)
                return manifest.get("dim", default_dim)
        except Exception:
            pass
    return default_dim

def execute_semantic_search(step, state, engine):
    import os
    from .vector_store import LocalVectorStore
    
    var_id = step.get("id")
    collection = resolve_templates(step.get("collection"), state)
    query_text = resolve_templates(step.get("query"), state)
    limit = step.get("limit", 5)
    filter_dict = resolve_templates(step.get("filter", {}), state)
    
    # 1. Resolve vector store or directory
    db_dir = state.get("vector_db_dir", r"D:\.lmstudio\a2e_vector_db")
    
    # Detect dimension dynamically
    dim = detect_collection_dim(db_dir, collection, default_dim=4)
    
    # 2. Get or instantiate vector store
    v_store = state.get("vector_store")
    if not v_store:
        v_store = LocalVectorStore(db_dir=db_dir, dim=dim)
        state["vector_store"] = v_store
    
    # Ensure the store dimension is updated if collection dimension differs
    v_store.dim = dim
    
    # 3. Generate query vector (real LM Studio embeddings or smart mock fallback)
    query_vector = None
    if dim == 4:
        # Smart keyword matching for 4D demo
        q_lower = query_text.lower()
        if any(w in q_lower for w in ["acelerar", "deep learning", "intel", "gpu", "directml"]):
            query_vector = [0.1, 0.95, 0.0, 0.0]
        elif any(w in q_lower for w in ["a2e", "protocolo", "agent", "ejecucion", "nota"]):
            query_vector = [0.95, 0.05, 0.0, 0.0]
        elif any(w in q_lower for w in ["pasta", "cocina", "italiana", "recetario", "receta"]):
            query_vector = [0.0, 0.0, 1.0, 0.0]
        else:
            query_vector = [0.5, 0.5, 0.0, 0.0]
    else:
        # Check cache first!
        import hashlib
        from .doc_store import LocalDocStore
        text_hash = hashlib.sha256(query_text.encode("utf-8")).hexdigest()
        doc_db_dir = state.get("doc_db_dir", r"D:\.lmstudio\a2e_db")
        
        cache_store = None
        try:
            cache_store = LocalDocStore(doc_db_dir)
            cache_col = cache_store.collection("embeddings_cache")
            cached = cache_col.findOne({"hash": text_hash})
            if cached and isinstance(cached.get("embedding"), list) and len(cached["embedding"]) == dim:
                query_vector = cached["embedding"]
                engine.log(f"[CACHE HIT] Usando vector de embeddings cacheado para '{query_text}'")
        except Exception:
            pass
            
        if not query_vector:
            # Query LM studio embeddings endpoint at port 1234
            import urllib.request
            import urllib.error
            try:
                req_data = json.dumps({"input": query_text}).encode("utf-8")
                req = urllib.request.Request(
                    "http://127.0.0.1:1234/v1/embeddings",
                    data=req_data,
                    headers={"Content-Type": "application/json"},
                    method="POST"
                )
                with urllib.request.urlopen(req, timeout=3) as resp:
                    res_body = json.loads(resp.read().decode("utf-8"))
                    data = res_body.get("data", [])
                    if data and isinstance(data, list):
                        emb = data[0].get("embedding")
                        if emb:
                            query_vector = emb
                            # Save to cache!
                            if cache_store:
                                try:
                                    cache_col.delete({"hash": text_hash})
                                    cache_col.insert({
                                        "hash": text_hash,
                                        "text": query_text,
                                        "embedding": query_vector
                                    })
                                    engine.log(f"[CACHE STORE] Guardado vector de embeddings para '{query_text}' en cache local")
                                except Exception:
                                    pass
            except Exception:
                # Fallback to random/mock unit vector of length dim if server is down
                import random
                random.seed(hash(query_text))
                raw_vec = [random.random() for _ in range(dim)]
                norm = sum(x*x for x in raw_vec)**0.5
                query_vector = [x/norm for x in raw_vec] if norm > 0 else [0.0]*dim
            
    # 4. Perform vector search
    results = v_store.search(
        col_name=collection,
        query_vector=query_vector,
        limit=limit,
        filter_query=filter_dict
    )
    
    state[var_id] = results
    return f"SEMANTIC_SEARCH [{collection}] QUERY '{query_text}' -> Encontrados {len(results)} resultados (guardado en '{var_id}')"

def execute_mcp_search(step, state, engine):
    import urllib.request
    import urllib.error
    
    var_id = step.get("id")
    query_text = resolve_templates(step.get("query"), state)
    limit = step.get("limit", 5)
    server_url = resolve_templates(step.get("serverUrl"), state)
    allowed_types = resolve_templates(step.get("allowedTypes", []), state)
    
    mcp_endpoint = "http://127.0.0.1:9003/search"
    
    payload = {
        "query": query_text,
        "k": limit,
    }
    if server_url:
        payload["serverUrl"] = server_url
    if allowed_types:
        payload["allowedTypes"] = allowed_types
        
    try:
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            mcp_endpoint,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST"
        )
        with urllib.request.urlopen(req, timeout=3) as resp:
            res_body = resp.read().decode("utf-8")
            parsed_res = json.loads(res_body)
            state[var_id] = parsed_res
            return f"MCP_SEARCH QUERY '{query_text}' -> {len(parsed_res)} resultados devueltos por MCPARDF_CLIENT (guardado en '{var_id}')"
    except Exception as e:
        # Fallback to mock search in offline demo mode gracefully
        mock_tools = []
        q_lower = query_text.lower()
        if "amazon" in q_lower or "precio" in q_lower or "extraer" in q_lower:
            mock_tools = [
                {
                    "id": "amazon_price_tracker",
                    "name": "Amazon Price Tracker",
                    "description": "Extrae precios e informacion de productos de Amazon a partir de un ASIN o URL.",
                    "type": "tool",
                    "serverUrl": "http://127.0.0.1:9001"
                },
                {
                    "id": "web_scraper",
                    "name": "General Web Scraper",
                    "description": "Extrae contenido HTML y lo convierte a Markdown.",
                    "type": "tool",
                    "serverUrl": "http://127.0.0.1:9001"
                }
            ]
        else:
            mock_tools = [
                {
                    "id": "calculator_tool",
                    "name": "Calculator",
                    "description": "Realiza calculos aritmeticos avanzados.",
                    "type": "tool",
                    "serverUrl": "http://127.0.0.1:9002"
                }
            ]
        
        # Filter mock tools by limit and allowedTypes
        if allowed_types:
            mock_tools = [t for t in mock_tools if t.get("type") in allowed_types]
        mock_results = mock_tools[:limit]
        
        state[var_id] = mock_results
        return f"MCP_SEARCH (MOCK-FALLBACK) QUERY '{query_text}' -> Fallo conexion a {mcp_endpoint} ({e}). Devueltos {len(mock_results)} mock-tools (guardado en '{var_id}')"

# --- Helper Functions ---

def evaluate_safe_condition(condition_str, state):
    """
    Evaluates a simple condition safely without eval.
    Examples:
      "len(active_posts) > 0"
      "user_role == admin"
    """
    # 1. Resolve templates in the condition string
    resolved = resolve_templates(condition_str, state)
    
    # 2. Support len(...) checks
    len_match = re.match(r'^len\(([^)]+)\)\s*(>|<|==|!=)\s*(\d+)$', condition_str.strip())
    if len_match:
        path = len_match.group(1).strip()
        op = len_match.group(2)
        val = int(len_match.group(3))
        
        arr = get_nested_value(state, path)
        length = len(arr) if isinstance(arr, list) else 0
        
        if op == ">": return length > val
        if op == "<": return length < val
        if op == "==": return length == val
        if op == "!=": return length != val
        
    # 3. Support simple exact matches/comparisons
    comp_match = re.match(r'^([^{\s]+)\s*(==|!=)\s*([^{\s]+)$', resolved.strip())
    if comp_match:
        left = comp_match.group(1).strip().strip('"').strip("'")
        op = comp_match.group(2)
        right = comp_match.group(3).strip().strip('"').strip("'")
        
        if op == "==": return left == right
        if op == "!=": return left != right
        
    # Fallback to boolean check of resolved string
    return bool(resolved and resolved.lower() not in ("false", "0", "null", "none"))

def handle_mock_api(url, method, body):
    """
    Simulates mock responses for testing n8n/WordPress workflows completely offline!
    """
    if "users/42" in url:
        return {
            "id": 42,
            "name": "Mauricio Perera",
            "role": "admin",
            "posts": [
                {"id": 101, "title": "Protocolo A2E: Agent-to-Execution", "status": "active"},
                {"id": 102, "title": "Optimizacion de LLMs locales en Intel Arc", "status": "active"},
                {"id": 103, "title": "Borrador de diseno obsoleto", "status": "draft"}
            ]
        }
    elif "notify" in url:
        return {"status": "success", "message": "Notificacion enviada correctamente", "received_data": body}
        
    return {"status": "default_mock", "url": url, "method": method}
