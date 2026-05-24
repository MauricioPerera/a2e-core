import re
import json

class A2ECompiler:
    def __init__(self):
        pass

    def compile(self, dsl_text):
        """
        Compiles the human-readable A2E DSL text into a list of declarative JSON step dicts.
        """
        raw_lines = dsl_text.split("\n")
        lines = []
        for rl in raw_lines:
            rl_str = rl.strip()
            # Resiliency feature: split inline END statements (common in ultra-small models like Qwen 0.5B)
            if rl_str.upper().endswith(" END") and not (rl_str.startswith("#") or rl_str.startswith("//")):
                lines.append(rl_str[:-4].strip())
                lines.append("END")
            else:
                lines.append(rl_str)
                
        steps = []
        stack = [] # Stack to manage nested structures (IF/ELSE/LOOP)
        
        current_block = steps
        
        for line_num, line in enumerate(lines, 1):
            line = line.strip()
            # Ignore comments and empty lines
            if not line or line.startswith("#") or line.startswith("//"):
                continue
                
            # --- 1. SET command ---
            # Format: SET var_name = value
            set_match = re.match(r'^SET\s+([a-zA-Z0-9_]+)\s*=\s*(.+)$', line, re.IGNORECASE)
            if set_match:
                var_name = set_match.group(1).strip()
                raw_val = set_match.group(2).strip()
                
                # Try to parse as integer, float, json, or keep as string
                try:
                    if raw_val.startswith('"') and raw_val.endswith('"'):
                        value = raw_val[1:-1]
                    elif raw_val.startswith("'") and raw_val.endswith("'"):
                        value = raw_val[1:-1]
                    else:
                        value = json.loads(raw_val)
                except ValueError:
                    value = raw_val
                    
                current_block.append({"op": "set", "var": var_name, "value": value})
                continue
                
            # --- 2. API_CALL command ---
            # Format: API_CALL id = GET|POST url [WITH body] [HEADERS headers]
            api_match = re.match(r'^API_CALL\s+([a-zA-Z0-9_]+)\s*=\s*(GET|POST|PUT|DELETE)?\s*([^\s]+)(?:\s+WITH\s+(.+))?$', line, re.IGNORECASE)
            if api_match:
                var_id = api_match.group(1).strip()
                method = api_match.group(2) or "GET"
                url = api_match.group(3).strip().strip('"').strip("'")
                raw_body = api_match.group(4)
                
                body = None
                headers = {}
                
                if raw_body:
                    raw_body = raw_body.strip()
                    # Check if body defines headers in syntax (e.g. HEADERS {...})
                    headers_match = re.search(r'HEADERS\s+(.+)$', raw_body, re.IGNORECASE)
                    if headers_match:
                        raw_headers = headers_match.group(1).strip()
                        raw_body = raw_body[:headers_match.start()].strip()
                        try:
                            headers = json.loads(raw_headers)
                        except json.JSONDecodeError:
                            headers = raw_headers
                            
                    # Remove "WITH" remainder if empty, or try to load JSON
                    if raw_body:
                        try:
                            body = json.loads(raw_body)
                        except json.JSONDecodeError:
                            body = raw_body.strip('"').strip("'")
                            
                current_block.append({
                    "op": "api_call",
                    "id": var_id,
                    "method": method.upper(),
                    "url": url,
                    "body": body,
                    "headers": headers
                })
                continue
                
            # --- 3. FILTER_DATA command ---
            # Format 1: FILTER_DATA id = source WHERE key == val
            # Format 2: FILTER_DATA id = source WHERE query == {"$and": ...}
            filter_match = re.match(r'^FILTER_DATA\s+([a-zA-Z0-9_]+)\s*=\s*([a-zA-Z0-9_.]+)\s+WHERE\s+(query|[\w.]+)\s*(==)\s*(.+)$', line, re.IGNORECASE)
            if filter_match:
                var_id = filter_match.group(1).strip()
                source = filter_match.group(2).strip()
                q_key = filter_match.group(3).strip()
                q_val = filter_match.group(5).strip()
                
                query_dict = {}
                if q_key.lower() == "query":
                    try:
                        query_dict = json.loads(q_val)
                    except json.JSONDecodeError:
                        query_dict = {"query": q_val.strip('"').strip("'")}
                else:
                    cleaned_val = q_val.strip('"').strip("'")
                    try:
                        parsed_val = json.loads(q_val)
                        if isinstance(parsed_val, (int, float, bool)):
                            cleaned_val = parsed_val
                    except json.JSONDecodeError:
                        pass
                    query_dict = {q_key: cleaned_val}
                
                current_block.append({
                    "op": "filter",
                    "id": var_id,
                    "source": source,
                    "query": query_dict
                })
                continue
                
            # --- 3b. SEMANTIC_SEARCH command ---
            # Format: SEMANTIC_SEARCH var_id = collection QUERY "query_text" [LIMIT limit] [FILTER filter_json]
            semantic_match = re.match(r'^SEMANTIC_SEARCH\s+([a-zA-Z0-9_]+)\s*=\s*([a-zA-Z0-9_]+)\s+QUERY\s+(.+?)(?:\s+LIMIT\s+(\d+))?(?:\s+FILTER\s+(.+))?$', line, re.IGNORECASE)
            if semantic_match:
                var_id = semantic_match.group(1).strip()
                collection = semantic_match.group(2).strip()
                query_text = semantic_match.group(3).strip().strip('"').strip("'")
                raw_limit = semantic_match.group(4)
                raw_filter = semantic_match.group(5)
                
                limit = int(raw_limit) if raw_limit else 5
                filter_dict = {}
                if raw_filter:
                    try:
                        filter_dict = json.loads(raw_filter.strip())
                    except json.JSONDecodeError:
                        pass
                
                current_block.append({
                    "op": "semantic_search",
                    "id": var_id,
                    "collection": collection,
                    "query": query_text,
                    "limit": limit,
                    "filter": filter_dict
                })
                continue
                
            # --- 3c. MCP_SEARCH command ---
            # Format: MCP_SEARCH var_id = "query_text" [LIMIT limit] [SERVER "url"] [TYPES types_json]
            mcp_match = re.match(r'^MCP_SEARCH\s+([a-zA-Z0-9_]+)\s*=\s*(.+?)(?:\s+LIMIT\s+(\d+))?(?:\s+SERVER\s+(.+?))?(?:\s+TYPES\s+(.+))?$', line, re.IGNORECASE)
            if mcp_match:
                var_id = mcp_match.group(1).strip()
                query_text = mcp_match.group(2).strip().strip('"').strip("'")
                raw_limit = mcp_match.group(3)
                raw_server = mcp_match.group(4)
                raw_types = mcp_match.group(5)
                
                limit = int(raw_limit) if raw_limit else 5
                server_url = raw_server.strip().strip('"').strip("'") if raw_server else None
                allowed_types = []
                if raw_types:
                    try:
                        allowed_types = json.loads(raw_types.strip())
                    except json.JSONDecodeError:
                        pass
                
                current_block.append({
                    "op": "mcp_search",
                    "id": var_id,
                    "query": query_text,
                    "limit": limit,
                    "serverUrl": server_url,
                    "allowedTypes": allowed_types
                })
                continue
                
            # --- 3d. DISCOVER_SKILLS command ---
            # Format: DISCOVER_SKILLS var_id = "source_url"
            disc_match = re.match(r'^DISCOVER_SKILLS\s+([a-zA-Z0-9_]+)\s*=\s*(.+)$', line, re.IGNORECASE)
            if disc_match:
                var_id = disc_match.group(1).strip()
                source_url = disc_match.group(2).strip().strip('"').strip("'")
                current_block.append({
                    "op": "discover_skills",
                    "id": var_id,
                    "source": source_url
                })
                continue
                
            # --- 3e. DOWNLOAD_SKILL command ---
            # Format: DOWNLOAD_SKILL var_id = "skill_url"
            down_match = re.match(r'^DOWNLOAD_SKILL\s+([a-zA-Z0-9_]+)\s*=\s*(.+)$', line, re.IGNORECASE)
            if down_match:
                var_id = down_match.group(1).strip()
                skill_url = down_match.group(2).strip().strip('"').strip("'")
                current_block.append({
                    "op": "download_skill",
                    "id": var_id,
                    "url": skill_url
                })
                continue
                
            # --- 4. IF / THEN command ---
            # Format: IF condition THEN
            if_match = re.match(r'^IF\s+(.+)\s+THEN$', line, re.IGNORECASE)
            if if_match:
                condition = if_match.group(1).strip()
                node = {
                    "op": "conditional",
                    "condition": condition,
                    "then": [],
                    "else": []
                }
                current_block.append(node)
                stack.append((current_block, node, "then"))
                current_block = node["then"]
                continue
                
            # --- 5. ELSE command ---
            # Format: ELSE
            if line.upper() == "ELSE":
                if not stack or stack[-1][1]["op"] != "conditional":
                    raise SyntaxError(f"Linea {line_num}: 'ELSE' sin un 'IF' correspondiente.")
                
                parent_block, node, phase = stack[-1]
                stack[-1] = (parent_block, node, "else")
                current_block = node["else"]
                continue
                
            # --- 6. LOOP command ---
            # Format: LOOP source AS iterator
            loop_match = re.match(r'^LOOP\s+([a-zA-Z0-9_.]+)\s+AS\s+([a-zA-Z0-9_]+)$', line, re.IGNORECASE)
            if loop_match:
                source = loop_match.group(1).strip()
                iterator = loop_match.group(2).strip()
                node = {
                    "op": "loop",
                    "source": source,
                    "as": iterator,
                    "steps": []
                }
                current_block.append(node)
                stack.append((current_block, node, "steps"))
                current_block = node["steps"]
                continue
                
            # --- 7. END command ---
            # Format: END
            if line.upper() == "END":
                if not stack:
                    raise SyntaxError(f"Linea {line_num}: 'END' sin un bloque de control correspondiente.")
                
                parent_block, node, phase = stack.pop()
                current_block = parent_block
                continue
                
            raise SyntaxError(f"Linea {line_num}: Sintaxis no reconocida: '{line}'")
            
        if stack:
            # Resiliency feature: auto-close remaining control structures (very common in ultra-small models like Granite 350M)
            while stack:
                parent_block, node, phase = stack.pop()
            
        return steps
