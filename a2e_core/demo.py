import json
import os
import shutil
from .compiler import A2ECompiler
from .engine import A2EEngine
from .doc_store import LocalDocStore
from .vector_store import LocalVectorStore

def run_demo():
    print("=" * 75)
    print("INTEGRACION MAESTRA DE LA TRINIDAD AGENTICA: A2E + DOC-STORE + VECTOR-STORE")
    print("=" * 75)
    
    # -------------------------------------------------------------------------
    # 1. DOC-STORE: Inicializar LocalDocStore (Persistencia de Estado)
    # -------------------------------------------------------------------------
    db_dir = r"D:\.lmstudio\a2e_db"
    print(f"\n1. Inicializando Base de Datos Documental (js-doc-store port) en: {db_dir}...")
    store = LocalDocStore(db_dir)
    sessions_col = store.collection("a2e_sessions")
    
    sessions_col.delete({}) # Limpiar
    sessions_col.insert({
        "sessionId": "session-101",
        "description": "Ejecucion de agente local actual",
        "status": "pending",
        "steps_count": 5,
        "meta": {"environment": "development", "version": "1.0.1"}
    })
    print("   [OK] Sesiones persistidas en disco en formato JSON.")
    
    # -------------------------------------------------------------------------
    # 2. VECTOR-STORE: Inicializar LocalVectorStore (Persistencia Semantica / RAG)
    # -------------------------------------------------------------------------
    vector_db_dir = r"D:\.lmstudio\a2e_vector_db"
    print(f"\n2. Inicializando Base de Datos Vectorial (js-vector-store port) en: {vector_db_dir}...")
    
    # Usaremos 4 dimensiones para simplificar los logs del demo
    v_store = LocalVectorStore(vector_db_dir, dim=4, model="gemma-300m-local")
    
    # Limpiar colecciones previas
    v_store.drop("knowledge_base")
    v_store.drop("user_notes")
    
    # Insertar documentos vectoriales en la coleccion 'knowledge_base'
    # Cada vector representa una representacion semantica de 4 dimensiones
    v_store.set("knowledge_base", "doc-1", [1.0, 0.0, 0.0, 0.0], {
        "title": "Protocolo A2E: Agent-to-Execution", 
        "category": "tech",
        "summary": "Protocolo declarativo para ejecucion de flujos de IA"
    })
    v_store.set("knowledge_base", "doc-2", [0.0, 1.0, 0.0, 0.0], {
        "title": "Aceleracion local con Intel Arc y DirectML", 
        "category": "tech",
        "summary": "Optimizacion de deep learning en tarjetas de video Intel"
    })
    v_store.set("knowledge_base", "doc-3", [0.0, 0.0, 1.0, 0.0], {
        "title": "Recetario de cocina italiana tradicional", 
        "category": "cooking",
        "summary": "Secretos para la mejor pasta fresca en casa"
    })
    
    # Insertar en otra coleccion 'user_notes' para mostrar busqueda cruzada
    v_store.set("user_notes", "note-1", [0.9, 0.1, 0.0, 0.0], {
        "title": "Notas de Mauricio: Integrar js-vector-store con A2E",
        "category": "personal"
    })
    
    # Guardar fisicamente los vectores en archivos binarios (.bin) y manifiestos (.json)
    v_store.flush()
    print("   [OK] Vectores e indices guardados exitosamente!")
    print("   Archivos binarios Float32 y JSON creados físicamente en disco:")
    for col in ["knowledge_base", "user_notes"]:
        if os.path.exists(os.path.join(vector_db_dir, col)):
            for f in os.listdir(os.path.join(vector_db_dir, col)):
                print(f"     - D:\\.lmstudio\\a2e_vector_db\\{col}\\{f}")
                
    # -------------------------------------------------------------------------
    # 3. DSL DE A2E: Definir y Compilar Flujo de Trabajo Inteligente
    # -------------------------------------------------------------------------
    print("\n" + "=" * 75)
    print("3. DEFINE Y COMPILA SCRIPT DSL DE A2E (USANDO OPERADORES AVANZADOS):")
    print("=" * 75)
    
    dsl_script = """
    # Paso 1: Configurar variables
    SET target_user = 42
    SET rag_query = "Como acelerar deep learning localmente con Intel Arc"
    SET mcp_query = "extraer precios de Amazon"
    
    # Paso 2: Consultar API de usuario (MOCK)
    API_CALL user_profile = GET "https://api.test/users/{{target_user}}"
    
    # Paso 3: Filtrar posts de forma avanzada tipo MongoDB (Filtro Inline JSON)
    FILTER_DATA filtered_posts = user_profile.posts WHERE query == {"$and": [{"status": "active"}, {"id": {"$gt": 101}}]}
    
    # Paso 4: Búsqueda Semántica RAG Nativa en LocalVectorStore
    SEMANTIC_SEARCH rag_results = knowledge_base QUERY "{{rag_query}}" LIMIT 2 FILTER {"category": "tech"}
    
    # Paso 5: Búsqueda MCP contra MCPARDF_CLIENT en puerto 9003
    MCP_SEARCH mcp_tools = "{{mcp_query}}" LIMIT 2 TYPES ["tool"]
    
    # Paso 6: Decision condicional e inyeccion de contexto
    IF len(rag_results) > 0 THEN
        SET notification_msg = "RAG: {{rag_results.0.metadata.title}}! Herramienta compatible: {{mcp_tools.0.name}}"
        API_CALL notify = POST "https://api.test/notify" WITH {"msg": "{{notification_msg}}"}
    ELSE
        SET notification_msg = "No se encontraron posts o resultados RAG."
    END
    """
    
    compiler = A2ECompiler()
    try:
        compiled_workflow = compiler.compile(dsl_script)
        print("   [OK] Flujo A2E compilado correctamente a Representacion Intermedia JSON:")
        print(json.dumps(compiled_workflow, indent=2, ensure_ascii=False))
    except Exception as e:
        print(f"Error de compilacion: {e}")
        return
        
    # -------------------------------------------------------------------------
    # 4. ENGINE DE A2E: Ejecutar el Flujo Declarativo
    # -------------------------------------------------------------------------
    print("\n" + "=" * 75)
    print("4. EJECUTANDO FLUJO DECLARATIVO CON EL A2E ENGINE:")
    print("=" * 75)
    
    initial_state = {
        "vector_store": v_store,
        "vector_db_dir": vector_db_dir
    }
    engine = A2EEngine(initial_state)
    final_state = engine.run_workflow(compiled_workflow)
    
    # -------------------------------------------------------------------------
    # 5. SEMANTICS & LOCAL RAG: Busqueda Hibrida y search_across en D:
    # -------------------------------------------------------------------------
    print("\n" + "=" * 75)
    print("5. DEMOSTRACION DE BUSQUEDA HIBRIDA Y RAG LOCAL (VECTOR-STORE):")
    print("=" * 75)
    
    # Simulamos un embedding de consulta generado por tu modelo local Gemma-300M
    # El usuario busca: "Como acelerar deep learning localmente"
    # El vector resultante apunta fuertemente hacia la dimension Y (computo/GPU)
    query_vector = [0.1, 0.95, 0.0, 0.0]
    
    print(f"   Consulta Semantica: 'Como acelerar deep learning localmente'")
    print(f"   Vector de consulta (Gemma-300M): {query_vector}")
    
    # Búsqueda híbrida: similitud vectorial + filtro de metadatos (tipo de categoría)
    # Buscamos solo en la categoría 'tech'
    filter_query = {"category": "tech"}
    print(f"   Aplicando filtro de metadatos (MongoDB-style): {json.dumps(filter_query)}")
    
    results = v_store.search("knowledge_base", query_vector, limit=2, filter_query=filter_query)
    
    print("\n   Resultados de la Busqueda Hibrida (Filtrada):")
    for idx, r in enumerate(results, 1):
        print(f"     [{idx}] ID: {r['id']} | Score: {r['score']:.4f} | Titulo: {r['metadata']['title']} | Categoria: {r['metadata']['category']}")
        
    # Búsqueda cruzada con Normalizacion de Scores a lo largo de multiples colecciones (search_across)
    # Buscaremos en 'knowledge_base' y 'user_notes'
    print("\n   Ejecutando busqueda cruzada (search_across) con normalizacion en 'knowledge_base' y 'user_notes'...")
    cross_query = [0.95, 0.05, 0.0, 0.0] # Apunta fuertemente al Protocolo A2E / notas
    
    cross_results = v_store.search_across(["knowledge_base", "user_notes"], cross_query, limit=2)
    
    print("\n   Resultados de la Busqueda Cruzada (Normalizados):")
    for idx, r in enumerate(cross_results, 1):
        print(f"     [{idx}] ID: {r['id']} | Score Normalizado: {r['score']:.4f} | Titulo: {r['metadata']['title']}")
        
    # -------------------------------------------------------------------------
    # 6. PERSISTENCIA FINAL: Guardar Auditoria en DocStore
    # -------------------------------------------------------------------------
    print("\n" + "=" * 75)
    print("6. PERSISTENCIA FINAL EN DISCO (ACTUALIZACION DE SESION):")
    print("=" * 75)
    
    # Marcar la sesion activa como completada e inyectar el mejor resultado RAG obtenido
    best_doc_title = results[0]['metadata']['title'] if results else "Ninguno"
    
    sessions_col.update(
        {"sessionId": "session-101"},
        {
            "$set": {
                "status": "completed", 
                "final_result": final_state.get("notification_msg"),
                "rag_retrieved_context": best_doc_title
            }
        }
    )
    
    final_session_doc = sessions_col.findOne({"sessionId": "session-101"})
    print("   [OK] Sesion persistida y auditada en el JSON final de js-doc-store:")
    print(json.dumps(final_session_doc, indent=2, ensure_ascii=False))
    print("=" * 75)

    # -------------------------------------------------------------------------
    # 7. LLMS-TXT-SKILLS: Descubrimiento y Descarga de Skills via llms.txt
    # -------------------------------------------------------------------------
    print("\n" + "=" * 75)
    print("7. INTEGRACION DE LLMS-TXT-SKILLS (DRAFT V0.4):")
    print("=" * 75)
    
    llms_txt_dsl = """
    # Paso 1: Configurar origen de busqueda
    SET skill_source = "https://img.automators.work"
    
    # Paso 2: Descubrir skills publicadas en el dominio
    DISCOVER_SKILLS found_skills = "{{skill_source}}"
    
    # Paso 3: Descargar la primera skill encontrada (placeholder.md)
    DOWNLOAD_SKILL placeholder_skill = "{{found_skills.0.url}}"
    """
    
    try:
        compiled_llms_txt = compiler.compile(llms_txt_dsl)
        print("   [OK] Flujo llms.txt compilado correctamente a AST JSON:")
        print(json.dumps(compiled_llms_txt, indent=2, ensure_ascii=False))
        
        print("\n   Ejecutando flujo de llms.txt con el motor A2E...")
        llms_state = {}
        llms_engine = A2EEngine(llms_state)
        llms_final_state = llms_engine.run_workflow(compiled_llms_txt)
        
        print("\n   Resultado de la Descarga y Extraccion de la Skill:")
        skill = llms_final_state.get("placeholder_skill", {})
        print(f"     - Nombre: {skill.get('name')}")
        print(f"     - Descripcion: {skill.get('description')}")
        print(f"     - Version: {skill.get('version')}")
        print(f"     - Licencia: {skill.get('license')}")
        print(f"     - Homepage: {skill.get('homepage')}")
        print(f"     - Contenido Snippet:\n   {skill.get('content')[:160]}...")
        print("=" * 75)
        
    except Exception as e:
        print(f"Error en flujo de llms.txt: {e}")

if __name__ == "__main__":
    run_demo()

