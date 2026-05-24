import time
import os
import shutil
import random
from .doc_store import LocalDocStore
from .vector_store import LocalVectorStore
from .compiler import A2ECompiler
from .engine import A2EEngine

def run_benchmarks():
    print("=" * 80)
    print("                A2E TRINITY FRAMEWORK - RUNTIME BENCHMARK SUITE")
    print("=" * 80)
    
    # Setup temporary directory in D: for benchmarks to avoid touching production data
    bench_dir = r"D:\.lmstudio\a2e_bench_temp"
    if os.path.exists(bench_dir):
        try:
            shutil.rmtree(bench_dir)
        except Exception:
            pass
    os.makedirs(bench_dir, exist_ok=True)
    
    try:
        # --- 1. BENCHMARK: DOCSTORE CRUD PERFORMANCE ---
        print("\n[1/4] Evaluando Rendimiento de LocalDocStore (Persistencia JSON en Disco D:)...")
        doc_store = LocalDocStore(bench_dir)
        col = doc_store.collection("test_performance")
        
        # 1a. Writes/Inserts
        start_time = time.perf_counter()
        num_docs = 500
        for i in range(num_docs):
            col.insert({
                "index": i,
                "uuid": f"doc-{i}",
                "value": random.random(),
                "nested": {"status": "active" if i % 2 == 0 else "inactive"}
            })
        write_duration = time.perf_counter() - start_time
        write_qps = num_docs / write_duration
        print(f"  -> Escritura (Insert): {num_docs} docs en {write_duration:.4f}s ({write_qps:.2f} op/s)")
        
        # 1b. Reads (findOne with filter)
        start_time = time.perf_counter()
        num_reads = 1000
        for _ in range(num_reads):
            target_idx = random.randint(0, num_docs - 1)
            col.findOne({"index": target_idx})
        read_duration = time.perf_counter() - start_time
        read_qps = num_reads / read_duration
        print(f"  -> Lectura (findOne con filtro): {num_reads} consultas en {read_duration:.4f}s ({read_qps:.2f} op/s)")
        
        # 1c. Updates ($set with filter)
        start_time = time.perf_counter()
        num_updates = 200
        for _ in range(num_updates):
            target_idx = random.randint(0, num_docs - 1)
            col.update({"index": target_idx}, {"$set": {"updated": True, "value": random.random()}})
        update_duration = time.perf_counter() - start_time
        update_qps = num_updates / update_duration
        print(f"  -> Actualizacion (update $set): {num_updates} docs en {update_duration:.4f}s ({update_qps:.2f} op/s)")
        
        # --- 2. BENCHMARK: VECTORSTORE COSINE SIMILARITY SEARCH ---
        print("\n[2/4] Evaluando Rendimiento de LocalVectorStore (Búsqueda Lineal 768-D)...")
        v_store = LocalVectorStore(bench_dir, dim=768)
        
        # Populate vectors
        num_vectors = 1000
        print(f"  -> Indexando {num_vectors} vectores de 768 dimensiones en disco...")
        for i in range(num_vectors):
            vec = [random.random() for _ in range(768)]
            v_store.set("test_vectors", f"vec-{i}", vec, {"category": "tech" if i % 2 == 0 else "cooking"})
        v_store.flush()
        
        # Search performance
        query_vector = [random.random() for _ in range(768)]
        num_searches = 100
        start_time = time.perf_counter()
        for _ in range(num_searches):
            v_store.search("test_vectors", query_vector, limit=5)
        search_duration = time.perf_counter() - start_time
        search_qps = num_searches / search_duration
        print(f"  -> Busqueda Semantica (Fuerza Bruta 768-D): {num_searches} busquedas en {search_duration:.4f}s ({search_qps:.2f} QPS)")
        
        # --- 3. BENCHMARK: EMBEDDINGS CACHE VS COLD CALL ---
        print("\n[3/4] Evaluando Impacto del Sistema de Caché de Embeddings (operations.py)...")
        # Populate cache
        cache_col = doc_store.collection("embeddings_cache")
        import hashlib
        dummy_text = "Como optimizar redes neuronales profundas en dispositivos edge"
        dummy_hash = hashlib.sha256(dummy_text.encode("utf-8")).hexdigest()
        dummy_vector = [random.random() for _ in range(768)]
        
        cache_col.delete({"hash": dummy_hash})
        cache_col.insert({
            "hash": dummy_hash,
            "text": dummy_text,
            "embedding": dummy_vector
        })
        
        # Test Cache Hit Time
        num_cache_hits = 1000
        start_time = time.perf_counter()
        for _ in range(num_cache_hits):
            cached = cache_col.findOne({"hash": dummy_hash})
            _ = cached.get("embedding")
        cache_hit_duration = time.perf_counter() - start_time
        cache_hit_qps = num_cache_hits / cache_hit_duration
        print(f"  -> Cache Hit (findOne en DocStore): {num_cache_hits} hits en {cache_hit_duration:.4f}s ({cache_hit_qps:.2f} hits/s)")
        print(f"  -> Ahorro de Latencia Estimado vs LLM Local: ~{(0.05 * num_cache_hits - cache_hit_duration):.2f}s guardados!")

        # --- 4. BENCHMARK: END-TO-END A2E DSL WORKFLOW COMPILATION & EXECUTION ---
        print("\n[4/4] Evaluando Compilacion y Ejecucion de Flujo Completo A2E DSL...")
        dsl_script = """
        SET target_user = 42
        API_CALL user_profile = GET "https://api.test/users/{{target_user}}"
        FILTER_DATA filtered_posts = user_profile.posts WHERE query == {"status": "active"}
        IF len(filtered_posts) > 0 THEN
            SET notification = "Posts activos encontrados"
        ELSE
            SET notification = "Sin posts activos"
        END
        """
        
        # 4a. Compilation
        compiler = A2ECompiler()
        start_time = time.perf_counter()
        num_compiles = 500
        for _ in range(num_compiles):
            compiler.compile(dsl_script)
        compile_duration = time.perf_counter() - start_time
        compile_rate = num_compiles / compile_duration
        print(f"  -> Compilacion DSL a JSON IR: {num_compiles} compilaciones en {compile_duration:.4f}s ({compile_rate:.2f} comp/s)")
        
        # 4b. Engine Execution
        compiled = compiler.compile(dsl_script)
        engine = A2EEngine()
        # Suppress logging to console for benchmark execution speed
        original_log = engine.log
        engine.log = lambda msg: None # Silent
        
        start_time = time.perf_counter()
        num_runs = 500
        for _ in range(num_runs):
            # Reset state for clean execution
            engine.state = {
                "user_profile": {
                    "posts": [
                        {"id": 1, "status": "active"},
                        {"id": 2, "status": "draft"}
                    ]
                }
            }
            engine.run_workflow(compiled)
        exec_duration = time.perf_counter() - start_time
        exec_rate = num_runs / exec_duration
        print(f"  -> Ejecucion del Motor (Secuencial): {num_runs} ejecuciones completas en {exec_duration:.4f}s ({exec_rate:.2f} ejec/s)")
        
        # Restore logger
        engine.log = original_log

        print("\n" + "=" * 80)
        print("                        BENCHMARK SUMMARY RESULTS")
        print("=" * 80)
        print(f"  DocStore Writes  : {write_qps:12.2f} op/s   (Time: {write_duration:.4f}s)")
        print(f"  DocStore Reads   : {read_qps:12.2f} op/s   (Time: {read_duration:.4f}s)")
        print(f"  DocStore Updates : {update_qps:12.2f} op/s   (Time: {update_duration:.4f}s)")
        print(f"  Vector Search    : {search_qps:12.2f} QPS    (Time: {search_duration:.4f}s)")
        print(f"  Embeddings Cache : {cache_hit_qps:12.2f} hit/s  (Time: {cache_hit_duration:.4f}s)")
        print(f"  DSL Compilation  : {compile_rate:12.2f} comp/s (Time: {compile_duration:.4f}s)")
        print(f"  DSL Execution    : {exec_rate:12.2f} run/s  (Time: {exec_duration:.4f}s)")
        print("=" * 80)
        print("   [ÉXITO] Todas las metricas de rendimiento local recopiladas con exito.")
        print("=" * 80)
        
    finally:
        # Cleanup benchmark directory
        if os.path.exists(bench_dir):
            try:
                shutil.rmtree(bench_dir)
            except Exception:
                pass

if __name__ == "__main__":
    run_benchmarks()
