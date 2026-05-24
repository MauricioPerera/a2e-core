import urllib.request
import json
import time

def test_qwen_compilation():
    print("=" * 80)
    print("PROBANDO MODELO COMPACTO: QWEN 2.5 0.5B EN EL FRAMEWORK A2E")
    print("=" * 80)
    
    # 1. Definir la instrucción en lenguaje natural del usuario
    user_prompt = "Busca en la base de tecnologia sobre DirectML para el usuario 42 y si encuentra algo notifica con la primera herramienta compatible de la categoria de tracking"
    print(f"Instrucción del usuario:\n  '{user_prompt}'\n")

    # 2. Diseñar el System Prompt optimizado para modelos de 0.5B (Few-Shot Prompting estricto)
    system_prompt = (
        "Eres un compilador estricto que traduce instrucciones en lenguaje natural a código DSL de A2E.\n"
        "Responde ÚNICAMENTE con el código DSL limpio. No agregues saludos, explicaciones, markdown ni comentarios.\n\n"
        "Comandos soportados:\n"
        "- SET variable = valor\n"
        "- API_CALL id = GET|POST url [WITH body]\n"
        "- FILTER_DATA id = source WHERE query == json_filter\n"
        "- SEMANTIC_SEARCH id = collection QUERY \"query\" LIMIT N FILTER json_filter\n"
        "- MCP_SEARCH id = \"query\" LIMIT N TYPES json_array\n"
        "- IF condicion THEN\n"
        "- ELSE\n"
        "- END\n\n"
        "Ejemplo 1:\n"
        "Usuario: \"Busca en la base de tecnologia sobre DirectML para el usuario 10\"\n"
        "Asistente:\n"
        "SET target_user = 10\n"
        "SEMANTIC_SEARCH rag_results = knowledge_base QUERY \"DirectML\" LIMIT 2 FILTER {\"category\": \"tech\"}\n"
        "IF len(rag_results) > 0 THEN\n"
        "  API_CALL notify = POST \"https://api.test/notify\" WITH {\"msg\": \"Encontrado\"}\n"
        "END"
    )

    payload = {
        "model": "ollama-imports/qwen2.5/qwen2.5-0.5b.gguf",
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        "temperature": 0.1,
        "max_tokens": 300
    }

    url = "http://127.0.0.1:1234/v1/chat/completions"
    headers = {"Content-Type": "application/json"}
    
    print("Enviando petición a Qwen 2.5 0.5B local en puerto 1234...")
    start_time = time.perf_counter()
    try:
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(url, data=data, headers=headers, method="POST")
        with urllib.request.urlopen(req, timeout=90) as resp:
            duration = time.perf_counter() - start_time
            res_body = json.loads(resp.read().decode("utf-8"))
            dsl_output = res_body["choices"][0]["message"]["content"].strip()
            
            print(f"Petición completada con éxito en {duration:.4f} segundos!\n")
            print("=" * 80)
            print("                    CÓDIGO DSL GENERADO POR QWEN 0.5B")
            print("=" * 80)
            print(dsl_output)
            print("=" * 80)
            
            # 3. Intentar compilar el script generado por Qwen usando el compilador real de A2E
            print("\nIntentando compilar el código generado usando A2ECompiler...")
            from a2e_core.compiler import A2ECompiler
            compiler = A2ECompiler()
            compiled = compiler.compile(dsl_output)
            print("   [ÉXITO] ¡El script generado por Qwen 0.5B compiló correctamente a JSON IR!")
            print(json.dumps(compiled, indent=2, ensure_ascii=False))
            
    except Exception as e:
        print(f"\n[ERROR] Error al conectar o procesar con Qwen 2.5 0.5B: {e}")
        print("Asegúrate de que LM Studio está encendido en el puerto 1234 y que el modelo está cargado.")

if __name__ == "__main__":
    test_qwen_compilation()
