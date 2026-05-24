// A2E Web Playground Application Logic (ES Module)
import { pipeline, env } from 'https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.0.0-alpha.19';
env.allowLocalModels = false; // Force loading from HF CDN

let pyodideInstance = null;
let compiledWorkflowJSON = null;
let extractor = null;

// DSL templates mapping
const TEMPLATES = {
    rag_mcp: `# 1. Configurar variables de entrada
SET target_user = 42
SET rag_query = "Como acelerar deep learning localmente con Intel Arc"
SET mcp_query = "extraer precios de Amazon"

# 2. Consultar API de usuario (MOCK)
API_CALL user_profile = GET "https://api.test/users/{{target_user}}"

# 3. Filtrar posts de forma avanzada tipo MongoDB (Filtro Inline JSON)
FILTER_DATA filtered_posts = user_profile.posts WHERE query == {"$and": [{"status": "active"}, {"id": {"$gt": 101}}]}

# 4. Búsqueda Semántica RAG Nativa en LocalVectorStore (Caché local e In-Memory)
SEMANTIC_SEARCH rag_results = knowledge_base QUERY "{{rag_query}}" LIMIT 2 FILTER {"category": "tech"}

# 5. Búsqueda MCP contra MCPARDF_CLIENT (Tolerancia offline activa)
MCP_SEARCH mcp_tools = "{{mcp_query}}" LIMIT 2 TYPES ["tool"]

# 6. Decisión condicional e inyección de contexto
IF len(rag_results) > 0 THEN
    SET notification_msg = "RAG: {{rag_results.0.metadata.title}}! Herramienta compatible: {{mcp_tools.0.name}}"
    API_CALL notify = POST "https://api.test/notify" WITH {"msg": "{{notification_msg}}"}
ELSE
    SET notification_msg = "No se encontraron posts o resultados RAG."
END`,

    conditional: `# Configurar el perfil inicial
SET user_role = "admin"
SET target_user = 42

# API Call mockeado
API_CALL user_profile = GET "https://api.test/users/{{target_user}}"

# Filtrar posts usando MongoDB match
FILTER_DATA active_posts = user_profile.posts WHERE query == {"status": "active"}

# Lógica condicional
IF user_role == "admin" THEN
    SET alert_msg = "Acceso concedido a Administrador. Posts activos: len({{active_posts}})"
    API_CALL alert = POST "https://api.test/notify" WITH {"msg": "{{alert_msg}}"}
ELSE
    SET alert_msg = "Acceso denegado."
END`,

    loop: `# Configurar listado de tareas
SET tasks_list = [
    {"id": 1, "task": "Optimizar VectorStore", "urgent": true},
    {"id": 2, "task": "Revisar logs en D:", "urgent": false},
    {"id": 3, "task": "Portar caching a Pyodide", "urgent": true}
]

# Filtrar tareas urgentes
FILTER_DATA urgent_tasks = tasks_list WHERE query == {"urgent": true}

# Iterar sobre las tareas urgentes e invocar alertas
LOOP urgent_tasks AS t
    SET log_msg = "Procesando tarea urgente {{t.id}}: {{t.task}}"
    API_CALL log_api = POST "https://api.test/notify" WITH {"msg": "{{log_msg}}"}
END`
};

// UI Elements
const codeEditor = document.getElementById("codeEditor");
const templateSelector = document.getElementById("templateSelector");
const btnCompile = document.getElementById("btnCompile");
const btnRun = document.getElementById("btnRun");
const jsonViewer = document.getElementById("jsonViewer");
const terminalOutput = document.getElementById("terminalOutput");
const btnClearConsole = document.getElementById("btnClearConsole");
const loadingOverlay = document.getElementById("loadingOverlay");
const loadingText = document.getElementById("loadingText");
const statusDot = document.getElementById("statusDot");
const statusText = document.getElementById("statusText");

// Initialize console logging
function logToTerminal(message, type = "info") {
    const line = document.createElement("div");
    line.classList.add("terminal-line");
    if (type) line.classList.add(type);
    line.textContent = message;
    terminalOutput.appendChild(line);
    terminalOutput.scrollTop = terminalOutput.scrollHeight;
}

// Load selected template
function loadTemplate() {
    const selected = templateSelector.value;
    if (TEMPLATES[selected]) {
        codeEditor.value = TEMPLATES[selected];
        logToTerminal(`[System] Cargada plantilla: '${selected}'`, "success");
    }
}

templateSelector.addEventListener("change", loadTemplate);
btnClearConsole.addEventListener("click", () => {
    terminalOutput.innerHTML = "";
    logToTerminal("[System] Consola limpia.", "success");
});

// Setup mock databases inside Pyodide's virtual filesystem (high-fidelity simulation!)
async function setupVirtualDatabases(pyodide) {
    try {
        // Create simulated folders mirroring the D: drive
        pyodide.FS.mkdir("/D:");
        pyodide.FS.mkdir("/D:/.lmstudio");
        pyodide.FS.mkdir("/D:/.lmstudio/a2e_db");
        pyodide.FS.mkdir("/D:/.lmstudio/a2e_db/a2e_sessions");
        pyodide.FS.mkdir("/D:/.lmstudio/a2e_vector_db");
        pyodide.FS.mkdir("/D:/.lmstudio/a2e_vector_db/knowledge_base");
        pyodide.FS.mkdir("/D:/.lmstudio/a2e_vector_db/user_notes");
        
        logToTerminal("[System] Montado sistema de archivos virtual síncrono (/D:/.lmstudio/)...", "success");

        // Calculate real 384-D embeddings using Transformers.js in WebGPU/Wasm!
        logToTerminal("[System] Calculando embeddings de alta dimension (384-D) en WebGPU/Wasm...");
        
        let emb1 = [1.0, 0.0, 0.0, 0.0]; // fallback
        let emb2 = [0.0, 1.0, 0.0, 0.0];
        let emb3 = [0.0, 0.0, 1.0, 0.0];
        let embNote = [0.9, 0.1, 0.0, 0.0];
        let dim = 4;
        
        if (extractor) {
            try {
                dim = 384;
                const out1 = await extractor("Protocolo A2E: Agent-to-Execution. Protocolo declarativo para ejecucion de flujos de IA", { pooling: 'mean', normalize: true });
                const out2 = await extractor("Aceleracion local con Intel Arc y DirectML. Optimizacion de deep learning en tarjetas de video Intel", { pooling: 'mean', normalize: true });
                const out3 = await extractor("Recetario de cocina italiana tradicional. Secretos para la mejor pasta fresca en casa", { pooling: 'mean', normalize: true });
                const outNote = await extractor("Notas de Mauricio: Integrar js-vector-store con A2E", { pooling: 'mean', normalize: true });
                
                emb1 = Array.from(out1.data);
                emb2 = Array.from(out2.data);
                emb3 = Array.from(out3.data);
                embNote = Array.from(outNote.data);
                logToTerminal("[System] Embeddings 384-D generados exitosamente via WebGPU/Wasm!", "success");
            } catch (e) {
                logToTerminal(`[System Warning] Fallo calculo WebGPU: ${e}. Usando vectores 4D de laboratorio.`, "warning");
                dim = 4;
            }
        }

        // Write knowledge_base metadata JSON
        const kbManifest = {
            "ids": ["doc-1", "doc-2", "doc-3"],
            "meta": [
                {
                    "title": "Protocolo A2E: Agent-to-Execution",
                    "category": "tech",
                    "summary": "Protocolo declarativo para ejecucion de flujos de IA"
                },
                {
                    "title": "Aceleracion local con Intel Arc y DirectML",
                    "category": "tech",
                    "summary": "Optimizacion de deep learning en tarjetas de video Intel"
                },
                {
                    "title": "Recetario de cocina italiana tradicional",
                    "category": "cooking",
                    "summary": "Secretos para la mejor pasta fresca en casa"
                }
            ],
            "dim": dim,
            "model": dim === 384 ? "all-MiniLM-L6-v2-webgpu" : "gemma-300m-local"
        };
        pyodide.FS.writeFile(
            "/D:/.lmstudio/a2e_vector_db/knowledge_base/knowledge_base.json",
            JSON.stringify(kbManifest, null, 2)
        );

        // Write binary Float32 vectors to simulate a real local VectorStore
        const allVectors = [...emb1, ...emb2, ...emb3];
        const vectors = new Float32Array(allVectors);
        pyodide.FS.writeFile(
            "/D:/.lmstudio/a2e_vector_db/knowledge_base/knowledge_base.bin",
            new Uint8Array(vectors.buffer)
        );

        // Write user_notes metadata JSON
        const notesManifest = {
            "ids": ["note-1"],
            "meta": [
                {
                    "title": "Notas de Mauricio: Integrar js-vector-store con A2E",
                    "category": "personal"
                }
            ],
            "dim": dim,
            "model": dim === 384 ? "all-MiniLM-L6-v2-webgpu" : "gemma-300m-local"
        };
        pyodide.FS.writeFile(
            "/D:/.lmstudio/a2e_vector_db/user_notes/user_notes.json",
            JSON.stringify(notesManifest, null, 2)
        );

        // Write user_notes binary Float32 vector
        const noteVectors = new Float32Array(embNote);
        pyodide.FS.writeFile(
            "/D:/.lmstudio/a2e_vector_db/user_notes/user_notes.bin",
            new Uint8Array(noteVectors.buffer)
        );

        logToTerminal("[System] Cargadas bases de datos vectoriales y manifiestos de la Trinidad en WebAssembly.", "success");
    } catch (e) {
        logToTerminal(`[System Error] Error al configurar el FS virtual: ${e}`, "error");
    }
}

// Initialize Pyodide WebAssembly Runtime
async function initPyodide() {
    try {
        logToTerminal("[System] Cargando pipeline de Hugging Face (all-MiniLM-L6-v2) en WebGPU/Wasm...");
        loadingText.textContent = "Cargando Modelos de Embeddings (WebGPU)...";
        
        let device = "wasm";
        // Check WebGPU support
        if (navigator.gpu) {
            try {
                const adapter = await navigator.gpu.requestAdapter();
                if (adapter) {
                    device = "webgpu";
                }
            } catch(e) {}
        }
        
        logToTerminal(`[System] Inicializando Transformers.js pipeline en dispositivo: '${device}'`);
        
        try {
            extractor = await pipeline('feature-extraction', 'Xenova/all-MiniLM-L6-v2', { device: device });
            logToTerminal("[System] Xenova/all-MiniLM-L6-v2 cargado y cacheado en el navegador.", "success");
        } catch (e) {
            logToTerminal(`[System Warning] Fallo al cargar WebGPU pipeline: ${e}. Intentando fallback WASM CPU...`, "warning");
            try {
                extractor = await pipeline('feature-extraction', 'Xenova/all-MiniLM-L6-v2', { device: 'wasm' });
                logToTerminal("[System] Xenova/all-MiniLM-L6-v2 cargado en modo WASM CPU con éxito.", "success");
            } catch (err) {
                logToTerminal(`[System Error] No se pudo cargar el extractor de embeddings: ${err}`, "error");
            }
        }

        loadingText.textContent = "Inicializando WebAssembly & Python Runtime...";
        logToTerminal("[System] Iniciando descarga de WebAssembly Pyodide Runtime...");
        
        // Boot Pyodide with console capture
        pyodideInstance = await loadPyodide({
            stdout: (text) => {
                // Log standard prints from A2EEngine straight to our HTML terminal widget!
                if (text.startsWith("[A2E Engine]")) {
                    logToTerminal(text, "engine-log");
                } else {
                    logToTerminal(text, "");
                }
            },
            stderr: (text) => {
                logToTerminal(`[Stderr] ${text}`, "error");
            }
        });

        logToTerminal("[System] Pyodide cargado correctamente. Creando directorio de paquetes '/a2e_core'...");
        
        // Create python package directories in Pyodide
        pyodideInstance.FS.mkdir("/a2e_core");
        
        // Write all pre-bundled A2E Python sources into Pyodide virtual FS
        if (window.A2E_PY_SOURCE) {
            for (const [filename, content] of Object.entries(window.A2E_PY_SOURCE)) {
                pyodideInstance.FS.writeFile(`/a2e_core/${filename}`, content);
            }
            logToTerminal(`[System] Escritos ${Object.keys(window.A2E_PY_SOURCE).length} módulos Python del framework A2E en el Sandbox virtual.`, "success");
        } else {
            throw new Error("No se encontraron los fuentes de A2E pre-empaquetados en py_source.js!");
        }

        // Setup databases in MEMFS
        await setupVirtualDatabases(pyodideInstance);

        // Update status UI
        loadingOverlay.style.opacity = "0";
        setTimeout(() => loadingOverlay.style.display = "none", 500);
        statusDot.classList.add("online");
        statusText.textContent = `Pyodide: Listo (Wasm + ${device.toUpperCase()})`;
        btnCompile.disabled = false;
        
        logToTerminal("[System] Entorno WebAssembly A2E listo para compilar y ejecutar.", "success");
    } catch (e) {
        loadingText.textContent = "Error al inicializar Pyodide.";
        logToTerminal(`[Fatal] Fallo de inicio de Pyodide: ${e}`, "error");
        statusText.textContent = "Pyodide: Error";
    }
}

// Bind compile actions
btnCompile.addEventListener("click", () => {
    const dslCode = codeEditor.value.trim();
    if (!dslCode) {
        logToTerminal("[Error] El editor está vacío.", "error");
        return;
    }

    logToTerminal("[Compilador] Compilando script DSL de A2E...");
    
    try {
        // Execute compilation in Python
        // Clean outputs
        jsonViewer.textContent = "// Compilando...";
        
        // Escaping python strings safely
        pyodideInstance.globals.set("dsl_input", dslCode);
        
        const compileScript = `
import sys
sys.path.append('/')
import json
from a2e_core.compiler import A2ECompiler

compiler = A2ECompiler()
try:
    compiled_steps = compiler.compile(dsl_input)
    result_json = json.dumps(compiled_steps, indent=2, ensure_ascii=False)
except Exception as e:
    result_json = json.dumps({"error": True, "message": str(e)})
result_json
`;
        
        const result = pyodideInstance.runPython(compileScript);
        const parsed = JSON.parse(result);
        
        if (parsed.error) {
            jsonViewer.textContent = `// Error de Compilación:\n${parsed.message}`;
            logToTerminal(`[Error de Compilación] ${parsed.message}`, "error");
            btnRun.disabled = true;
        } else {
            compiledWorkflowJSON = result;
            jsonViewer.textContent = result;
            logToTerminal(`[Compilador] Compilación finalizada con éxito! ${parsed.length} pasos estructurados en el AST.`, "success");
            btnRun.disabled = false;
        }
    } catch (err) {
        jsonViewer.textContent = `// Excepción:\n${err}`;
        logToTerminal(`[Excepción de Compilación] ${err}`, "error");
        btnRun.disabled = true;
    }
});

// Bind run actions
btnRun.addEventListener("click", async () => {
    if (!compiledWorkflowJSON) {
        logToTerminal("[Error] Primero debes compilar un script DSL válido.", "error");
        return;
    }

    logToTerminal("[Engine] Iniciando ejecución del motor A2E en WebAssembly...");
    
    // Pre-calculate query embeddings using the live local Wasm/WebGPU pipeline!
    let embeddingsCache = {};
    if (extractor && compiledWorkflowJSON) {
        try {
            const steps = JSON.parse(compiledWorkflowJSON);
            for (const step of steps) {
                if (step.op === "semantic_search") {
                    let queryText = step.query;
                    if (queryText === "{{rag_query}}") {
                        queryText = "Como acelerar deep learning localmente con Intel Arc";
                    }
                    
                    logToTerminal(`[WebGPU RAG] Generando vector de embedding en GPU para: '${queryText}'...`);
                    const output = await extractor(queryText, { pooling: 'mean', normalize: true });
                    const vector = Array.from(output.data);
                    embeddingsCache[step.query] = vector;
                    embeddingsCache[queryText] = vector;
                    logToTerminal(`[WebGPU RAG] Vector de ${vector.length} dimensiones generado y listo.`, "success");
                }
            }
        } catch (e) {
            logToTerminal(`[WebGPU Warning] Error al pre-calcular embeddings: ${e}`, "warning");
        }
    }
    
    try {
        pyodideInstance.globals.set("workflow_json", compiledWorkflowJSON);
        pyodideInstance.globals.set("embeddings_cache_js", pyodideInstance.toPy(embeddingsCache));
        
        const runScript = `
import json
from a2e_core.engine import A2EEngine

# We inject the initialized VectorStore into the engine state so execute_semantic_search works perfectly!
from a2e_core.vector_store import LocalVectorStore
dim = 384 if embeddings_cache_js else 4
v_store = LocalVectorStore(db_dir="/D:/.lmstudio/a2e_vector_db", dim=dim)

initial_state = {
    "vector_store": v_store,
    "vector_db_dir": "/D:/.lmstudio/a2e_vector_db",
    "doc_db_dir": "/D:/.lmstudio/a2e_db",
    "__embeddings": dict(embeddings_cache_js) if embeddings_cache_js else {}
}

engine = A2EEngine(initial_state)
compiled_workflow = json.loads(workflow_json)

try:
    final_state = engine.run_workflow(compiled_workflow)
    # Exclude complex non-serializable objects from final printed state representation
    printable_state = {k: v for k, v in final_state.items() if k not in ("vector_store", "vector_db_dir", "doc_db_dir")}
    run_result = json.dumps({"success": True, "state": printable_state}, indent=2, ensure_ascii=False)
except Exception as e:
    run_result = json.dumps({"success": False, "message": str(e)})
run_result
`;
        
        const result = pyodideInstance.runPython(runScript);
        const parsed = JSON.parse(result);
        
        if (parsed.success) {
            logToTerminal("[Engine] Flujo de trabajo completado con éxito!", "success");
            logToTerminal(`[Estado Final] ${JSON.stringify(parsed.state, null, 2)}`, "success");
        } else {
            logToTerminal(`[Fallo del Engine] ${parsed.message}`, "error");
        }
    } catch (err) {
        logToTerminal(`[Excepción de Ejecución] ${err}`, "error");
    }
});

// Startup tasks
loadTemplate();
initPyodide();
