import json
from .operations import (
    execute_set,
    execute_api_call,
    execute_filter,
    execute_conditional,
    execute_loop,
    execute_semantic_search,
    execute_mcp_search
)

class A2EEngine:
    def __init__(self, initial_state=None):
        self.state = initial_state if initial_state is not None else {}
        self.logs = []
        self.op_registry = {
            "set": execute_set,
            "api_call": execute_api_call,
            "filter": execute_filter,
            "conditional": execute_conditional,
            "loop": execute_loop,
            "semantic_search": execute_semantic_search,
            "mcp_search": execute_mcp_search
        }

    def log(self, message):
        timestamp = self.logs.append(message)
        print(f"[A2E Engine] {message}")

    def register_op(self, op_name, handler_func):
        """
        Extensibility feature: developers can register custom operations
        (e.g., specific wp-a2e tools like create_post, update_menu).
        """
        self.op_registry[op_name] = handler_func
        self.log(f"Operacion personalizada registrada: '{op_name}'")

    def execute_step(self, step):
        if not isinstance(step, dict):
            self.log("[ERROR] El paso no es un diccionario valido.")
            return False
            
        op = step.get("op")
        if not op:
            self.log("[ERROR] El paso no define la propiedad de operacion 'op'.")
            return False
            
        handler = self.op_registry.get(op)
        if not handler:
            self.log(f"[ERROR] Operacion no soportada: '{op}'")
            return False
            
        try:
            result = handler(step, self.state, self)
            if result:
                self.log(result)
            return True
        except Exception as e:
            self.log(f"[EXCEPCION] Fallo al ejecutar '{op}': {e}")
            return False

    def run_workflow(self, workflow):
        """
        Runs an A2E workflow represented as a list of step dictionaries (representing JSONL lines).
        """
        self.log("=" * 60)
        self.log("INICIANDO EJECUCION DE FLUJO DE TRABAJO A2E")
        self.log("=" * 60)
        
        if isinstance(workflow, str):
            # Attempt to parse as JSON or JSONL
            try:
                # Try single JSON array
                workflow = json.loads(workflow)
            except json.JSONDecodeError:
                # Try JSONL format
                steps = []
                for line in workflow.strip().split("\n"):
                    if line.strip():
                        steps.append(json.loads(line))
                workflow = steps
                
        if not isinstance(workflow, list):
            self.log("[ERROR] Formato de flujo invalido. Debe ser una lista de pasos.")
            return self.state
            
        for idx, step in enumerate(workflow, 1):
            self.log(f"--- Paso {idx}/{len(workflow)} ---")
            success = self.execute_step(step)
            if not success:
                self.log(f"[PARADA] Flujo interrumpido en el paso {idx} debido a un error.")
                break
                
        self.log("=" * 60)
        self.log("EJECUCION FINALIZADA")
        self.log("=" * 60)
        return self.state
