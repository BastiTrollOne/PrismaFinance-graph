import os
import requests
import sys

# Librerías
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_community.llms import Ollama
from langchain_core.documents import Document
from langchain_neo4j import Neo4jGraph

# --- CONFIGURACIÓN ---
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://neo4j-db:7687")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "prismafinance123")

_BASE_INGEST_URL = os.getenv("INGESTION_URL", "http://ingestion-service:8000")
if _BASE_INGEST_URL.endswith("/upload"):
    INGESTION_URL = _BASE_INGEST_URL
else:
    INGESTION_URL = f"{_BASE_INGEST_URL}/upload"

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://ollama-service:11434")
MODEL_NAME = "qwen2.5:3b" 

def connect_to_neo4j():
    try:
        graph = Neo4jGraph(url=NEO4J_URI, username=NEO4J_USERNAME, password=NEO4J_PASSWORD)
        graph.refresh_schema()
        print("✅ Conexión a Neo4j exitosa.")
        return graph
    except Exception as e:
        raise Exception(f"Error conectando a Neo4j: {e}")

def unify_entities(graph):
    """
    PROTOCOLO DE UNIFICACIÓN 'ESTEROIDES' (Versión Final):
    1. Limpieza de ruido.
    2. Normalización segura de nombres (con APOC merge).
    3. Inferencia de relaciones implícitas (Triangulación por Documento).
    4. Conexión por Palabras Clave (Bridge).
    5. Fusión final por similitud de texto.
    """
    print("🧹 ACTIVANDO PROTOCOLO DE LIMPIEZA Y FUSIÓN AVANZADA...")
    
    try:
        # --- FASE 1: RECOLECCIÓN DE BASURA ---
        # Borra nodos que son errores de OCR (muy cortos o terminan en puntos)
        query_trash = """
        MATCH (n) 
        WHERE size(n.id) < 3 OR n.id ENDS WITH '...' OR n.id CONTAINS 'copyright'
        DETACH DELETE n
        """
        graph.query(query_trash)

        # --- FASE 2: NORMALIZACIÓN SEGURA (ANTI-COLISIÓN) ---
        # Quita prefijos. Si el nombre limpio ya existe, FUSIONA los nodos.
        print("   - Normalizando nombres (Sr./Ing./Dr.)...")
        prefixes = ["Sr.", "Sra.", "Srta.", "Ing.", "Dr.", "Lic.", "Mag.", "Mr.", "Ms."]
        
        for prefix in prefixes:
            # Esta query usa APOC para mezclar nodos si encuentra duplicados al limpiar
            query_smart_rename = f"""
            MATCH (n)
            WHERE n.id STARTS WITH '{prefix}'
            WITH n, trim(substring(n.id, size('{prefix}'))) as new_name
            
            // Buscar si ya existe el 'gemelo' limpio
            OPTIONAL MATCH (target) WHERE target.id = new_name
            
            // CASO A: Existe el gemelo -> Fusionar 'n' dentro de 'target'
            FOREACH (_ IN CASE WHEN target IS NOT NULL THEN [1] ELSE [] END |
                CALL apoc.refactor.mergeNodes([target, n], {{properties: 'discard', mergeRels: true}}) YIELD node 
                RETURN count(*)
            )
            
            // CASO B: No existe -> Simplemente renombrar 'n'
            FOREACH (_ IN CASE WHEN target IS NULL THEN [1] ELSE [] END |
                SET n.id = new_name
            )
            """
            try:
                graph.query(query_smart_rename)
            except Exception as e_apoc:
                # Si falla APOC (raro), hacemos fallback a renombrado simple
                # print(f"     (Nota: Fallback simple para {prefix})") 
                pass

        # --- FASE 3: INFERENCIA LÓGICA (TRIANGULACIÓN) ---
        # Si Persona y Proyecto/Org están en el mismo Doc, conéctalos directamente
        print("   - Infiriendo relaciones directas...")
        
        queries_inference = [
            # Persona -> Organización
            """
            MATCH (p:Persona)<-[:MENTIONS]-(d:Document)-[:MENTIONS]->(o:Organizacion)
            MERGE (p)-[:PERTENECE_A {source: 'Inferencia'}]->(o)
            """,
            # Persona -> Proyecto
            """
            MATCH (p:Persona)<-[:MENTIONS]-(d:Document)-[:MENTIONS]->(proj:Proyecto)
            MERGE (p)-[:TRABAJA_EN {source: 'Inferencia'}]->(proj)
            """
        ]
        for q in queries_inference:
            graph.query(q)

        # --- FASE 4: PEGAMENTO DE PALABRAS CLAVE (BRIDGE) ---
        # Une islas disconexas si comparten palabras clave críticas del negocio
        keywords = ["Cobre", "Titán", "Expansión", "Modernización", "Vibra", "Finanzas"]
        for kw in keywords:
            query_glue = f"""
            MATCH (n1), (n2)
            WHERE elementId(n1) <> elementId(n2)
              AND toLower(n1.id) CONTAINS toLower('{kw}')
              AND toLower(n2.id) CONTAINS toLower('{kw}')
              AND NOT (n1)-[:RELACIONADO_CON]->(n2)
              AND labels(n1) = labels(n2) // Solo unir si son del mismo tipo
            MERGE (n1)-[:ES_LO_MISMO_QUE {{motivo: 'Keyword {kw}'}}]->(n2)
            """
            graph.query(query_glue)

        # --- FASE 5: FUSIÓN FINAL POR SIMILITUD ---
        # La red de seguridad para "Juan Perez" vs "Juan Pérez"
        print("   - Fusionando por similitud de texto...")
        query_merge_sim = """
        MATCH (n1), (n2)
        WHERE elementId(n1) < elementId(n2)
          AND size(n1.id) > 4 AND size(n2.id) > 4
          AND (toLower(n1.id) CONTAINS toLower(n2.id) OR toLower(n2.id) CONTAINS toLower(n1.id))
        MERGE (n1)-[:ES_LO_MISMO_QUE]->(n2)
        """
        graph.query(query_merge_sim)

        print("✨ GRAFO OPTIMIZADO, LIMPIO Y CONECTADO.")
        
    except Exception as e:
        print(f"⚠️ Alerta en proceso de limpieza: {e}")

def ingest_document(file_path):
    print(f"📤 Enviando '{file_path}' a ingesta...")
    try:
        with open(file_path, 'rb') as f:
            response = requests.post(
                INGESTION_URL, 
                files={'file': f}, 
                params={"chunk_size": 2000, "chunk_overlap": 200} # Chunk más grande para mejor contexto
            )
        if response.status_code != 200:
            raise Exception(f"Error Ingesta: {response.text}")
        return response.json()
    except Exception as e:
        raise Exception(f"Fallo en request: {e}")

def run_graph_extraction(file_path):
    # 1. Ingesta
    try:
        raw_data = ingest_document(file_path)
    except Exception as e:
        print(f"❌ Error crítico en ingesta: {e}")
        return

    # Convertir a LangChain
    lc_docs = []
    if "page_content" in raw_data:
        lc_docs.append(Document(page_content=raw_data["page_content"], metadata=raw_data.get("metadata", {})))
    elif "documents" in raw_data:
        for d in raw_data["documents"]:
            lc_docs.append(Document(page_content=d["page_content"], metadata=d.get("metadata", {})))
            
    print(f"📄 Procesando {len(lc_docs)} fragmentos.")

    # 2. Configurar LLM
    print(f"🧠 Inicializando {MODEL_NAME}...")
    try:
        llm = Ollama(model=MODEL_NAME, base_url=OLLAMA_BASE_URL, temperature=0)
        
        # Prompt en Español para la extracción
        llm_transformer = LLMGraphTransformer(
            llm=llm,
            allowed_nodes=["Organizacion", "Persona", "Proyecto", "Monto", "Fecha", "Concepto"],
            allowed_relationships=["FINANCIA", "DIRIGE", "TIENE_COSTO", "TIENE_PRESUPUESTO", "PERTENECE_A"],
        )
    except Exception as e:
        print(f"❌ Error LLM: {e}")
        return

    # 3. Extraer
    print("⛏️  Extrayendo grafo...")
    try:
        graph_documents = llm_transformer.convert_to_graph_documents(lc_docs)
    except Exception as e:
        print(f"❌ Falló extracción: {e}")
        return

    # 4. Guardar y Unificar
    if graph_documents:
        try:
            print("💾 Guardando en Neo4j...")
            graph = connect_to_neo4j()
            graph.add_graph_documents(graph_documents, baseEntityLabel=True, include_source=True)
            
            # AQUÍ OCURRE LA MAGIA AUTOMÁTICA
            unify_entities(graph) 
            
            print("🎉 ¡Conocimiento guardado y conectado!")
        except Exception as e:
             print(f"❌ Error guardando: {e}")
    else:
        print("⚠️ No se encontró información relevante.")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        run_graph_extraction(sys.argv[1])