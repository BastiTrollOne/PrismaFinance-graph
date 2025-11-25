import os
import requests
import sys
import time
import traceback
import re
import csv
from io import StringIO

# --- IMPORTS ---
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_community.chat_models import ChatOllama
from langchain_core.documents import Document
from langchain_community.graphs import Neo4jGraph

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

# Chunking optimizado
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# ==============================================================================
# 1. ETL CON LIMPIEZA DE "BASURA" DE INGESTA
# ==============================================================================
def clean_text_content(text):
    if not text: return ""
    return re.sub(r'\s+', ' ', str(text)).strip()

def csv_to_narrative(text_chunk):
    """Transforma CSV a narrativa ignorando líneas decorativas."""
    if text_chunk is None: return ""
    
    # 1. LIMPIEZA PREVIA (CRUCIAL): Quitar líneas "=== Hoja..."
    lines = text_chunk.strip().split('\n')
    clean_lines = [line for line in lines if not line.startswith("===")]
    cleaned_text = "\n".join(clean_lines)

    # Si no quedó nada, devolvemos vacío
    if not cleaned_text: return text_chunk

    # 2. Detección rápida
    if "ID" not in cleaned_text and "Monto" not in cleaned_text:
        # Intento de última oportunidad: buscar comas
        if "," not in cleaned_text and ";" not in cleaned_text:
            return text_chunk

    try:
        # 3. Separador
        first_line = cleaned_text.strip().split('\n')[0]
        delimiter = ";" if ";" in first_line and "," not in first_line else ","
        print(f"   🔎 DEBUG: Separador: '{delimiter}' | Primera línea real: {first_line[:50]}...")

        f = StringIO(cleaned_text)
        reader = csv.DictReader(f, delimiter=delimiter)
        
        # 4. Validación Cabeceras
        if not reader.fieldnames: 
            print("   ⚠️ DEBUG: Sin cabeceras válidas.")
            return text_chunk
            
        headers = [str(h).lower() for h in reader.fieldnames]
        print(f"   🔎 DEBUG: Cabeceras detectadas: {headers}")
        
        # Filtro permisivo
        if not any(x in str(headers) for x in ['monto', 'valor', 'costo', 'presupuesto', 'usd', 'precio', 'id']):
            print("   ⚠️ DEBUG: No parece financiero.")
            return text_chunk

        print(f"   📊 [ETL] Generando narrativa...")
        narrative = []
        
        for row in reader:
            row_norm = {str(k).lower().strip(): str(v).strip() for k, v in row.items() if k and v}
            
            def get_val(patterns):
                for p in patterns:
                    for k, v in row_norm.items():
                        if p in k: return v
                return "Desconocido"

            concepto = get_val(['concepto', 'glosa', 'proyecto', 'nombre', 'item', 'ítem'])
            categoria = get_val(['categor', 'tipo'])
            org = get_val(['organi', 'fuente', 'proveedor', 'empresa', 'responsable'])
            persona = get_val(['persona', 'sponsor', 'responsable', 'encargado'])
            monto_raw = get_val(['monto', 'valor', 'costo', 'usd'])
            
            monto = re.sub(r'[^\d]', '', monto_raw)
            
            if monto:
                # Frase optimizada para el modelo
                oracion = f"Registro: La entidad '{org}' (Representante: {persona}) tiene una relación '{categoria}' con el proyecto '{concepto}' por un monto de {monto}."
                narrative.append(oracion)
            
        if narrative:
            print(f"   ✅ [ETL] {len(narrative)} historias creadas.")
            return "\n".join(narrative)
            
    except Exception as e:
        print(f"   ❌ DEBUG Parsing: {e}")
    
    return text_chunk

def connect_to_neo4j():
    try:
        graph = Neo4jGraph(url=NEO4J_URI, username=NEO4J_USERNAME, password=NEO4J_PASSWORD)
        graph.refresh_schema()
        return graph
    except Exception as e:
        print(f"   ❌ Error Neo4j: {e}")
        return None

# ==============================================================================
# 2. UNIFICACIÓN
# ==============================================================================
def unify_entities(graph):
    print("🚀 [UNIFICACIÓN] Conectando nodos...")
    try:
        graph.query("MATCH (n) WHERE size(n.id) < 2 OR n.id CONTAINS 'copyright' DETACH DELETE n")
        
        graph.query("""
        MATCH (n) WITH toLower(n.id) as name, collect(n) as nodes 
        WHERE size(nodes) > 1 
        CALL apoc.refactor.mergeNodes(nodes, {properties:'combine', mergeRels:true}) 
        YIELD node RETURN count(node)
        """)
        
        # Inferencia Contextual
        graph.query("MATCH (p:Persona)<-[:MENTIONS]-(d)-[:MENTIONS]->(o:Organizacion) MERGE (p)-[:PERTENECE_A]->(o)")
        graph.query("MATCH (o:Organizacion)<-[:MENTIONS]-(d)-[:MENTIONS]->(m) WHERE labels(m) IN [['Monto'],['Costo'],['Presupuesto']] MERGE (o)-[:TIENE_PRESUPUESTO]->(m)")
        
        print("✨ Grafo optimizado.")
    except Exception as e:
        print(f"   ⚠️ Aviso unificación: {e}")

# ==============================================================================
# 3. INGESTA BLINDADA
# ==============================================================================
def ingest_document(file_path):
    print(f"📤 Procesando '{file_path}'...")
    try:
        with open(file_path, 'rb') as f:
            response = requests.post(
                INGESTION_URL, files={'file': f}, 
                params={"chunk_size": CHUNK_SIZE, "chunk_overlap": CHUNK_OVERLAP}
            )
        if response.status_code != 200:
            print(f"   ❌ Error API: {response.text}")
            return {}
        return response.json()
    except Exception as e:
        print(f"   ❌ Excepción Ingesta: {e}")
        return {}

def run_graph_extraction(file_path):
    raw_data = ingest_document(file_path)
    if not raw_data: return

    def safe_doc(c, m): return Document(page_content=str(c) if c else "", metadata=m or {})
    
    lc_docs = []
    if "page_content" in raw_data:
        clean = csv_to_narrative(raw_data.get("page_content"))
        lc_docs.append(safe_doc(clean, raw_data.get("metadata")))
    elif "documents" in raw_data:
        for d in raw_data["documents"]:
            if isinstance(d, dict):
                clean = csv_to_narrative(d.get("page_content"))
                lc_docs.append(safe_doc(clean, d.get("metadata")))

    print(f"📄 Extrayendo grafo de {len(lc_docs)} fragmentos...")

    try:
        llm = ChatOllama(model=MODEL_NAME, base_url=OLLAMA_BASE_URL, temperature=0)
        transformer = LLMGraphTransformer(
            llm=llm,
            allowed_nodes=["Organizacion", "Persona", "Proyecto", "Monto", "Concepto", "Costo", "Presupuesto"],
            allowed_relationships=["FINANCIA", "DIRIGE", "TIENE_COSTO", "PERTENECE_A", "MENTIONS"],
            node_properties=False
        )
    except Exception as e:
        print(f"   ❌ Error LLM: {e}")
        return

    graph = connect_to_neo4j()
    if not graph: return

    success_count = 0
    for i, doc in enumerate(lc_docs):
        try:
            res = transformer.convert_to_graph_documents([doc])
            if res:
                graph.add_graph_documents(res, baseEntityLabel=True, include_source=True)
                success_count += 1
                print(f"   ✅ Chunk {i+1} OK.")
            else:
                print(f"   ⚠️ Chunk {i+1} sin entidades.")
                
        except Exception as e:
            print(f"   ❌ Error en Chunk {i+1}: {e}")

    if success_count > 0:
        unify_entities(graph)
        print("🎉 ¡Proceso FINALIZADO!")
    else:
        print("💀 Error: No se guardaron datos.")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        run_graph_extraction(sys.argv[1])