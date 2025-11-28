import os
import requests
import sys
import time
import traceback
import re
import csv
import json
from io import StringIO

# --- IMPORTS ---
from langchain_openai import ChatOpenAI
from langchain_community.graphs import Neo4jGraph
from langchain_community.graphs.graph_document import GraphDocument, Node, Relationship
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser

# --- CONFIGURACIÓN ---
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://neo4j-db:7687")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "prismafinance123")

_BASE_INGEST_URL = os.getenv("INGESTION_URL", "http://ingestion-service:8000")
if _BASE_INGEST_URL.endswith("/upload"):
    INGESTION_URL = _BASE_INGEST_URL
else:
    INGESTION_URL = f"{_BASE_INGEST_URL}/upload"

OPENAI_BASE = os.getenv("OPENAI_API_BASE", "http://192.168.137.1:8001/v1")
OPENAI_KEY = os.getenv("OPENAI_API_KEY", "sk-no-key-needed")
MODEL_NAME = os.getenv("LLM_MODEL_ID", "qwen/qwen3-4b-2507")

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# ==============================================================================
# 1. EXTRACTOR PERSONALIZADO (Corregido: Doble corchete para literales)
# ==============================================================================
# --- HEADER DE EXTRACCIÓN ESTRICTA ---
STRICT_INSTRUCTION_HEADER = """
*** INSTRUCCIONES DEL SISTEMA - LEER ATENTAMENTE ***
ROL: Eres un Ingeniero de Datos riguroso construyendo un Grafo Financiero.
TAREA: Extraer CADA ENTIDAD y RELACIÓN del texto a continuación.
REGLAS CRÍTICAS:
1. PROHIBIDO RESUMIR: Si hay 20 transacciones, debes extraer 20 sets de nodos. No omitas nada.
2. TIPOS DE ENTIDAD: Usa SOLO: 'Organizacion', 'Persona', 'Proyecto', 'Monto', 'Concepto'.
3. RELACIONES PERMITIDAS:
   - Persona -> GESTIONA -> Proyecto
   - Organizacion -> TIENE_COSTO -> Monto
   - Organizacion -> PAGADO_A -> Organizacion (Proveedor)
   - Persona -> PERTENECE_A -> Organizacion
4. MONTOS SON NODOS: Los valores numéricos (ej. '450000') son NODOS de tipo 'Monto', NO propiedades.
5. CONCEPTOS: La descripción del gasto es un nodo 'Concepto'.
*** FIN INSTRUCCIONES - COMIENZAN LOS DATOS ***
"""

def custom_graph_extraction(llm, text_chunk):
    """Extrae nodos/relaciones usando prompting directo."""
    prompt = ChatPromptTemplate.from_messages([
        ("system", EXTRACTION_SYSTEM_PROMPT),
        ("human", "{input}")
    ])
    
    # Cadena: Prompt -> LLM -> Parser JSON
    chain = prompt | llm | JsonOutputParser()
    
    try:
        return chain.invoke({"input": text_chunk})
    except Exception as e:
        print(f"   ⚠️ Error parseando JSON del modelo: {e}")
        return {"nodes": [], "relationships": []}

def convert_json_to_graph_document(source_doc, json_data):
    """Convierte el JSON crudo a objetos GraphDocument de LangChain."""
    nodes = []
    relationships = []
    
    # Procesar nodos
    for n in json_data.get("nodes", []):
        nodes.append(Node(id=n["id"], type=n["type"]))
        
    # Procesar relaciones
    for r in json_data.get("relationships", []):
        # Buscar nodos completos para source y target (necesario para LangChain)
        source_node = next((x for x in nodes if x.id == r["source"]), Node(id=r["source"], type="Unknown"))
        target_node = next((x for x in nodes if x.id == r["target"]), Node(id=r["target"], type="Unknown"))
        
        relationships.append(Relationship(
            source=source_node,
            target=target_node,
            type=r["type"]
        ))
        
    return GraphDocument(nodes=nodes, relationships=relationships, source=source_doc)

# ==============================================================================
# 2. UTILIDADES (Limpieza y Parsing)
# ==============================================================================
def clean_text_content(text):
    if not text: return ""
    return re.sub(r'\s+', ' ', str(text)).strip()

def csv_to_narrative(text_chunk):
    if text_chunk is None: return ""
    lines = text_chunk.strip().split('\n')
    clean_lines = [line for line in lines if not line.startswith("===")]
    cleaned_text = "\n".join(clean_lines)
    if not cleaned_text: return text_chunk
    
    # Detección rápida de CSV
    if "ID" not in cleaned_text and "Monto" not in cleaned_text:
        if "," not in cleaned_text and ";" not in cleaned_text:
            return text_chunk

    try:
        first_line = cleaned_text.strip().split('\n')[0]
        delimiter = ";" if ";" in first_line and "," not in first_line else ","
        f = StringIO(cleaned_text)
        reader = csv.DictReader(f, delimiter=delimiter)
        if not reader.fieldnames: return text_chunk
        headers = [str(h).lower() for h in reader.fieldnames]
        
        if not any(x in str(headers) for x in ['monto', 'valor', 'costo', 'presupuesto']):
            return text_chunk

        narrative = []
        for row in reader:
            # Limpieza básica de filas
            row_norm = {str(k).lower().strip(): str(v).strip() for k, v in row.items() if k and v}
            
            # Extracción simple basada en columnas clave
            persona = row_norm.get('persona') or row_norm.get('responsable') or "Desconocido"
            monto = row_norm.get('monto') or row_norm.get('valor') or row_norm.get('costo')
            concepto = row_norm.get('concepto') or row_norm.get('proyecto') or "General"
            
            if monto:
                oracion = f"Registro: {persona} gestiona un monto de {monto} para {concepto}."
                narrative.append(oracion)
            
        if narrative:
            return "\n".join(narrative)
    except:
        pass # Si falla el CSV, devolvemos el texto original
    return text_chunk

def connect_to_neo4j():
    try:
        graph = Neo4jGraph(url=NEO4J_URI, username=NEO4J_USERNAME, password=NEO4J_PASSWORD)
        graph.refresh_schema()
        return graph
    except Exception as e:
        print(f"   ❌ Error Neo4j: {e}")
        return None

def unify_entities(graph):
    print("🚀 [UNIFICACIÓN] Conectando nodos...")
    try:
        graph.query("MATCH (n) WHERE size(n.id) < 2 DETACH DELETE n")
        # Inferencia simple
        graph.query("MATCH (p:Persona)-[:GESTIONA]->(m:Monto) MERGE (p)-[:TIENE_COSTO]->(m)")
        print("✨ Grafo optimizado.")
    except: pass

# ==============================================================================
# 3. PIPELINE PRINCIPAL
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
            print(f"   ❌ Error API Ingesta: {response.text}")
            return {}
        return response.json()
    except Exception as e:
        print(f"   ❌ Excepción Ingesta: {e}")
        return {}

def run_graph_extraction(file_path):
    raw_data = ingest_document(file_path)
    if not raw_data: return

    # Preparar documentos base
    lc_docs = []
    if "page_content" in raw_data:
        clean = csv_to_narrative(raw_data.get("page_content"))
        lc_docs.append(Document(page_content=str(clean), metadata=raw_data.get("metadata")))
    elif "documents" in raw_data:
        for d in raw_data["documents"]:
            clean = csv_to_narrative(d.get("page_content"))
            lc_docs.append(Document(page_content=str(clean), metadata=d.get("metadata")))

    print(f"📄 Extrayendo grafo de {len(lc_docs)} fragmentos con {MODEL_NAME}...")

    # Conectar a Servicios
    try:
        llm = ChatOpenAI(
            model=MODEL_NAME, 
            openai_api_base=OPENAI_BASE,
            openai_api_key=OPENAI_KEY,
            temperature=0
        )
    except Exception as e:
        print(f"   ❌ Error Config LLM: {e}")
        return

    graph = connect_to_neo4j()
    if not graph: return

    success_count = 0
    for i, doc in enumerate(lc_docs):
        print(f"   🧠 Procesando Chunk {i+1} (Prompting manual)...")
        
        # 1. Extracción (JSON puro)
        extracted_json = custom_graph_extraction(llm, doc.page_content)
        
        # 2. Conversión a GraphDocument
        if extracted_json and (extracted_json.get("nodes") or extracted_json.get("relationships")):
            try:
                graph_doc = convert_json_to_graph_document(doc, extracted_json)
                
                # 3. Guardar en Neo4j
                graph.add_graph_documents([graph_doc], baseEntityLabel=True, include_source=True)
                success_count += 1
                print(f"   ✅ Chunk {i+1}: {len(graph_doc.nodes)} nodos guardados.")
                
            except Exception as e:
                print(f"   ❌ Error guardando Chunk {i+1}: {e}")
        else:
            print(f"   ⚠️ Chunk {i+1}: No se encontraron entidades.")

    if success_count > 0:
        unify_entities(graph)
        print("🎉 ¡Proceso FINALIZADO!")
    else:
        print("💀 Fin: No se guardaron datos.")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        run_graph_extraction(sys.argv[1])