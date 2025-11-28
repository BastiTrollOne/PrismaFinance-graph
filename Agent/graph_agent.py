import os
import requests
import sys
import re
import traceback
from langchain_openai import ChatOpenAI
from langchain_community.graphs import Neo4jGraph
from langchain_community.graphs.graph_document import GraphDocument, Node, Relationship
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# ==============================================================================
# 1. CONFIGURACIÓN DE ALTO RENDIMIENTO
# ==============================================================================
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://neo4j-db:7687")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "prismafinance123")

_BASE_INGEST_URL = os.getenv("INGESTION_URL", "http://ingestion-service:8000")
INGESTION_URL = _BASE_INGEST_URL if _BASE_INGEST_URL.endswith("/upload") else f"{_BASE_INGEST_URL}/upload"

# CONEXIÓN REMOTA
OPENAI_BASE = os.getenv("OPENAI_API_BASE", "http://192.168.50.1:8900/v1")
OPENAI_KEY = os.getenv("OPENAI_API_KEY", "sk-no-key-needed")

# ¡OJO! Pon aquí el nombre EXACTO de tu modelo Qwen
MODEL_NAME = "qwen/qwen3-4b-2507"

# CHUNKING OPTIMIZADO PARA DENSIDAD
# 1500 tokens es el punto dulce: suficiente contexto para entender, 
# pero suficientemente corto para que Qwen no se "canse" y extraiga todo.
CHUNK_SIZE = 1500  
CHUNK_OVERLAP = 150

# ==============================================================================
# 2. PROMPT "PIPE" AGRESIVO (Velocidad + Exhaustividad)
# ==============================================================================
STRICT_INSTRUCTION_HEADER = """
*** SYSTEM INSTRUCTIONS ***
ROLE: High-Performance Knowledge Graph Engine.
TASK: Extract ALL relationships from the text into a pipe-delimited format.
OUTPUT MODE: RAW DATA ONLY. No thinking, no markdown, no json.

Taxonomy (Types):
- Persona (Humans, Roles)
- Organizacion (Companies, Banks, Suppliers)
- Monto (Specific money values)
- Concepto (Docs, Categories, Items, Status)
- Proyecto (Mines, Sites, Specific Projects)
- Fecha (Dates, Periods)

FORMAT (Strictly 5 columns separated by '|'):
Entity1|Type1|RELATION_VERB|Entity2|Type2

EXTRACTION RULES:
1. EXHAUSTIVE: If a sentence implies a connection, extract it. 
2. ATOMIC: Break complex sentences into multiple lines.
3. RELATIONS: Use UPPERCASE verbs (e.g., PAGO, GESTIONA, PERTENECE_A, ES_PARA).
4. DATES: Connect events to dates (e.g., Factura -> TIENE_FECHA -> 2024-01).
5. MONEY: Connect Amounts to Concepts (e.g., 500 USD -> ES_VALOR_DE -> Factura).

EXAMPLE INPUT:
"Juan Perez (Gerente) de Candelaria aprobó 500 USD para el Proyecto Norte en Enero."

EXAMPLE OUTPUT:
Juan Perez|Persona|TIENE_ROL|Gerente|Concepto
Juan Perez|Persona|PERTENECE_A|Candelaria|Organizacion
Juan Perez|Persona|APROBO|500 USD|Monto
500 USD|Monto|ASIGNADO_A|Proyecto Norte|Proyecto
Proyecto Norte|Proyecto|TIENE_FECHA|Enero|Fecha
"""

# ==============================================================================
# 3. PARSER ULTRARRÁPIDO (Python Nativo)
# ==============================================================================
def parse_pipe_output(text):
    """Procesa la respuesta cruda y extrae las líneas válidas."""
    results = []
    lines = text.strip().split('\n')
    
    for line in lines:
        line = line.strip()
        # Filtrado rápido: debe tener 4 tuberías para tener 5 columnas
        if line.count('|') != 4:
            continue
            
        parts = [p.strip() for p in line.split('|')]
        
        # Validación extra de longitud para evitar errores
        if len(parts) == 5 and all(parts):
            results.append({
                "entity1": parts[0],
                "type1": parts[1],
                "rel": parts[2],
                "entity2": parts[3],
                "type2": parts[4]
            })
    return results

def custom_graph_extraction(llm, text_chunk):
    try:
        prompt = ChatPromptTemplate.from_messages([
            ("system", STRICT_INSTRUCTION_HEADER),
            ("human", "{input}")
        ])
        
        # StrOutputParser es lo más ligero que existe
        chain = prompt | llm | StrOutputParser()
        
        # Invocación
        response_text = chain.invoke({"input": text_chunk})
        
        # DEBUG: Ver la primera línea para confirmar formato
        debug_snippet = response_text.strip().split('\n')[0][:80]
        print(f"   ⚡ Qwen Output: {debug_snippet}...")
        
        return parse_pipe_output(response_text)

    except Exception as e:
        print(f"   ⚠️ Error extracción: {e}")
        return []

def convert_to_graph_document(source_doc, items_list):
    nodes_dict = {}
    relationships = []
    
    for item in items_list:
        # Limpieza y normalización
        e1 = item["entity1"].strip()
        t1 = item["type1"].strip()
        rel = item["rel"].upper().replace(" ", "_").strip()
        e2 = item["entity2"].strip()
        t2 = item["type2"].strip()
        
        # Gestión de Nodos (Evitar duplicados)
        if e1 not in nodes_dict: nodes_dict[e1] = Node(id=e1, type=t1)
        if e2 not in nodes_dict: nodes_dict[e2] = Node(id=e2, type=t2)
            
        relationships.append(Relationship(
            source=nodes_dict[e1],
            target=nodes_dict[e2],
            type=rel
        ))
    
    return GraphDocument(
        nodes=list(nodes_dict.values()), 
        relationships=relationships, 
        source=source_doc
    )

# ==============================================================================
# 4. ORQUESTADOR
# ==============================================================================
def connect_to_neo4j():
    try:
        return Neo4jGraph(url=NEO4J_URI, username=NEO4J_USERNAME, password=NEO4J_PASSWORD)
    except Exception as e:
        print(f"   ❌ Error Neo4j: {e}")
        return None

def ingest_document(file_path):
    print(f"📤 Procesando '{file_path}'...")
    try:
        with open(file_path, 'rb') as f:
            res = requests.post(INGESTION_URL, files={'file': f}, 
                                params={"chunk_size": CHUNK_SIZE, "chunk_overlap": CHUNK_OVERLAP})
        if res.status_code != 200: return {}
        return res.json()
    except Exception: return {}

def run_graph_extraction(file_path):
    # 1. Ingesta (OCR/Parsing)
    raw_data = ingest_document(file_path)
    if not raw_data: return

    # 2. Preparar Documentos LangChain
    lc_docs = []
    source_docs = raw_data.get("documents", [])
    if "page_content" in raw_data: source_docs = [raw_data]
    
    for d in source_docs:
        # Truco: Añadimos metadata del nombre de archivo al contenido para contexto
        content = f"Fuente: {d.get('metadata', {}).get('filename', 'Doc')}\n{d.get('page_content', '')}"
        lc_docs.append(Document(page_content=content, metadata=d.get("metadata", {})))

    print(f"📄 Extrayendo grafo de {len(lc_docs)} fragmentos con {MODEL_NAME}...")
    
    # 3. Configuración LLM (Optimizada para Qwen Instruct)
    try:
        llm = ChatOpenAI(
            model=MODEL_NAME, 
            openai_api_base=OPENAI_BASE, 
            openai_api_key=OPENAI_KEY, 
            temperature=0.0,       # Cero creatividad, pura extracción
            max_tokens=4096,       # Permitir respuestas largas
            request_timeout=300    # 5 min timeout (seguridad)
        )
    except Exception as e:
        print(f"   ❌ Error Init LLM: {e}")
        return

    graph = connect_to_neo4j()
    if not graph: return

    # 4. Loop de Extracción
    success_count = 0
    total_nodes = 0
    
    for i, doc in enumerate(lc_docs):
        print(f"   🧠 Procesando Chunk {i+1}/{len(lc_docs)}...")
        
        items = custom_graph_extraction(llm, doc.page_content)
        
        if items:
            try:
                graph_doc = convert_to_graph_document(doc, items)
                if graph_doc.nodes:
                    graph.add_graph_documents([graph_doc], baseEntityLabel=True, include_source=True)
                    count = len(graph_doc.nodes)
                    success_count += 1
                    total_nodes += count
                    print(f"   ✅ Chunk {i+1}: {count} nodos y {len(graph_doc.relationships)} relaciones guardadas.")
                else:
                    print(f"   ⚠️ Chunk {i+1}: Datos extraídos pero sin nodos válidos.")
            except Exception as e:
                print(f"   ❌ Error guardando Chunk {i+1}: {e}")
                traceback.print_exc()
        else:
            print(f"   ⚠️ Chunk {i+1}: Sin datos válidos.")

    if success_count > 0:
        print(f"🎉 ¡Proceso FINALIZADO! Total Nodos: {total_nodes}")
    else:
        print("💀 Fin: No se guardaron datos.")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        run_graph_extraction(sys.argv[1])