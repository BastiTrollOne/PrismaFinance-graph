import os
import requests
import sys
import time
import traceback
import re

# --- IMPORTS ---
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_community.chat_models import ChatOllama
from langchain_core.documents import Document
from langchain_community.graphs import Neo4jGraph

# ==============================================================================
# 1. CONFIGURACIÓN DEL ENTORNO
# ==============================================================================
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

# Configuración de Chunking (Optimizada para densidad)
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# ==============================================================================
# 2. ETL AVANZADO: DE DATOS A NARRATIVA
# ==============================================================================
def clean_text_content(text):
    """Limpia caracteres extraños que confunden al LLM."""
    if not text: return ""
    # Eliminar múltiples espacios y saltos de línea raros
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def csv_to_narrative(text_chunk):
    """
    Transforma filas de CSV financiero en narrativa rica semánticamente.
    Diseñado para maximizar la detección de relaciones 'FINANCIA' y 'TIENE_COSTO'.
    """
    # Heurística: Si no parece CSV financiero, devolver tal cual
    if "ID Transacción" not in text_chunk and "Monto" not in text_chunk and "," not in text_chunk:
        return text_chunk
    
    narrative = []
    lines = text_chunk.split('\n')
    is_csv_context = False

    print("📊 [ETL] Detectado patrón CSV. Iniciando conversión narrativa profunda...")

    for line in lines:
        # Detección de cabeceras
        if "ID Transacción" in line or "Monto (Valor)" in line:
            is_csv_context = True
            continue
        # Ignorar metadatos de paginación
        if not line.strip() or "=== Hoja" in line:
            continue
            
        if is_csv_context:
            try:
                # Intento robusto de parsing
                parts = line.split(',')
                if len(parts) >= 5:
                    # Mapeo flexible de columnas (ajusta índices si tu Excel cambia)
                    # Estructura esperada: ID, Fecha, Concepto, Categoria, Org, Monto...
                    concepto = clean_text_content(parts[2])
                    categoria = clean_text_content(parts[3])
                    org = clean_text_content(parts[4])
                    monto_raw = clean_text_content(parts[5])
                    
                    # Normalización de montos para el grafo
                    monto = monto_raw.replace('.', '').replace('$', '').strip()

                    # Generación de narrativa según tipo de gasto (Business Logic)
                    if "Costo" in categoria:
                        oracion = f"La organización '{org}' es el proveedor asociado al Costo de {monto} por el concepto de '{concepto}'."
                    elif "Presupuesto" in categoria:
                        oracion = f"La organización '{org}' ha aprobado un Presupuesto oficial de {monto} destinado a '{concepto}'."
                    elif "Financia" in categoria or "Financiamiento" in categoria:
                        oracion = f"La entidad financiera '{org}' FINANCIA directamente el proyecto '{concepto}' con un aporte de {monto}."
                    else:
                        oracion = f"Existe una relación comercial entre '{org}' y el proyecto '{concepto}' valorizada en {monto}."
                    
                    narrative.append(oracion)
                else:
                    # Si la línea está rota, la pasamos limpia
                    narrative.append(clean_text_content(line))
            except Exception:
                continue # Saltamos líneas corruptas sin romper el flujo
        else:
            narrative.append(clean_text_content(line))
            
    return "\n".join(narrative)

def connect_to_neo4j():
    """Conexión resiliente con reintentos."""
    max_retries = 3
    for attempt in range(max_retries):
        try:
            graph = Neo4jGraph(url=NEO4J_URI, username=NEO4J_USERNAME, password=NEO4J_PASSWORD)
            # Crear índices para acelerar la fusión
            graph.query("CREATE INDEX IF NOT EXISTS FOR (n:Persona) ON (n.id)")
            graph.query("CREATE INDEX IF NOT EXISTS FOR (n:Organizacion) ON (n.id)")
            graph.query("CREATE INDEX IF NOT EXISTS FOR (n:Proyecto) ON (n.id)")
            graph.refresh_schema()
            return graph
        except Exception as e:
            print(f"⚠️ Fallo conexión Neo4j (Intento {attempt+1}/{max_retries}): {e}")
            time.sleep(2)
    raise Exception("❌ No se pudo conectar a Neo4j tras varios intentos.")

# ==============================================================================
# 3. PROTOCOLO DE UNIFICACIÓN "AGUJERO NEGRO" (THE ISLAND KILLER)
# ==============================================================================
def unify_entities(graph):
    """
    Ejecuta una cascada de algoritmos para fusionar nodos duplicados y conectar islas.
    Usa APOC masivamente.
    """
    print("🚀 [UNIFICACIÓN] Iniciando Protocolo de Alta Densidad (7 Etapas)...")
    
    # Helper para ejecutar queries silenciosamente
    def run_query(name, cypher):
        try:
            graph.query(cypher)
            print(f"   ✅ Etapa {name} completada.")
        except Exception as e:
            # Ignoramos errores de "Node not found" que ocurren cuando se fusiona recursivamente
            if "NotFoundException" not in str(e):
                print(f"   ⚠️ Etapa {name} con aviso: {e}")

    # 1. LIMPIEZA PREVIA
    run_query("1-Limpieza", """
        MATCH (n) 
        WHERE size(n.id) < 2 
           OR n.id CONTAINS 'copyright' 
           OR n.id CONTAINS '......'
        DETACH DELETE n
    """)

    # 2. FUSIÓN EXACTA (Case Insensitive)
    # Une "Banco Regional" con "BANCO REGIONAL"
    run_query("2-FusiónExacta", """
        MATCH (n)
        WITH toLower(n.id) as name, labels(n) as lbls, collect(n) as nodes
        WHERE size(nodes) > 1
        CALL apoc.refactor.mergeNodes(nodes, {properties: 'combine', mergeRels: true})
        YIELD node RETURN count(node)
    """)

    # 3. MAGNETO DE DOMINIO (Fusión por Palabras Clave)
    # Esta es la clave para unir islas. Si ambos tienen "Candelaria" y son Organizaciones, ¡PÉGALOS!
    domain_keywords = ["Candelaria", "Banco", "Minera", "Constructora", "Proyecto", "Regional", "Andino", "Lundin", "Komatsu"]
    print(f"   🧲 Aplicando Magneto de Dominio para: {domain_keywords}")
    
    for kw in domain_keywords:
        query_magnet = f"""
        MATCH (n1:Organizacion), (n2:Organizacion)
        WHERE elementId(n1) < elementId(n2)
          AND toLower(n1.id) CONTAINS toLower('{kw}')
          AND toLower(n2.id) CONTAINS toLower('{kw}')
        WITH n1, n2 LIMIT 50
        CALL apoc.refactor.mergeNodes([n1, n2], {{properties: 'combine', mergeRels: true}})
        YIELD node RETURN count(node)
        """
        run_query(f"3-Magneto-{kw}", query_magnet)

    # 4. FUSIÓN POR CONTENCIÓN (Jerarquía de Nombres)
    # Une "Ingeniero Pedro Maza" con "Pedro Maza"
    for label in ["Persona", "Organizacion", "Proyecto"]:
        query_containment = f"""
        MATCH (n1:{label}), (n2:{label})
        WHERE elementId(n1) <> elementId(n2)
          AND size(n1.id) > 3 AND size(n2.id) > 3
          AND (
            toLower(n1.id) CONTAINS toLower(n2.id) OR 
            toLower(n2.id) CONTAINS toLower(n1.id)
          )
        WITH n1, n2 LIMIT 100
        CALL apoc.refactor.mergeNodes([n1, n2], {{properties: 'combine', mergeRels: true}})
        YIELD node RETURN count(node)
        """
        run_query(f"4-Contención-{label}", query_containment)

    # 5. FUSIÓN DIFUSA (Fuzzy Matching)
    # Red de seguridad para errores tipográficos (Similitud > 0.85)
    run_query("5-FuzzyMatching", """
        MATCH (n1), (n2)
        WHERE elementId(n1) < elementId(n2) 
          AND labels(n1) = labels(n2)
          AND n1.id <> n2.id
          AND apoc.text.sorensenDiceSimilarity(toLower(n1.id), toLower(n2.id)) > 0.85
        WITH n1, n2 LIMIT 100
        CALL apoc.refactor.mergeNodes([n1, n2], {properties: 'combine', mergeRels: true})
        YIELD node RETURN count(node)
    """)

    # 6. PEGAMENTO CONTEXTUAL (Relaciones Automáticas)
    # Si quedaron nodos sueltos pero vinieron del mismo documento, conéctalos.
    print("   🔨 Construyendo puentes contextuales...")
    
    # Persona <-> Organización (en el mismo doc)
    run_query("6a-PuentePersOrg", """
        MATCH (p:Persona)<-[:MENTIONS]-(d:Document)-[:MENTIONS]->(o:Organizacion)
        MERGE (p)-[r:PERTENECE_A]->(o) 
        ON CREATE SET r.source = 'Auto-Inferencia'
    """)
    
    # Organización <-> Proyecto (en el mismo doc)
    run_query("6b-PuenteOrgProj", """
        MATCH (o:Organizacion)<-[:MENTIONS]-(d:Document)-[:MENTIONS]->(proj:Proyecto)
        MERGE (o)-[r:FINANCIA]->(proj)
        ON CREATE SET r.source = 'Auto-Inferencia'
    """)

    # Dinero <-> Organización (en el mismo doc)
    run_query("6c-PuenteOrgMoney", """
        MATCH (m)<-[:MENTIONS]-(d:Document)-[:MENTIONS]->(o:Organizacion)
        WHERE labels(m) IN [['Monto'], ['Costo'], ['Presupuesto']]
        MERGE (o)-[r:TIENE_PRESUPUESTO]->(m)
        ON CREATE SET r.source = 'Auto-Inferencia'
    """)

    print("✨ [UNIFICACIÓN] Proceso finalizado. El grafo debería ser denso.")

# ==============================================================================
# 4. INGESTA Y EJECUCIÓN ATÓMICA (RESILIENTE)
# ==============================================================================
def ingest_document(file_path):
    print(f"📤 Ingestando '{file_path}'...")
    try:
        with open(file_path, 'rb') as f:
            response = requests.post(
                INGESTION_URL, 
                files={'file': f}, 
                params={"chunk_size": CHUNK_SIZE, "chunk_overlap": CHUNK_OVERLAP}
            )
        if response.status_code != 200:
            raise Exception(f"Error HTTP {response.status_code}: {response.text}")
        return response.json()
    except Exception as e:
        raise Exception(f"Fallo crítico en servicio de ingesta: {e}")

def run_graph_extraction(file_path):
    # 1. Obtener texto crudo
    try:
        raw_data = ingest_document(file_path)
    except Exception as e:
        print(f"❌ Error fatal en ingesta: {e}")
        return

    # 2. Preparar chunks
    lc_docs = []
    # Manejo de formato single vs multi
    if "page_content" in raw_data:
        clean = csv_to_narrative(raw_data["page_content"])
        lc_docs.append(Document(page_content=clean, metadata=raw_data.get("metadata", {})))
    elif "documents" in raw_data:
        for d in raw_data["documents"]:
            clean = csv_to_narrative(d["page_content"])
            lc_docs.append(Document(page_content=clean, metadata=d.get("metadata", {})))
            
    print(f"📄 Preparado para procesar {len(lc_docs)} fragmentos de forma atómica.")

    # 3. Inicializar LLM
    try:
        llm = ChatOllama(model=MODEL_NAME, base_url=OLLAMA_BASE_URL, temperature=0)
        
        # Definición de esquema exhaustiva
        llm_transformer = LLMGraphTransformer(
            llm=llm,
            allowed_nodes=[
                "Organizacion", "Persona", "Proyecto", 
                "Monto", "Concepto", "Costo", "Presupuesto", 
                "Fecha", "Lugar", "Contrato"
            ],
            allowed_relationships=[
                "FINANCIA", "DIRIGE", "TIENE_COSTO", 
                "TIENE_PRESUPUESTO", "PERTENECE_A", "MENTIONS", 
                "TIENE_MONTO", "UBICADO_EN", "FIRMA", "REALIZO"
            ],
            node_properties=False # Clave para estabilidad con modelos pequeños
        )
    except Exception as e:
        print(f"❌ Error config LLM: {e}")
        return

    # 4. BUCLE DE EXTRACCIÓN ATÓMICA (Fail-Safe)
    success_count = 0
    graph = connect_to_neo4j()
    
    print("⛏️  Iniciando minería de grafo (Chunk por Chunk)...")
    
    for i, doc in enumerate(lc_docs):
        try:
            # Procesamos UN solo chunk a la vez
            # Si este falla, el `except` lo atrapa y seguimos con el siguiente
            chunks_result = llm_transformer.convert_to_graph_documents([doc])
            
            if chunks_result:
                graph.add_graph_documents(chunks_result, baseEntityLabel=True, include_source=True)
                success_count += 1
                print(f"   ✅ Chunk {i+1}/{len(lc_docs)} procesado correctamente.")
            else:
                print(f"   ⚠️ Chunk {i+1}/{len(lc_docs)} vacío (sin entidades).")
                
        except Exception as e:
            # Análisis de error para feedback
            error_msg = str(e)
            if "tail_type" in error_msg or "string indices" in error_msg:
                print(f"   ⚠️ Salto Chunk {i+1}: Error de formato JSON del LLM (común en tablas complejas). Omitiendo fragmento.")
            else:
                print(f"   ⚠️ Error desconocido en Chunk {i+1}: {e}")
            continue

    # 5. FINALIZACIÓN
    if success_count > 0:
        print(f"💾 Persistencia completada ({success_count}/{len(lc_docs)} chunks exitosos).")
        # Ejecutar unificación SOLO si hubo datos nuevos
        unify_entities(graph)
        print("🎉 ¡Proceso finalizado con éxito!")
    else:
        print("❌ No se pudo extraer información válida de ningún fragmento. Revisa el formato del archivo.")

if __name__ == "__main__":
    # Modo CLI para pruebas manuales
    if len(sys.argv) > 1:
        run_graph_extraction(sys.argv[1])