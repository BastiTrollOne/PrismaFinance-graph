# MCP/Dockerfile
FROM python:3.11-slim

WORKDIR /app

# Instalamos el servidor MCP
RUN pip install --no-cache-dir mcp-neo4j-cypher uvicorn

# Creamos el script parcheado
# - Habilitamos CORS (allow_origins=['*']) para que Open WebUI pueda entrar.
# - Usamos la ruta por defecto (/mcp/) ya que no se puede cambiar.
RUN echo "import asyncio" > start.py && \
    echo "import os" >> start.py && \
    echo "from mcp_neo4j_cypher import server" >> start.py && \
    echo "if __name__ == '__main__':" >> start.py && \
    echo "    print('🚀 Iniciando Servidor MCP Neo4j (Con CORS)...')" >> start.py && \
    echo "    db_url = os.getenv('NEO4J_URI', 'bolt://neo4j-db:7687')" >> start.py && \
    echo "    user = os.getenv('NEO4J_USERNAME', 'neo4j')" >> start.py && \
    echo "    password = os.getenv('NEO4J_PASSWORD', 'prismafinance123')" >> start.py && \
    echo "    database = os.getenv('NEO4J_DATABASE', 'neo4j')" >> start.py && \
    echo "    asyncio.run(server.main(" >> start.py && \
    echo "        transport='sse'," >> start.py && \
    echo "        port=8082," >> start.py && \
    echo "        host='0.0.0.0'," >> start.py && \
    echo "        db_url=db_url," >> start.py && \
    echo "        username=user," >> start.py && \
    echo "        password=password," >> start.py && \
    echo "        database=database," >> start.py && \
    echo "        allow_origins=['*']" >> start.py && \
    echo "    ))" >> start.py

# Exponemos el puerto
EXPOSE 8082

# Ejecutamos el script
CMD ["python", "start.py"]