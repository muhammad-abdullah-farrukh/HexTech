"""
Neo4j connection config. Set these via environment variables, or just
edit the defaults below for local dev.

    export NEO4J_URI="bolt://localhost:7687"
    export NEO4J_USER="neo4j"
    export NEO4J_PASSWORD="your-password"
"""
import os

NEO4J_URI      = os.environ.get("NEO4J_URI", "neo4j://127.0.0.1:7687")
NEO4J_USER     = os.environ.get("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.environ.get("NEO4J_PASSWORD", "5053811238")
NEO4J_DATABASE = os.environ.get("NEO4J_DATABASE", "neo4j")