from google.adk.tools import FunctionTool
import json
from typing import Dict, List, Any, Optional, Union
import os
from pathlib import Path
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from neo4j import GraphDatabase
from langchain_community.graphs import Neo4jGraph
import datetime

# --- Global Settings ---
# Use a local embedding model to avoid dependency on OpenAI API keys
Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")

# --- Path Setup ---
# Get the directory of the current script to robustly locate the data file.
SCRIPT_DIR = Path(__file__).resolve().parent
# The data file is located in the 'ml/data' directory.
DATA_FILE = SCRIPT_DIR.parent / "data" / "ng_elections_narratives_101.json"
print(f"[DataEngTools] Resolved DATA_FILE path: {DATA_FILE}")

# --- Knowledge Base Setup ---
knowledge_base_index = None # Initialize as None, will be loaded on demand

def get_knowledge_base_index(narratives_file: Path = DATA_FILE):
    global knowledge_base_index
    if knowledge_base_index is not None:
        return knowledge_base_index

    if not narratives_file.exists():
        print(f"[KnowledgeBase] Narratives file not found at {narratives_file}. Index will be empty.")
        return None

    print(f"[KnowledgeBase] Loading data from {narratives_file}...")
    # Use SimpleDirectoryReader to load the specific file.
    reader = SimpleDirectoryReader(
        input_files=[narratives_file]
    )
    documents = reader.load_data()
    
    if not documents:
        print("[KnowledgeBase] No documents were loaded. Index will be empty.")
        return None
        
    print("[KnowledgeBase] Creating vector store index...")
    index = VectorStoreIndex.from_documents(documents)
    print("[KnowledgeBase] Index created successfully.")
    knowledge_base_index = index
    return knowledge_base_index

# --- GraphDB Setup ---
graph_driver = None # Initialize as None, will be loaded on demand
graph = None # Initialize as None, will be loaded on demand

def get_graph_db_driver():
    global graph_driver
    if graph_driver is not None:
        return graph_driver

    uri = os.environ.get("NEO4J_URI")
    user = os.environ.get("NEO4J_USER")
    password = os.environ.get("NEO4J_PASSWORD")
    if not all([uri, user, password]):
        print("[GraphDB] Neo4j credentials not found in environment variables.")
        return None
    try:
        driver = GraphDatabase.driver(uri, auth=(user, password))
        driver.verify_connectivity()
        graph_driver = driver
        print("[GraphDB] Neo4j driver initialized successfully.")
        return graph_driver
    except Exception as e:
        print(f"[GraphDB] Failed to initialize Neo4j driver: {e}")
        return None

def populate_graph_db(driver: GraphDatabase.driver, narratives_file: Path = DATA_FILE):
    global graph
    if graph is not None:
        return

    if not driver:
        print("[GraphDB] Driver not initialized. Skipping population.")
        return
    if not narratives_file.exists():
        print(f"[GraphDB] Narratives file not found at {narratives_file}. Skipping population.")
        return

    print(f"[GraphDB] Populating database from {narratives_file}...")
    with open(narratives_file, 'r') as f:
        data = json.load(f)

    with driver.session() as session:
        for narrative in data.get("african_election_narratives", []):
            session.execute_write(_create_narrative_graph, narrative)
    print("[GraphDB] Database population complete.")
    
    # If you want to use the LangChain wrapper elsewhere
    graph = Neo4jGraph(
        url=os.environ.get("NEO4J_URI"),
        username=os.environ.get("NEO4J_USER"),
        password=os.environ.get("NEO4J_PASSWORD")
    )
    print("[GraphDB] LangChain Neo4jGraph initialized.")

def _create_narrative_graph(tx, narrative):
    """
    A transaction function to create narrative nodes and relationships.
    """
    narrative_name = narrative.get("meta_narrative")
    purpose = narrative.get("purpose_effect")

    # Create Narrative Node
    tx.run("MERGE (n:Narrative {name: $name, description: $desc})", name=narrative_name, desc=narrative.get("description"))

    # Create Purpose Node and connect to Narrative
    if purpose:
        tx.run("MERGE (p:Purpose {name: $name})", name=purpose)
        tx.run("""
            MATCH (n:Narrative {name: $narrative_name})
            MATCH (p:Purpose {name: $purpose_name})
            MERGE (n)-[:HAS_PURPOSE]->(p)
        """, narrative_name=narrative_name, purpose_name=purpose)

    # Create Tactic Nodes and connect to Narrative
    for tactic in narrative.get("tactics_used", []):
        tx.run("MERGE (t:Tactic {name: $name})", name=tactic)
        tx.run("""
            MATCH (n:Narrative {name: $narrative_name})
            MATCH (t:Tactic {name: $tactic_name})
            MERGE (n)-[:USES_TACTIC]->(t)
        """, narrative_name=narrative_name, tactic_name=tactic)

    # Create Location Nodes and connect to Narrative
    for location in narrative.get("common_locations", []):
        tx.run("MERGE (l:Location {name: $name})", name=location)
        tx.run("""
            MATCH (n:Narrative {name: $narrative_name})
            MATCH (l:Location {name: $location_name})
            MERGE (n)-[:COMMON_IN]->(l)
        """, narrative_name=narrative_name, location_name=location)

# --- Social Media Collection ---
def social_media_collector(platform: str, query: str, count: int = 10) -> Dict[str, Any]:
    """
    Fetches data from social media platforms like Twitter/X and Facebook.
    
    TODO: Implement real social media API integrations:
    - Twitter API v2 for tweet collection
    - Facebook Graph API for post collection
    - Instagram Basic Display API
    - TikTok Research API
    - Reddit API
    """
    print(f"[DataEngTool] Collecting {count} items from {platform} with query: '{query}'")
    
    # TODO: Implement real API integrations
    # Each platform requires:
    # 1. API key management and authentication
    # 2. Rate limiting and error handling
    # 3. Data normalization across platforms
    # 4. Content filtering and compliance

    return {
        "status": "error",
        "message": f"Real {platform} API integration not yet implemented",
        "platform": platform,
        "query": query,
        "requested_count": count
    }

# --- NLP Pipeline ---
def run_nlp_pipeline(text: str, language: str = "en") -> Dict[str, Any]:
    """
    Prepares and analyzes text for downstream tasks.
    
    TODO: Implement comprehensive NLP pipeline with:
    - Advanced language detection
    - Multi-language tokenization
    - Named entity recognition
    - Sentiment analysis
    - Topic modeling
    """
    print(f"[DataEngTool] Running NLP pipeline on text (lang: {language}): '{text[:50]}...'")
    
    # TODO: Implement real NLP pipeline
    # This should include:
    # 1. Language detection using fasttext or similar
    # 2. Tokenization with spaCy or NLTK
    # 3. Text cleaning and normalization
    # 4. Feature extraction for downstream tasks

    return {
        "status": "error",
        "message": "Real NLP pipeline not yet implemented",
        "original_text": text,
        "specified_language": language
    }

# --- Database Update ---
def database_updater(collection_name: str, data: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Stores collected data and analysis results in MongoDB through CloudDatabaseService.
    """
    if not isinstance(data, list):
        return {"status": "error", "message": "Data must be a list of records."}
    
    print(f"[DataEngTool] Updating collection '{collection_name}' with {len(data)} records.")
    
    try:
        # Import and use CloudDatabaseService for real database operations
        from services.cloud_database_service import CloudDatabaseService
        db_service = CloudDatabaseService()
        
        # Map collection names to appropriate methods
        if collection_name.lower() in ['actors', 'actor']:
            results = []
            for record in data:
                try:
                    uuid = db_service.create_actor(record)
                    results.append({"uuid": uuid, "status": "success"})
                except Exception as e:
                    results.append({"status": "error", "error": str(e), "record": record})
                    
        elif collection_name.lower() in ['narratives', 'narrative']:
            results = []
            for record in data:
                try:
                    uuid = db_service.create_narrative(record)
                    results.append({"uuid": uuid, "status": "success"})
                except Exception as e:
                    results.append({"status": "error", "error": str(e), "record": record})
                    
        elif collection_name.lower() in ['lexicons', 'lexicon', 'terms']:
            results = []
            for record in data:
                try:
                    uuid = db_service.create_lexicon_term(record)
                    results.append({"uuid": uuid, "status": "success"})
                except Exception as e:
                    results.append({"status": "error", "error": str(e), "record": record})
                    
        elif collection_name.lower() in ['content', 'content_items']:
            results = []
            for record in data:
                try:
                    uuid = db_service.create_content_item(record)
                    results.append({"uuid": uuid, "status": "success"})
                except Exception as e:
                    results.append({"status": "error", "error": str(e), "record": record})
                    
        else:
            # For other collections, use direct MongoDB insertion
            collection = db_service.mongo_db[collection_name]
            
            # Add timestamps to records
            for record in data:
                record['created_at'] = datetime.datetime.now(datetime.timezone.utc)
                record['updated_at'] = datetime.datetime.now(datetime.timezone.utc)
            
            result = collection.insert_many(data)
            results = [{"inserted_id": str(id), "status": "success"} for id in result.inserted_ids]

        success_count = sum(1 for r in results if r.get("status") == "success")
        error_count = len(results) - success_count
        
        return {
            "status": "success",
            "collection_name": collection_name,
            "total_records": len(data),
            "successful_inserts": success_count,
            "failed_inserts": error_count,
            "results": results
        }
        
    except ImportError:
        print("[DataEngTool] CloudDatabaseService not available")
        return {
            "status": "error",
            "message": "CloudDatabaseService not available. Please check MongoDB connection.",
            "collection_name": collection_name,
            "record_count": len(data)
        }
    except Exception as e:
        print(f"[DataEngTool] Database error: {e}")
        return {
            "status": "error",
            "message": f"Database error: {str(e)}",
            "collection_name": collection_name,
            "record_count": len(data)
        }

# --- Graph Database Management ---
def manage_graph_db(action: str, node_type: Optional[str] = None, edge_type: Optional[str] = None, properties: Optional[Dict[str, Any]] = None, source_node_id: Optional[str] = None, target_node_id: Optional[str] = None) -> Dict[str, Any]:
    """
    Handles integration and updates for a relationship mapping database (e.g., Neo4j).
    Supported actions: 'add_node', 'add_edge', 'update_node', 'query'.
    For 'query', the 'properties' field should contain a 'cypher' key with the query.
    """
    if not graph_driver:
        return {"status": "error", "message": "GraphDB driver not initialized."}

    print(f"[GraphDB] Performing action '{action}'")
    with graph_driver.session() as session:
        if action == "add_node":
            if not node_type or not properties or not properties.get("name"):
                return {"status": "error", "message": "For 'add_node', 'node_type' and 'properties' (with 'name') are required."}
            
            prop_str = ", ".join([f"n.{key} = ${key}" for key in properties.keys()])
            query = f"MERGE (n:{node_type} {{name: $name}}) SET {prop_str}"
            session.run(query, **properties)
            return {"status": "success", "action": action, "details": f"Node '{properties.get('name')}' of type '{node_type}' added/updated."}

        elif action == "add_edge":
            if not source_node_id or not target_node_id or not edge_type:
                return {"status": "error", "message": "For 'add_edge', 'source_node_id', 'target_node_id', and 'edge_type' are required."}
            
            query = """
                MATCH (a {name: $source_name}), (b {name: $target_name})
                MERGE (a)-[r:%s]->(b)
            """ % edge_type
            session.run(query, source_name=source_node_id, target_name=target_node_id)
            return {"status": "success", "action": action, "details": f"Edge '{edge_type}' from '{source_node_id}' to '{target_node_id}' added."}
        
        elif action == "query":
            if not properties or not properties.get("cypher"):
                 return {"status": "error", "message": "For 'query' action, a 'cypher' property is required."}
            
            result = session.run(properties["cypher"])
            return {"status": "success", "action": action, "data": [dict(record) for record in result]}

        else:
            return {"status": "error", "message": f"Unknown graph DB action: {action}"}

def query_knowledge_base(query: str) -> Dict[str, Any]:
    """
    Queries the integrated knowledge base (built with LlamaIndex)
    to find relevant documents, entities, and relationships.
    """
    print(f"[DataEngTool] Querying knowledge base for: '{query}'")
    if knowledge_base_index is None:
        return {"status": "error", "message": "Knowledge base index is not initialized."}

    query_engine = knowledge_base_index.as_query_engine()
    response = query_engine.query(query)

    # Format the response to be structured
    results = [
        {
            "text": str(response),
            "source_nodes": [
                {
                    "node_id": node.node.id_,
                    "text": node.node.get_text(),
                    "score": node.score,
                    "metadata": node.node.metadata,
                }
                for node in response.source_nodes
            ]
        }
    ]

    return {
        "status": "success",
        "query": query,
        "results": results,
    }

# --- Multimedia Data Processing ---
def extract_text_from_image(image_url: str, language_hint: str = "en") -> Dict[str, Any]:
    """
    Extract text from image using the cloud database service OCR capability.
    Integrates with Google Cloud Vision API through the CloudDatabaseService.
    """
    print(f"[DataEngTool-Multimedia] OCR: Extracting text from image: {image_url} (Language hint: {language_hint})")
    
    try:
        # Try to import and use the cloud database service
        from services.cloud_database_service import CloudDatabaseService
        
        db_service = CloudDatabaseService()
        result = db_service.extract_text_from_image(image_url, language_hint)
        
        return result
        
    except ImportError:
        print("[DataEngTool-Multimedia] CloudDatabaseService not available")
        return {
            "status": "error",
            "message": "CloudDatabaseService not available for OCR processing",
            "image_url": image_url,
            "language_hint": language_hint
        }
    except Exception as e:
        print(f"[DataEngTool-Multimedia] Error in OCR processing: {e}")
        return {
            "status": "error",
            "error": str(e),
            "image_url": image_url
        }

def extract_audio_transcript_from_video(video_url: str, language_hint: str = "en") -> Dict[str, Any]:
    """
    Extracts audio transcript from a video using speech-to-text services.
    
    TODO: Implement real video processing pipeline:
    - Video downloading and audio extraction
    - Speech-to-text API integration (Google Speech-to-Text, Azure Speech, etc.)
    - Multi-language support
    - Timestamp alignment
    """
    print(f"[DataEngTool-Multimedia] ASR: Attempting to extract transcript from video: {video_url} (Language hint: {language_hint})")
    
    # TODO: Implement real video transcript extraction
    # This should:
    # 1. Download or stream video content
    # 2. Extract audio track
    # 3. Use speech-to-text APIs
    # 4. Handle multiple speakers and languages
    # 5. Provide timestamp information

    return {
        "status": "error",
        "message": "Real video transcript extraction not yet implemented",
        "video_url": video_url,
        "language_hint": language_hint
    }

# Create FunctionTool instances
social_media_collector_tool = FunctionTool(
    func=social_media_collector,
)
run_nlp_pipeline_tool = FunctionTool(
    func=run_nlp_pipeline,
)
database_updater_tool = FunctionTool(
    func=database_updater,
)
manage_graph_db_tool = FunctionTool(
    func=manage_graph_db,
)
query_knowledge_base_tool = FunctionTool(
    func=query_knowledge_base,
)
extract_text_from_image_tool = FunctionTool(
    func=extract_text_from_image,
)
extract_audio_transcript_from_video_tool = FunctionTool(
    func=extract_audio_transcript_from_video,
)