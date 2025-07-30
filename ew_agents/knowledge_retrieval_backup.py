"""
Knowledge Retrieval System using LlamaIndex + MongoDB
====================================================

This module provides semantic search and retrieval-augmented generation (RAG)
capabilities for the Election Watch agents by integrating LlamaIndex with MongoDB.

Key Features:
- Vector embeddings for all knowledge collections
- Semantic search across narratives, techniques, and meta-narratives
- RAG-powered analysis and recommendations
- Context-aware knowledge retrieval for agent tools
"""

import os
import logging
from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass
import asyncio

# LlamaIndex imports
try:
    from llama_index.core import VectorStoreIndex, Document, Settings
    from llama_index.core.node_parser import SimpleNodeParser
    from llama_index.core.retrievers import VectorIndexRetriever
    from llama_index.core.query_engine import RetrieverQueryEngine
    from llama_index.core.response_synthesizers import ResponseMode
    from llama_index.embeddings.huggingface import HuggingFaceEmbedding
except ImportError:
    # Fallback for older llama-index versions
    from llama_index import VectorStoreIndex, Document, ServiceContext
    from llama_index.node_parser import SimpleNodeParser
    from llama_index.retrievers import VectorIndexRetriever
    from llama_index.query_engine import RetrieverQueryEngine
    from llama_index.response_synthesizers import ResponseMode
    from llama_index.embeddings.huggingface import HuggingFaceEmbedding
    
    # For older versions, we need to handle Settings differently
    class Settings:
        embed_model = None
        node_parser = None

# MongoDB imports
from pymongo import MongoClient
from pymongo.collection import Collection

# Environment
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)


@dataclass
class KnowledgeQuery:
    """Structured query for knowledge retrieval"""
    query_text: str
    collections: List[str] = None  # Specific collections to search
    max_results: int = 10
    similarity_threshold: float = 0.7
    context_type: str = "analysis"  # analysis, recommendation, detection


class KnowledgeRetriever:
    """
    Main knowledge retrieval system that integrates LlamaIndex with MongoDB
    for semantic search and RAG capabilities.
    """
    
    def __init__(self, 
                 mongodb_uri: str = None,
                 database_name: str = "knowledge",
                 embedding_model: str = "BAAI/bge-small-en-v1.5"):
        """
        Initialize the knowledge retrieval system.
        
        Args:
            mongodb_uri: MongoDB connection string (Atlas URI recommended)
            database_name: Name of the knowledge database
            embedding_model: HuggingFace embedding model to use
        """
        # Use MONGODB_ATLAS_URI as primary, MONGODB_URI as fallback for consistency
        self.mongodb_uri = mongodb_uri or os.getenv(
            "MONGODB_ATLAS_URI", 
            os.getenv("MONGODB_URI", "mongodb://localhost:27017")
        )
        self.database_name = database_name
        self.embedding_model_name = embedding_model
        
        # Initialize components
        self.client = None
        self.db = None
        self.embedding_model = None
        self.indexes = {}
        self.query_engines = {}
        
        # Knowledge collections
        self.collections = [
            "narratives",
            "disarm_techniques", 
            "meta_narratives",
            "threat_actors",
            "known_incidents",
            "tools_and_software",
            "mitigations"
        ]
        
        logger.info(f"Initialized KnowledgeRetriever with model: {embedding_model}")
    
    async def initialize(self):
        """Initialize MongoDB connection, embeddings, and build indexes"""
        try:
            # Connect to MongoDB
            self.client = MongoClient(self.mongodb_uri)
            self.db = self.client[self.database_name]
            
            # Initialize embedding model
            self.embedding_model = HuggingFaceEmbedding(model_name=self.embedding_model_name)
            
            # Configure LlamaIndex settings
            Settings.embed_model = self.embedding_model
            Settings.node_parser = SimpleNodeParser.from_defaults(chunk_size=512, chunk_overlap=50)
            
            # Build indexes for all collections
            await self._build_indexes()
            
            logger.info("KnowledgeRetriever initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize KnowledgeRetriever: {e}")
            raise
    
    async def _build_indexes(self):
        """Build vector indexes for all knowledge collections"""
        for collection_name in self.collections:
            try:
                logger.info(f"Building index for collection: {collection_name}")
                
                # Load documents from MongoDB collection
                documents = await self._load_documents_from_collection(collection_name)
                
                if not documents:
                    logger.warning(f"No documents found in collection: {collection_name}")
                    continue
                
                # Create vector index
                index = VectorStoreIndex.from_documents(
                    documents,
                    embed_model=self.embedding_model
                )
                
                # Create query engine
                query_engine = index.as_query_engine(
                    response_mode=ResponseMode.COMPACT,
                    similarity_top_k=10
                )
                
                self.indexes[collection_name] = index
                self.query_engines[collection_name] = query_engine
                
                logger.info(f"Successfully built index for {collection_name} with {len(documents)} documents")
                
            except Exception as e:
                logger.error(f"Failed to build index for {collection_name}: {e}")
    
    async def _load_documents_from_collection(self, collection_name: str) -> List[Document]:
        """Load documents from a MongoDB collection and convert to LlamaIndex format"""
        try:
            collection = self.db[collection_name]
            cursor = collection.find({})
            documents = []
            
            # MongoDB cursor iteration (not async)
            for doc in cursor:
                # Convert MongoDB document to LlamaIndex Document
                doc_id = str(doc.get("_id", ""))
                
                # Create text content based on collection type
                text_content = self._create_text_content(doc, collection_name)
                
                # Create metadata
                metadata = {
                    "collection": collection_name,
                    "doc_id": doc_id,
                    **{k: v for k, v in doc.items() if k != "_id" and isinstance(v, (str, int, float, bool))}
                }
                
                # Create LlamaIndex Document
                document = Document(
                    text=text_content,
                    metadata=metadata,
                    id_=doc_id
                )
                
                documents.append(document)
            
            return documents
            
        except Exception as e:
            logger.error(f"Failed to load documents from {collection_name}: {e}")
            return []
    
    def _create_text_content(self, doc: Dict[str, Any], collection_name: str) -> str:
        """Create searchable text content from MongoDB document based on collection type"""
        
        if collection_name == "narratives":
            return f"""
            Narrative ID: {doc.get('id', '')}
            Category: {doc.get('category', '')}
            Meta-Narrative: {doc.get('meta_narrative', '')}
            Scenario: {doc.get('scenario', '')}
            Key Indicators: {', '.join(doc.get('key_indicators_for_ai', []))}
            Primary Disarm Technique: {doc.get('primary_disarm_technique', '')}
            Common Platforms: {', '.join(doc.get('common_platforms', []))}
            """.strip()
        
        elif collection_name == "disarm_techniques":
            sub_techniques_text = ""
            if doc.get('sub_techniques'):
                sub_techniques_text = "\nSub-techniques: " + "; ".join([
                    f"{st.get('name', '')}: {st.get('summary', '')}" 
                    for st in doc.get('sub_techniques', [])
                ])
            
            return f"""
            Technique ID: {doc.get('disarm_id', '')}
            Name: {doc.get('name', '')}
            Tactic: {doc.get('tactic_id', '')}
            Summary: {doc.get('summary', '')}
            {sub_techniques_text}
            """.strip()
        
        elif collection_name == "meta_narratives":
            return f"""
            Meta-Narrative: {doc.get('meta_narrative', '')}
            Description: {doc.get('description', '')}
            Tactics Used: {', '.join(doc.get('tactics_used', []))}
            Common Locations: {', '.join(doc.get('common_locations', []))}
            Purpose/Effect: {doc.get('purpose_effect', '')}
            """.strip()
        
        elif collection_name == "threat_actors":
            return f"""
            Group: {doc.get('group_name', '')}
            Aliases: {doc.get('aliases', '')}
            Country: {doc.get('country', '')}
            Description: {doc.get('description', '')}
            Attribution: {doc.get('attribution', '')}
            """.strip()
        
        elif collection_name == "mitigations":
            return f"""
            Mitigation: {doc.get('mitigation_name', '')}
            Phase: {doc.get('phase', '')}
            Description: {doc.get('description', '')}
            """.strip()
        
        else:
            # Generic text creation for other collections
            text_fields = []
            for key, value in doc.items():
                if key != "_id" and isinstance(value, str) and value.strip():
                    text_fields.append(f"{key.replace('_', ' ').title()}: {value}")
            
            return "\n".join(text_fields)
    
    async def semantic_search(self, query: KnowledgeQuery) -> Dict[str, Any]:
        """
        Perform semantic search across knowledge collections
        
        Args:
            query: KnowledgeQuery object with search parameters
            
        Returns:
            Dictionary with search results from relevant collections
        """
        try:
            results = {}
            collections_to_search = query.collections or self.collections
            
            for collection_name in collections_to_search:
                if collection_name not in self.query_engines:
                    logger.warning(f"Query engine not found for collection: {collection_name}")
                    continue
                
                # Perform search
                query_engine = self.query_engines[collection_name]
                response = await asyncio.to_thread(
                    query_engine.query, 
                    query.query_text
                )
                
                # Extract relevant information
                results[collection_name] = {
                    "response": str(response),
                    "source_nodes": [
                        {
                            "text": node.text,
                            "metadata": node.metadata,
                            "score": getattr(node, 'score', None)
                        }
                        for node in response.source_nodes[:query.max_results]
                    ]
                }
            
            return results
            
        except Exception as e:
            logger.error(f"Semantic search failed: {e}")
            return {}
    
    async def get_narrative_recommendations(self, content_text: str) -> List[Dict[str, Any]]:
        """
        Get narrative detection and countermeasure recommendations
        
        Args:
            content_text: Text content to analyze
            
        Returns:
            List of recommendations with matched narratives and countermeasures
        """
        try:
            # Search for matching narratives
            narrative_query = KnowledgeQuery(
                query_text=content_text,
                collections=["narratives"],
                max_results=5,
                context_type="detection"
            )
            
            narrative_results = await self.semantic_search(narrative_query)
            
            recommendations = []
            
            if "narratives" in narrative_results:
                for node in narrative_results["narratives"]["source_nodes"]:
                    narrative_meta = node["metadata"]
                    
                    # Get corresponding disarm technique
                    technique_name = narrative_meta.get("primary_disarm_technique", "")
                    if technique_name:
                        technique_query = KnowledgeQuery(
                            query_text=technique_name,
                            collections=["disarm_techniques"],
                            max_results=1
                        )
                        technique_results = await self.semantic_search(technique_query)
                        
                        technique_info = None
                        if "disarm_techniques" in technique_results and technique_results["disarm_techniques"]["source_nodes"]:
                            technique_info = technique_results["disarm_techniques"]["source_nodes"][0]
                    
                    recommendation = {
                        "narrative_id": narrative_meta.get("id"),
                        "narrative_category": narrative_meta.get("category"),
                        "scenario": narrative_meta.get("scenario"),
                        "confidence_score": node.get("score", 0.0),
                        "key_indicators": narrative_meta.get("key_indicators_for_ai", []),
                        "disarm_technique": technique_info["metadata"] if technique_info else None,
                        "recommended_platforms": narrative_meta.get("common_platforms", [])
                    }
                    
                    recommendations.append(recommendation)
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Failed to get narrative recommendations: {e}")
            return []
    
    async def get_contextual_analysis(self, content_text: str, analysis_type: str = "comprehensive") -> Dict[str, Any]:
        """
        Get comprehensive contextual analysis using knowledge base
        
        Args:
            content_text: Content to analyze
            analysis_type: Type of analysis (comprehensive, threat_assessment, mitigation_focused)
            
        Returns:
            Comprehensive analysis with recommendations
        """
        try:
            # Determine collections to search based on analysis type
            if analysis_type == "threat_assessment":
                collections = ["narratives", "threat_actors", "known_incidents"]
            elif analysis_type == "mitigation_focused":
                collections = ["disarm_techniques", "mitigations", "meta_narratives"]
            else:  # comprehensive
                collections = self.collections
            
            query = KnowledgeQuery(
                query_text=content_text,
                collections=collections,
                max_results=3,
                context_type="analysis"
            )
            
            search_results = await self.semantic_search(query)
            
            # Get narrative recommendations
            narrative_recs = await self.get_narrative_recommendations(content_text)
            
            analysis = {
                "content_summary": content_text[:500] + "..." if len(content_text) > 500 else content_text,
                "analysis_type": analysis_type,
                "narrative_matches": narrative_recs,
                "knowledge_insights": search_results,
                "risk_indicators": self._extract_risk_indicators(search_results),
                "recommended_actions": self._generate_action_recommendations(search_results, narrative_recs)
            }
            
            return analysis
            
        except Exception as e:
            logger.error(f"Contextual analysis failed: {e}")
            return {"error": str(e)}
    
    def _extract_risk_indicators(self, search_results: Dict[str, Any]) -> List[str]:
        """Extract risk indicators from search results"""
        indicators = []
        
        # Extract from narratives
        if "narratives" in search_results:
            for node in search_results["narratives"]["source_nodes"]:
                key_indicators = node["metadata"].get("key_indicators_for_ai", [])
                indicators.extend(key_indicators)
        
        # Extract from threat actors
        if "threat_actors" in search_results:
            for node in search_results["threat_actors"]["source_nodes"]:
                group_name = node["metadata"].get("group_name", "")
                if group_name:
                    indicators.append(f"Potential {group_name} activity patterns")
        
        return list(set(indicators))  # Remove duplicates
    
    def _generate_action_recommendations(self, search_results: Dict[str, Any], narrative_recs: List[Dict[str, Any]]) -> List[str]:
        """Generate actionable recommendations based on analysis"""
        recommendations = []
        
        # From mitigations
        if "mitigations" in search_results:
            for node in search_results["mitigations"]["source_nodes"]:
                mitigation_name = node["metadata"].get("mitigation_name", "")
                if mitigation_name:
                    recommendations.append(f"Apply {mitigation_name} strategy")
        
        # From narrative matches
        for rec in narrative_recs[:3]:  # Top 3 recommendations
            if rec.get("disarm_technique"):
                technique_name = rec["disarm_technique"].get("name", "")
                if technique_name:
                    recommendations.append(f"Counter using {technique_name} approach")
        
        # Default recommendations
        if not recommendations:
            recommendations = [
                "Monitor content for escalation patterns",
                "Apply fact-checking and verification",
                "Engage with authoritative counter-narratives"
            ]
        
        return recommendations
    
    async def close(self):
        """Close MongoDB connection"""
        if self.client:
            self.client.close()
            logger.info("MongoDB connection closed")


# Global instance
knowledge_retriever = None

async def get_knowledge_retriever() -> KnowledgeRetriever:
    """Get or initialize the global knowledge retriever instance"""
    global knowledge_retriever
    
    if knowledge_retriever is None:
        knowledge_retriever = KnowledgeRetriever()
        await knowledge_retriever.initialize()
    
    return knowledge_retriever

async def search_knowledge(query_text: str, collections: Optional[List[str]] = None) -> Dict[str, Any]:
    """Convenience function for knowledge search"""
    retriever = await get_knowledge_retriever()
    query = KnowledgeQuery(query_text=query_text, collections=collections)
    return await retriever.semantic_search(query)

async def analyze_content(content_text: str, analysis_type: str = "comprehensive") -> Dict[str, Any]:
    """Convenience function for content analysis"""
    retriever = await get_knowledge_retriever()
    return await retriever.get_contextual_analysis(content_text, analysis_type) 