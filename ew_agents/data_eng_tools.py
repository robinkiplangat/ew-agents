"""
Data Engineering Tools for Election Watch Agents
=================================================

Enhanced tools that integrate with MongoDB knowledge base via LlamaIndex
for real-time content analysis, narrative detection, and threat assessment.
"""

import json
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime
import asyncio

# Import knowledge retrieval system
try:
    from .knowledge_retrieval import get_knowledge_retriever, analyze_content, search_knowledge, KnowledgeQuery
    KNOWLEDGE_AVAILABLE = True
except ImportError:
    # Fallback for direct execution
    try:
        from knowledge_retrieval import get_knowledge_retriever, analyze_content, search_knowledge, KnowledgeQuery
        KNOWLEDGE_AVAILABLE = True
    except ImportError:
        # Graceful fallback when knowledge system is unavailable
        KNOWLEDGE_AVAILABLE = False
        logger.warning("Knowledge retrieval system not available - using fallback mode")

logger = logging.getLogger(__name__)

async def analyze_content_nlp(content: str, analysis_type: str = "comprehensive") -> Dict[str, Any]:
    """
    Perform NLP analysis on content using knowledge-enhanced AI models.
    
    Args:
        content: Text content to analyze
        analysis_type: Type of analysis (comprehensive, threat_assessment, mitigation_focused)
    
    Returns:
        Dict containing comprehensive analysis results
    """
    try:
        logger.info(f"Starting NLP analysis for content of length {len(content)}")
        
        # Get knowledge-enhanced analysis
        analysis_result = await analyze_content(content, analysis_type)
        
        if "error" in analysis_result:
            logger.error(f"Analysis failed: {analysis_result['error']}")
            return {
                "success": False,
                "error": analysis_result["error"],
                "timestamp": datetime.now().isoformat()
            }
        
        # Extract key metrics
        risk_score = calculate_risk_score(analysis_result)
        sentiment_analysis = analyze_sentiment(content)
        language_patterns = detect_language_patterns(content)
        
        # Compile comprehensive results
        nlp_results = {
            "success": True,
            "analysis_type": analysis_type,
            "content_summary": analysis_result.get("content_summary", ""),
            "risk_assessment": {
                "overall_risk_score": risk_score,
                "risk_level": get_risk_level(risk_score),
                "risk_indicators": analysis_result.get("risk_indicators", [])
            },
            "narrative_detection": {
                "matched_narratives": analysis_result.get("narrative_matches", []),
                "narrative_count": len(analysis_result.get("narrative_matches", [])),
                "top_narrative_categories": extract_top_categories(analysis_result.get("narrative_matches", []))
            },
            "knowledge_insights": analysis_result.get("knowledge_insights", {}),
            "recommended_actions": analysis_result.get("recommended_actions", []),
            "linguistic_analysis": {
                "sentiment": sentiment_analysis,
                "language_patterns": language_patterns,
                "content_length": len(content),
                "word_count": len(content.split())
            },
            "timestamp": datetime.now().isoformat()
        }
        
        logger.info(f"NLP analysis completed successfully. Risk score: {risk_score}")
        return nlp_results
        
    except Exception as e:
        logger.error(f"NLP analysis failed: {str(e)}")
        return {
            "success": False,
            "error": f"Analysis failed: {str(e)}",
            "timestamp": datetime.now().isoformat()
        }

async def detect_narrative_patterns(content: str, platform: str = "unknown") -> Dict[str, Any]:
    """
    Detect known narrative patterns in content using the knowledge base.
    
    Args:
        content: Content to analyze
        platform: Platform where content was found
    
    Returns:
        Dict with detected narrative patterns and recommendations
    """
    try:
        logger.info(f"Detecting narrative patterns for {platform} content")
        
        # Get knowledge retriever if available
        if KNOWLEDGE_AVAILABLE:
            try:
                retriever = await get_knowledge_retriever()
                # Get narrative recommendations
                narrative_recs = await retriever.get_narrative_recommendations(content)
            except Exception as e:
                logger.warning(f"Knowledge retriever failed: {e}, using fallback")
                narrative_recs = {"narratives": [], "confidence": 0.1}
        else:
            narrative_recs = {"narratives": [], "confidence": 0.1}
        
        # Search for platform-specific patterns if knowledge available  
        if KNOWLEDGE_AVAILABLE and 'retriever' in locals():
            try:
                platform_query = KnowledgeQuery(
                    query_text=f"{content} {platform}",
                    collections=["narratives", "meta_narratives"],
                    max_results=5,
                    context_type="detection"
                )
                
                platform_results = await retriever.semantic_search(platform_query)
            except Exception as e:
                logger.warning(f"Platform search failed: {e}")
                platform_results = {"narratives": [], "meta_narratives": []}
        else:
            platform_results = {"narratives": [], "meta_narratives": []}
        
        # Analyze patterns
        detected_patterns = []
        confidence_scores = []
        
        for rec in narrative_recs:
            pattern = {
                "narrative_id": rec.get("narrative_id"),
                "category": rec.get("narrative_category"),
                "confidence": rec.get("confidence_score", 0.0),
                "key_indicators": rec.get("key_indicators", []),
                "scenario_match": rec.get("scenario", ""),
                "recommended_platforms": rec.get("recommended_platforms", []),
                "disarm_technique": rec.get("disarm_technique", {}).get("name", "") if rec.get("disarm_technique") else "",
                "platform_relevance": platform in rec.get("recommended_platforms", [])
            }
            detected_patterns.append(pattern)
            confidence_scores.append(rec.get("confidence_score", 0.0))
        
        # Calculate overall detection confidence
        avg_confidence = sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0.0
        
        return {
            "success": True,
            "platform": platform,
            "detection_summary": {
                "patterns_detected": len(detected_patterns),
                "average_confidence": avg_confidence,
                "high_confidence_count": len([s for s in confidence_scores if s > 0.8])
            },
            "detected_patterns": detected_patterns,
            "platform_specific_insights": platform_results,
            "recommended_monitoring": generate_monitoring_recommendations(detected_patterns),
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Narrative pattern detection failed: {str(e)}")
        return {
            "success": False,
            "error": f"Pattern detection failed: {str(e)}",
            "timestamp": datetime.now().isoformat()
        }

async def extract_key_entities(content: str) -> Dict[str, Any]:
    """
    Extract key entities (people, organizations, locations) relevant to election monitoring.
    
    Args:
        content: Text content to analyze
        
    Returns:
        Dict with extracted entities and their contextual relevance
    """
    try:
        logger.info("Extracting key entities from content")
        
        # Search for entity-related knowledge if available
        if KNOWLEDGE_AVAILABLE:
            try:
                entity_query = KnowledgeQuery(
                    query_text=content,
                    collections=["threat_actors", "known_incidents", "narratives"],
                    max_results=10,
                    context_type="analysis"
                )
                
                knowledge_results = await search_knowledge(content, ["threat_actors", "known_incidents"])
            except Exception as e:
                logger.warning(f"Knowledge search failed: {e}, using fallback")
                knowledge_results = {}
        else:
            knowledge_results = {}
        
        # Extract entities from knowledge matches
        entities = {
            "threat_actors": [],
            "locations": [],
            "organizations": [],
            "incidents": [],
            "techniques": []
        }
        
        # Process threat actors
        if "threat_actors" in knowledge_results:
            for node in knowledge_results["threat_actors"]["source_nodes"]:
                metadata = node["metadata"]
                entities["threat_actors"].append({
                    "name": metadata.get("group_name", ""),
                    "aliases": metadata.get("aliases", ""),
                    "country": metadata.get("country", ""),
                    "confidence": getattr(node, 'score', 0.0)
                })
        
        # Process incidents
        if "known_incidents" in knowledge_results:
            for node in knowledge_results["known_incidents"]["source_nodes"]:
                metadata = node["metadata"]
                entities["incidents"].append({
                    "name": metadata.get("incident_name", ""),
                    "description": metadata.get("description", ""),
                    "start_date": metadata.get("start_date", ""),
                    "confidence": getattr(node, 'score', 0.0)
                })
        
        # Extract basic linguistic entities
        words = content.split()
        
        # Simple entity extraction (can be enhanced with NER models)
        potential_locations = extract_location_mentions(content)
        potential_organizations = extract_organization_mentions(content)
        
        entities["locations"] = potential_locations
        entities["organizations"] = potential_organizations
        
        return {
            "success": True,
            "entities": entities,
            "entity_summary": {
                "total_entities": sum(len(v) for v in entities.values()),
                "threat_actors_found": len(entities["threat_actors"]),
                "incidents_referenced": len(entities["incidents"]),
                "locations_mentioned": len(entities["locations"])
            },
            "contextual_relevance": calculate_entity_relevance(entities),
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Entity extraction failed: {str(e)}")
        return {
            "success": False,
            "error": f"Entity extraction failed: {str(e)}",
            "timestamp": datetime.now().isoformat()
        }

async def categorize_content_type(content: str, metadata: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    Categorize content type and determine appropriate analysis approach.
    
    Args:
        content: Content to categorize
        metadata: Additional metadata about the content
        
    Returns:
        Dict with content categorization and analysis recommendations
    """
    try:
        logger.info("Categorizing content type")
        
        metadata = metadata or {}
        
        # Search for similar content patterns in knowledge base
        category_query = KnowledgeQuery(
            query_text=content[:500],  # Use first 500 chars for categorization
            collections=["narratives", "meta_narratives"],
            max_results=3,
            context_type="analysis"
        )
        
        category_results = await search_knowledge(content[:500], ["narratives", "meta_narratives"])
        
        # Determine content characteristics
        content_length = len(content)
        word_count = len(content.split())
        
        # Basic content type detection
        content_type = determine_basic_content_type(content, metadata)
        
        # Extract categories from knowledge matches
        suggested_categories = []
        narrative_types = []
        
        if "narratives" in category_results:
            for node in category_results["narratives"]["source_nodes"]:
                category = node["metadata"].get("category", "")
                if category and category not in suggested_categories:
                    suggested_categories.append(category)
        
        if "meta_narratives" in category_results:
            for node in category_results["meta_narratives"]["source_nodes"]:
                meta_narrative = node["metadata"].get("meta_narrative", "")
                if meta_narrative and meta_narrative not in narrative_types:
                    narrative_types.append(meta_narrative)
        
        # Determine analysis priority
        priority = calculate_analysis_priority(content, suggested_categories, narrative_types)
        
        return {
            "success": True,
            "content_type": content_type,
            "characteristics": {
                "length": content_length,
                "word_count": word_count,
                "estimated_read_time": word_count // 200,  # Average reading speed
                "complexity_score": calculate_complexity_score(content)
            },
            "suggested_categories": suggested_categories,
            "narrative_types": narrative_types,
            "analysis_priority": priority,
            "recommended_analysis_type": determine_recommended_analysis(content_type, priority),
            "metadata": metadata,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Content categorization failed: {str(e)}")
        return {
            "success": False,
            "error": f"Content categorization failed: {str(e)}",
            "timestamp": datetime.now().isoformat()
        }

# Helper functions

def calculate_risk_score(analysis_result: Dict[str, Any]) -> float:
    """Calculate overall risk score based on analysis results"""
    score = 0.0
    
    # Base score from narrative matches
    narrative_matches = analysis_result.get("narrative_matches", [])
    if narrative_matches:
        avg_confidence = sum(m.get("confidence_score", 0.0) for m in narrative_matches) / len(narrative_matches)
        score += avg_confidence * 0.4
    
    # Risk indicators weight
    risk_indicators = analysis_result.get("risk_indicators", [])
    score += min(len(risk_indicators) * 0.1, 0.3)
    
    # Knowledge insights weight
    knowledge_insights = analysis_result.get("knowledge_insights", {})
    if knowledge_insights:
        score += 0.2
    
    return min(score, 1.0)

def get_risk_level(risk_score: float) -> str:
    """Convert risk score to categorical level"""
    if risk_score >= 0.8:
        return "HIGH"
    elif risk_score >= 0.5:
        return "MEDIUM"
    elif risk_score >= 0.2:
        return "LOW"
    else:
        return "MINIMAL"

def extract_top_categories(narrative_matches: List[Dict[str, Any]]) -> List[str]:
    """Extract top narrative categories from matches"""
    categories = {}
    for match in narrative_matches:
        category = match.get("narrative_category", "Unknown")
        categories[category] = categories.get(category, 0) + 1
    
    return sorted(categories.keys(), key=lambda x: categories[x], reverse=True)[:3]

def analyze_sentiment(content: str) -> Dict[str, Any]:
    """Basic sentiment analysis"""
    # Simple keyword-based sentiment analysis
    positive_words = ["good", "great", "excellent", "positive", "success", "win", "victory"]
    negative_words = ["bad", "terrible", "awful", "negative", "fail", "loss", "defeat", "rigged", "corrupt"]
    
    words = content.lower().split()
    positive_count = sum(1 for word in words if word in positive_words)
    negative_count = sum(1 for word in words if word in negative_words)
    
    total_sentiment_words = positive_count + negative_count
    
    if total_sentiment_words == 0:
        sentiment = "neutral"
        confidence = 0.5
    else:
        sentiment_score = (positive_count - negative_count) / total_sentiment_words
        if sentiment_score > 0.2:
            sentiment = "positive"
        elif sentiment_score < -0.2:
            sentiment = "negative"
        else:
            sentiment = "neutral"
        confidence = abs(sentiment_score)
    
    return {
        "sentiment": sentiment,
        "confidence": confidence,
        "positive_indicators": positive_count,
        "negative_indicators": negative_count
    }

def detect_language_patterns(content: str) -> Dict[str, Any]:
    """Detect language patterns relevant to disinformation"""
    patterns = {
        "urgency_markers": ["urgent", "immediate", "now", "quickly", "emergency"],
        "authority_claims": ["official", "confirmed", "verified", "expert", "insider"],
        "emotional_appeals": ["shocking", "amazing", "terrible", "unbelievable", "scandal"],
        "conspiratorial_language": ["secret", "hidden", "cover-up", "conspiracy", "they don't want you to know"]
    }
    
    content_lower = content.lower()
    detected_patterns = {}
    
    for pattern_type, keywords in patterns.items():
        matches = [keyword for keyword in keywords if keyword in content_lower]
        detected_patterns[pattern_type] = {
            "matches": matches,
            "count": len(matches)
        }
    
    return detected_patterns

def generate_monitoring_recommendations(detected_patterns: List[Dict[str, Any]]) -> List[str]:
    """Generate monitoring recommendations based on detected patterns"""
    recommendations = []
    
    high_confidence_patterns = [p for p in detected_patterns if p.get("confidence", 0) > 0.7]
    
    if high_confidence_patterns:
        recommendations.append("Implement enhanced monitoring for high-confidence narrative matches")
    
    categories = set(p.get("category", "") for p in detected_patterns)
    if "Voter Manipulation" in categories:
        recommendations.append("Monitor for voter suppression or manipulation tactics")
    
    if "Identity and Polarization" in categories:
        recommendations.append("Watch for divisive content targeting specific communities")
    
    if not recommendations:
        recommendations.append("Continue standard monitoring protocols")
    
    return recommendations

def extract_location_mentions(content: str) -> List[Dict[str, Any]]:
    """Extract potential location mentions"""
    # Simple keyword-based location extraction
    known_locations = ["nigeria", "lagos", "abuja", "kano", "ibadan", "ghana", "kenya", "senegal"]
    content_lower = content.lower()
    
    locations = []
    for location in known_locations:
        if location in content_lower:
            locations.append({
                "name": location.title(),
                "type": "location",
                "context": "election monitoring region"
            })
    
    return locations

def extract_organization_mentions(content: str) -> List[Dict[str, Any]]:
    """Extract potential organization mentions"""
    # Simple keyword-based organization extraction
    known_orgs = ["inec", "apc", "pdp", "lp", "nnpp", "eu", "un", "ecowas"]
    content_lower = content.lower()
    
    organizations = []
    for org in known_orgs:
        if org in content_lower:
            organizations.append({
                "name": org.upper(),
                "type": "organization",
                "context": "political/electoral entity"
            })
    
    return organizations

def calculate_entity_relevance(entities: Dict[str, List]) -> Dict[str, float]:
    """Calculate relevance scores for different entity types"""
    relevance = {}
    
    for entity_type, entity_list in entities.items():
        if entity_list:
            # Simple relevance based on count and confidence
            avg_confidence = sum(e.get("confidence", 0.5) for e in entity_list) / len(entity_list)
            relevance[entity_type] = min(avg_confidence * len(entity_list) / 5, 1.0)
        else:
            relevance[entity_type] = 0.0
    
    return relevance

def determine_basic_content_type(content: str, metadata: Dict[str, Any]) -> str:
    """Determine basic content type"""
    if metadata.get("source_type"):
        return metadata["source_type"]
    
    content_lower = content.lower()
    
    if any(keyword in content_lower for keyword in ["breaking:", "news:", "report:"]):
        return "news_article"
    elif any(keyword in content_lower for keyword in ["vote", "election", "candidate"]):
        return "election_content"
    elif len(content.split()) < 50:
        return "social_media_post"
    elif any(keyword in content_lower for keyword in ["analysis", "research", "study"]):
        return "analytical_content"
    else:
        return "general_content"

def calculate_analysis_priority(content: str, categories: List[str], narrative_types: List[str]) -> str:
    """Calculate analysis priority based on content characteristics"""
    high_priority_categories = ["Voter Manipulation", "Voter Intimidation", "Undermining Electoral Institutions"]
    high_priority_narratives = ["Rigged Elections / Stolen Mandate", "Tech Manipulation Narratives"]
    
    if any(cat in high_priority_categories for cat in categories):
        return "HIGH"
    elif any(nar in high_priority_narratives for nar in narrative_types):
        return "HIGH"
    elif categories or narrative_types:
        return "MEDIUM"
    else:
        return "LOW"

def determine_recommended_analysis(content_type: str, priority: str) -> str:
    """Determine recommended analysis type based on content and priority"""
    if priority == "HIGH":
        return "comprehensive"
    elif content_type in ["election_content", "news_article"]:
        return "threat_assessment"
    else:
        return "comprehensive"

def calculate_complexity_score(content: str) -> float:
    """Calculate content complexity score"""
    words = content.split()
    if not words:
        return 0.0
    
    # Simple complexity metrics
    avg_word_length = sum(len(word) for word in words) / len(words)
    sentence_count = content.count('.') + content.count('!') + content.count('?')
    avg_sentence_length = len(words) / max(sentence_count, 1)
    
    # Normalize to 0-1 scale
    complexity = min((avg_word_length - 3) / 7 + (avg_sentence_length - 10) / 20, 1.0)
    return max(complexity, 0.0)

def extract_entities_fallback(text: str) -> List[Dict[str, Any]]:
    """Fallback entity extraction using simple patterns"""
    import re
    entities = []
    
    # Extract @mentions
    mentions = re.findall(r'@(\w+)', text)
    for mention in mentions:
        entities.append({"text": mention, "type": "PERSON", "confidence": 0.8})
    
    # Extract hashtags
    hashtags = re.findall(r'#(\w+)', text)
    for hashtag in hashtags:
        entities.append({"text": hashtag, "type": "EVENT", "confidence": 0.7})
    
    # Extract political parties (Nigeria-specific)
    parties = re.findall(r'\b(APC|PDP|Labour|NNPP|SDP)\b', text, re.IGNORECASE)
    for party in parties:
        entities.append({"text": party, "type": "ORG", "confidence": 0.9})
    
    # Extract locations (Nigeria-specific)
    locations = re.findall(r'\b(Nigeria|Lagos|Abuja|Kano|Rivers|Kaduna|Ogun|Imo|Anambra)\b', text, re.IGNORECASE)
    for location in locations:
        entities.append({"text": location, "type": "GPE", "confidence": 0.8})
    
    return entities

def analyze_sentiment_fallback(text: str) -> str:
    """Fallback sentiment analysis using keyword matching"""
    positive_words = ['good', 'great', 'excellent', 'support', 'vote', 'trust', 'hope', 'progress']
    negative_words = ['bad', 'corrupt', 'rigged', 'fraud', 'fake', 'ghost', 'manipulation', 'theft']
    
    text_lower = text.lower()
    positive_count = sum(1 for word in positive_words if word in text_lower)
    negative_count = sum(1 for word in negative_words if word in text_lower)
    
    if positive_count > negative_count:
        return "positive"
    elif negative_count > positive_count:
        return "negative"
    else:
        return "neutral"

def detect_risk_patterns_fallback(text: str) -> List[str]:
    """Fallback risk pattern detection"""
    risk_indicators = []
    text_lower = text.lower()
    
    if any(word in text_lower for word in ['rigged', 'fraud', 'manipulation', 'fake']):
        risk_indicators.append("election_fraud_claims")
    
    if any(word in text_lower for word in ['violence', 'fight', 'war', 'attack']):
        risk_indicators.append("violence_incitement")
    
    if any(word in text_lower for word in ['ghost', 'dead', 'multiple', 'duplicate']):
        risk_indicators.append("voter_fraud_claims")
    
    return risk_indicators

# =============================================================================
# Legacy Tool Wrappers for Backwards Compatibility
# =============================================================================

def social_media_collector(platform: str, query: str, count: int = 10) -> Dict[str, Any]:
    """
    Legacy wrapper for social media collection.
    TODO: Implement real social media API integrations.
    """
    logger.info(f"[Legacy] Collecting {count} items from {platform} with query: '{query}'")
    return {
        "status": "stub",
        "message": f"Social media collection from {platform} not yet implemented",
        "platform": platform,
        "query": query,
        "requested_count": count
    }

def run_nlp_pipeline(text: str, language: str = "en") -> Dict[str, Any]:
    """
    Legacy wrapper that runs the enhanced NLP analysis.
    """
    logger.info(f"[Legacy] Running NLP pipeline on text (lang: {language})")
    
    try:
        # Handle uvloop compatibility issue
        try:
            loop = asyncio.get_running_loop()
            # We're in an async context - use thread executor to avoid uvloop issues
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(asyncio.run, analyze_content_nlp(text, "comprehensive"))
                result = future.result(timeout=30)
        except RuntimeError:
            # No running loop, safe to create new one
            result = asyncio.run(analyze_content_nlp(text, "comprehensive"))
        except Exception as loop_error:
            # Fallback to synchronous analysis if async fails
            logger.warning(f"Async analysis failed ({loop_error}), using fallback")
            result = {
                "success": True,
                "entities": extract_entities_fallback(text),
                "sentiment": analyze_sentiment_fallback(text),
                "language": language,
                "complexity_score": calculate_complexity_score(text),
                "risk_indicators": detect_risk_patterns_fallback(text),
                "fallback_mode": True
            }
        
        return result
    except Exception as e:
        logger.error(f"NLP pipeline failed: {e}")
        return {
            "success": False,
            "error": f"NLP pipeline failed: {str(e)}",
            "original_text": text,
            "specified_language": language
        }

def extract_text_from_image(image_url: str, language_hint: str = "en") -> Dict[str, Any]:
    """
    Legacy wrapper for OCR processing.
    TODO: Implement real OCR integration.
    """
    logger.info(f"[Legacy] OCR: Extracting text from image: {image_url}")
    return {
        "status": "stub",
        "message": "OCR processing not yet implemented",
        "image_url": image_url,
        "language_hint": language_hint
    }

def extract_audio_transcript_from_video(video_url: str, language_hint: str = "en") -> Dict[str, Any]:
    """
    Legacy wrapper for video transcript extraction.
    TODO: Implement real video processing.
    """
    logger.info(f"[Legacy] ASR: Extracting transcript from video: {video_url}")
    return {
        "status": "stub",
        "message": "Video transcript extraction not yet implemented",
        "video_url": video_url,
        "language_hint": language_hint
    }

def database_updater(collection_name: str, data: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Legacy wrapper for database updates.
    TODO: Implement real database operations.
    """
    logger.info(f"[Legacy] Updating collection '{collection_name}' with {len(data)} records")
    return {
        "status": "stub",
        "message": "Database operations not yet implemented",
        "collection_name": collection_name,
        "record_count": len(data)
    }

def manage_graph_db(action: str, **kwargs) -> Dict[str, Any]:
    """
    Legacy wrapper for graph database management.
    TODO: Implement real graph DB operations.
    """
    logger.info(f"[Legacy] Graph DB action: {action}")
    return {
        "status": "stub",
        "message": "Graph database operations not yet implemented",
        "action": action,
        "parameters": kwargs
    }

def query_knowledge_base(query: str) -> Dict[str, Any]:
    """
    Legacy wrapper that uses the new knowledge retrieval system.
    """
    logger.info(f"[Legacy] Querying knowledge base: '{query}'")
    
    try:
        # Run the async function in a synchronous context
        import asyncio
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        result = loop.run_until_complete(search_knowledge(query))
        loop.close()
        
        # Format for legacy compatibility
        formatted_results = []
        for collection, data in result.items():
            if data.get("source_nodes"):
                for node in data["source_nodes"]:
                    formatted_results.append({
                        "collection": collection,
                        "text": node.get("text", ""),
                        "metadata": node.get("metadata", {}),
                        "score": node.get("score", 0.0)
                    })
        
        return {
            "status": "success",
            "query": query,
            "results": formatted_results[:10],  # Limit to top 10
            "total_found": len(formatted_results)
        }
        
    except Exception as e:
        logger.error(f"Knowledge base query failed: {e}")
        return {
            "status": "error",
            "message": f"Knowledge base query failed: {str(e)}",
            "query": query
        }

# =============================================================================
# Tool Registration for Agent Framework
# =============================================================================

try:
    from google.adk.tools import FunctionTool
    
    # Create FunctionTool instances for the agent framework
    social_media_collector_tool = FunctionTool(func=social_media_collector)
    run_nlp_pipeline_tool = FunctionTool(func=run_nlp_pipeline)
    extract_text_from_image_tool = FunctionTool(func=extract_text_from_image)
    extract_audio_transcript_from_video_tool = FunctionTool(func=extract_audio_transcript_from_video)
    database_updater_tool = FunctionTool(func=database_updater)
    manage_graph_db_tool = FunctionTool(func=manage_graph_db)
    query_knowledge_base_tool = FunctionTool(func=query_knowledge_base)
    
    # New enhanced tools
    analyze_content_nlp_tool = FunctionTool(func=lambda content, analysis_type="comprehensive": 
                                          asyncio.run(analyze_content_nlp(content, analysis_type)))
    detect_narrative_patterns_tool = FunctionTool(func=lambda content, platform="unknown": 
                                                 asyncio.run(detect_narrative_patterns(content, platform)))
    extract_key_entities_tool = FunctionTool(func=lambda content: 
                                            asyncio.run(extract_key_entities(content)))
    
    logger.info("FunctionTool instances created successfully")
    
except ImportError:
    logger.warning("Google ADK FunctionTool not available - tools will not be registered")
    # Create dummy objects to prevent AttributeError
    social_media_collector_tool = None
    run_nlp_pipeline_tool = None
    extract_text_from_image_tool = None
    extract_audio_transcript_from_video_tool = None
    database_updater_tool = None
    manage_graph_db_tool = None
    query_knowledge_base_tool = None
    analyze_content_nlp_tool = None
    detect_narrative_patterns_tool = None
    extract_key_entities_tool = None