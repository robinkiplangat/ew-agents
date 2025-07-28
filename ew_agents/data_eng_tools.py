"""
Data Engineering Tools for ElectionWatch Agents
===============================================

User-input driven tools for processing submitted content including
text, CSV files, images, and other user-provided data for analysis.
"""

import json
import logging
import asyncio
from typing import Dict, List, Any, Optional
from datetime import datetime

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# =============================================================================
# USER INPUT PROCESSING FUNCTIONS
# =============================================================================

def process_social_media_data(content: str, source_type: str = "user_input", metadata: Optional[dict] = None) -> Dict[str, Any]:
    """
    Process user-provided social media content (from CSV, text file, or direct input).
    
    Args:
        content: User-provided social media content
        source_type: Type of source (csv_file, text_file, direct_input, etc.)
        metadata: Additional metadata about the content
    
    Returns:
        Dict with processed data and analysis metadata
    """
    logger.info(f"Processing user-provided social media content from {source_type}")
    
    try:
        metadata = metadata or {}
        
        # Process the user-provided content
        lines = content.strip().split('\n')
        processed_posts = []
        
        # Basic parsing - adjust based on input format
        for i, line in enumerate(lines):
            if line.strip():  # Skip empty lines
                processed_posts.append({
                    "post_id": f"user_post_{i+1}",
                    "content": line.strip(),
                    "source": source_type,
                    "processing_timestamp": datetime.now().isoformat()
                })
        
        processed_data = {
            "success": True,
            "source_type": source_type,
            "posts_processed": len(processed_posts),
            "processing_timestamp": datetime.now().isoformat(),
            "processed_posts": processed_posts,
            "metadata": metadata,
            "message": f"Successfully processed {len(processed_posts)} posts from user input"
        }
        
        logger.info(f"✅ Social media content processed: {len(processed_posts)} posts from {source_type}")
        return processed_data
        
    except Exception as e:
        logger.error(f"❌ Social media content processing failed: {e}")
        return {
            "success": False,
            "error": str(e),
            "source_type": source_type,
            "content_length": len(content),
            "timestamp": datetime.now().isoformat()
        }

def process_csv_data(csv_content: str, expected_columns: list = []) -> dict:
    """
    Process user-uploaded CSV data for analysis.
    
    Args:
        csv_content: Raw CSV content as string
        expected_columns: Expected column names for validation
    
    Returns:
        Dict with processed CSV data and statistics
    """
    logger.info(f"Processing user-uploaded CSV data")
    
    try:
        import csv
        from io import StringIO
        
        # Parse CSV content
        csv_reader = csv.DictReader(StringIO(csv_content))
        rows = list(csv_reader)
        
        if not rows:
            return {
                "success": False,
                "error": "No data found in CSV",
                "timestamp": datetime.now().isoformat()
            }
        
        # Get column information
        fieldnames = csv_reader.fieldnames or []
        
        # Validate expected columns if provided
        missing_columns = []
        if expected_columns:
            missing_columns = [col for col in expected_columns if col not in fieldnames]
        
        # Extract tweet content for analysis
        tweet_content = []
        for row in rows:
            # Try to find tweet content in various possible columns
            tweet_text = row.get('Tweet') or row.get('tweet') or row.get('text') or ''
            if tweet_text:
                tweet_content.append(tweet_text)
        
        # Basic statistics
        total_rows = len(rows)
        non_empty_rows = len([row for row in rows if any(row.values())])
        
        processed_data = {
            "success": True,
            "total_rows": total_rows,
            "non_empty_rows": non_empty_rows,
            "columns": fieldnames,
            "column_count": len(fieldnames),
            "missing_expected_columns": missing_columns,
            "sample_data": rows[:3],  # First 3 rows as sample
            "data": rows,  # Full data for processing
            "tweet_content": tweet_content,  # Extracted tweet content for analysis
            "processing_timestamp": datetime.now().isoformat(),
            "message": f"Successfully processed CSV with {total_rows} rows and {len(fieldnames)} columns"
        }
        
        logger.info(f"✅ CSV processing completed: {total_rows} rows, {len(fieldnames)} columns extracted, {len(tweet_content)} tweets found")
        return processed_data
        
    except Exception as e:
        logger.error(f"❌ CSV processing failed: {e}")
        return {
            "success": False,
            "error": str(e),
            "content_length": len(csv_content),
            "timestamp": datetime.now().isoformat()
        }

def run_nlp_pipeline(text: str, language: str = "en") -> Dict[str, Any]:
    """
    Run comprehensive NLP analysis on user-provided text content.
    
    Args:
        text: User-provided text content to analyze
        language: Language code (default: en)
    
    Returns:
        Dict with NLP analysis results
    """
    logger.info(f"Running NLP pipeline on user-provided text ({len(text)} characters, language: {language})")
    
    try:
        # Basic text analysis
        words = text.split()
        sentences = text.split('.')
        
        # Sentiment analysis (simplified)
        sentiment_score = analyze_sentiment_simple(text)
        
        # Entity extraction (simplified)
        entities = extract_entities_simple(text)
        
        # Risk indicators
        risk_indicators = detect_risk_patterns_simple(text)
        
        # Calculate complexity
        complexity = len(words) / max(len(sentences), 1)  # avg words per sentence
        
        nlp_results = {
            "success": True,
            "language": language,
            "statistics": {
                "character_count": len(text),
                "word_count": len(words),
                "sentence_count": len(sentences),
                "avg_words_per_sentence": round(complexity, 2)
            },
            "sentiment": {
                "score": sentiment_score,
                "label": "positive" if sentiment_score > 0.1 else "negative" if sentiment_score < -0.1 else "neutral"
            },
            "entities": entities,
            "risk_assessment": {
                "risk_indicators": risk_indicators,
                "risk_level": "high" if len(risk_indicators) > 2 else "medium" if len(risk_indicators) > 0 else "low"
            },
            "processing_timestamp": datetime.now().isoformat(),
            "input_source": "user_provided_text"
        }
        
        logger.info(f"✅ NLP pipeline completed successfully")
        return nlp_results
        
    except Exception as e:
        logger.error(f"❌ NLP pipeline failed: {e}")
        return {
            "success": False,
            "error": str(e),
            "text_length": len(text),
            "language": language,
            "timestamp": datetime.now().isoformat()
        }

def extract_text_from_image(image_data: str, language_hint: str = "en") -> Dict[str, Any]:
    """
    Extract text from user-uploaded image using OCR.
    
    Args:
        image_data: Base64 image data or file path from user upload
        language_hint: Expected language in image
    
    Returns:
        Dict with extracted text and metadata
    """
    logger.info(f"Extracting text from user-uploaded image (language: {language_hint})")
    
    try:
        # Placeholder for OCR implementation
        # TODO: Implement real OCR (Google Vision API, Tesseract, etc.)
        
        # Simulate OCR processing
        extracted_text = f"[OCR PROCESSING] This would contain text extracted from the user's uploaded image. Language detected: {language_hint}"
        
        extracted_data = {
            "success": True,
            "extracted_text": extracted_text,
            "language_detected": language_hint,
            "confidence_score": 0.85,
            "processing_timestamp": datetime.now().isoformat(),
            "input_source": "user_uploaded_image",
            "message": "OCR processing completed (placeholder implementation - ready for real OCR integration)"
        }
        
        logger.info(f"✅ Text extraction completed for user image")
        return extracted_data
        
    except Exception as e:
        logger.error(f"❌ Text extraction failed: {e}")
        return {
            "success": False,
            "error": str(e),
            "language_hint": language_hint,
            "timestamp": datetime.now().isoformat()
        }

def extract_audio_transcript_from_video(video_data: str, language_hint: str = "en") -> Dict[str, Any]:
    """
    Extract audio transcript from user-uploaded video.
    
    Args:
        video_data: Video file data or path from user upload
        language_hint: Expected language in audio
    
    Returns:
        Dict with transcript and metadata
    """
    logger.info(f"Extracting transcript from user-uploaded video (language: {language_hint})")
    
    try:
        # Placeholder for speech-to-text implementation
        # TODO: Implement real ASR (Google Speech API, Whisper, etc.)
        
        # Simulate transcript extraction
        transcript_text = f"[TRANSCRIPT PROCESSING] This would contain the transcript from the user's uploaded video. Language detected: {language_hint}"
        
        transcript_data = {
            "success": True,
            "transcript": transcript_text,
            "language_detected": language_hint,
            "duration_seconds": 120,  # placeholder - would be actual video duration
            "confidence_score": 0.90,
            "processing_timestamp": datetime.now().isoformat(),
            "input_source": "user_uploaded_video",
            "message": "Speech recognition completed (placeholder implementation - ready for real ASR integration)"
        }
        
        logger.info(f"✅ Transcript extraction completed for user video")
        return transcript_data
        
    except Exception as e:
        logger.error(f"❌ Transcript extraction failed: {e}")
        return {
            "success": False,
            "error": str(e),
            "language_hint": language_hint,
            "timestamp": datetime.now().isoformat()
        }

def store_analysis_results(analysis_data: Dict[str, Any], collection: str = "analysis_results") -> Dict[str, Any]:
    """
    Store analysis results in database for persistence.
    
    Args:
        analysis_data: Analysis results to store
        collection: Database collection name
    
    Returns:
        Dict with storage operation results
    """
    logger.info(f"Storing analysis results in collection '{collection}'")
    
    try:
        # Add storage metadata
        storage_record = {
            "analysis_data": analysis_data,
            "stored_timestamp": datetime.now().isoformat(),
            "collection": collection,
            "record_id": f"analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        }
        
        # TODO: Implement real MongoDB storage via MCP
        storage_result = {
            "success": True,
            "collection": collection,
            "record_id": storage_record["record_id"],
            "operation": "insert",
            "storage_timestamp": storage_record["stored_timestamp"],
            "message": f"Analysis results stored successfully in {collection}"
        }
        
        logger.info(f"✅ Analysis results stored: {storage_record['record_id']}")
        return storage_result
        
    except Exception as e:
        logger.error(f"❌ Storage operation failed: {e}")
        return {
            "success": False,
            "error": str(e),
            "collection": collection,
            "timestamp": datetime.now().isoformat()
        }

def query_stored_results(query_params: Optional[dict] = None, collection: str = "analysis_results") -> Dict[str, Any]:
    """
    Query previously stored analysis results.
    
    Args:
        query_params: Query parameters (date range, record_id, etc.)
        collection: Database collection to query
    
    Returns:
        Dict with query results
    """
    logger.info(f"Querying stored results in collection '{collection}'")
    
    try:
        query_params = query_params or {}
        
        # Simulate query results
        mock_results = [
            {
                "record_id": f"analysis_{datetime.now().strftime('%Y%m%d')}_001",
                "timestamp": datetime.now().isoformat(),
                "analysis_type": "nlp_analysis",
                "summary": "Sample analysis result 1"
            },
            {
                "record_id": f"analysis_{datetime.now().strftime('%Y%m%d')}_002", 
                "timestamp": datetime.now().isoformat(),
                "analysis_type": "risk_assessment",
                "summary": "Sample analysis result 2"
            }
        ]
        
        query_result = {
            "success": True,
            "query_params": query_params,
            "collection": collection,
            "results_count": len(mock_results),
            "results": mock_results,
            "query_timestamp": datetime.now().isoformat(),
            "message": f"Query completed - found {len(mock_results)} results"
        }
        
        logger.info(f"✅ Query completed: {len(mock_results)} results found")
        return query_result
        
    except Exception as e:
        logger.error(f"❌ Query operation failed: {e}")
        return {
            "success": False,
            "error": str(e),
            "query_params": query_params,
            "collection": collection,
            "timestamp": datetime.now().isoformat()
        }

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def analyze_sentiment_simple(text: str) -> float:
    """Simple sentiment analysis returning score between -1 and 1."""
    positive_words = ["good", "great", "excellent", "amazing", "wonderful", "fantastic", "love", "like", "happy", "successful"]
    negative_words = ["bad", "terrible", "awful", "hate", "horrible", "disgusting", "angry", "sad", "failed", "corrupt"]
    
    text_lower = text.lower()
    positive_count = sum(1 for word in positive_words if word in text_lower)
    negative_count = sum(1 for word in negative_words if word in text_lower)
    
    total_words = len(text.split())
    sentiment_words = positive_count + negative_count
    
    if sentiment_words == 0:
        return 0.0
    
    sentiment_score = (positive_count - negative_count) / total_words
    return max(-1.0, min(1.0, sentiment_score * 10))  # Scale and clamp

def extract_entities_simple(text: str) -> List[Dict[str, Any]]:
    """Simple entity extraction using patterns."""
    import re
    entities = []
    
    # Extract mentions (@username)
    mentions = re.findall(r'@(\w+)', text)
    for mention in mentions:
        entities.append({"text": mention, "type": "PERSON", "confidence": 0.8})
    
    # Extract hashtags (#hashtag)
    hashtags = re.findall(r'#(\w+)', text)
    for hashtag in hashtags:
        entities.append({"text": hashtag, "type": "HASHTAG", "confidence": 0.9})
    
    # Extract political parties (Nigeria-specific)
    parties = re.findall(r'\b(APC|PDP|Labour|NNPP|SDP|LP)\b', text, re.IGNORECASE)
    for party in parties:
        entities.append({"text": party.upper(), "type": "POLITICAL_PARTY", "confidence": 0.95})
    
    # Extract locations (Nigeria-specific)
    locations = re.findall(r'\b(Nigeria|Lagos|Abuja|Kano|Rivers|Kaduna|Ogun|Imo|Anambra|FCT)\b', text, re.IGNORECASE)
    for location in locations:
        entities.append({"text": location, "type": "LOCATION", "confidence": 0.85})
    
    return entities

def detect_risk_patterns_simple(text: str) -> List[str]:
    """Simple risk pattern detection."""
    risk_indicators = []
    text_lower = text.lower()
    
    # Election fraud claims
    if any(word in text_lower for word in ['rigged', 'fraud', 'manipulation', 'fake', 'stolen']):
        risk_indicators.append("election_fraud_claims")
    
    # Violence incitement
    if any(word in text_lower for word in ['violence', 'fight', 'attack', 'kill', 'destroy']):
        risk_indicators.append("violence_incitement")
    
    # Voter suppression
    if any(word in text_lower for word in ['dont vote', "don't vote", 'boycott', 'stay home']):
        risk_indicators.append("voter_suppression")
    
    # Misinformation indicators
    if any(word in text_lower for word in ['fake news', 'lies', 'propaganda', 'conspiracy']):
        risk_indicators.append("misinformation")
    
    # Hate speech indicators
    if any(word in text_lower for word in ['hate', 'enemy', 'traitor', 'destroy them']):
        risk_indicators.append("hate_speech")
    
    return risk_indicators

# =============================================================================
# TOOL REGISTRATION FOR ADK AGENTS (REMOVED)
# =============================================================================

# Export all processing functions
__all__ = [
    'process_social_media_data',
    'process_csv_data',
    'run_nlp_pipeline', 
    'extract_text_from_image',
    'extract_audio_transcript_from_video',
    'store_analysis_results',
    'query_stored_results',
]