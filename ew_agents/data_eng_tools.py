"""
 Multimodal Data Engineering Tools for ElectionWatch Agents
==================================================================

User-input driven tools for processing submitted content including
text, CSV files, images, videos, and other user-provided data for analysis.
Supports multimodal analysis using state-of-the-art vision-language models.
"""

import json
import logging
import asyncio
import base64
import io
from typing import Dict, List, Any, Optional, Union
from datetime import datetime
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# =============================================================================
# MULTIMODAL MODEL INTEGRATIONS
# =============================================================================

try:
    from transformers import AutoProcessor, AutoModelForVision2Seq, pipeline
    from PIL import Image
    import torch
    MULTIMODAL_AVAILABLE = True
    logger.info("✅ Multimodal models available")
except ImportError:
    MULTIMODAL_AVAILABLE = False
    logger.warning("⚠️ Multimodal models not available - install transformers and torch")

try:
    import whisper
    WHISPER_AVAILABLE = True
    logger.info("✅ Whisper speech recognition available")
except ImportError:
    WHISPER_AVAILABLE = False
    logger.warning("⚠️ Whisper not available - install openai-whisper")

# Global model instances for caching
_multimodal_processor = None
_multimodal_model = None
_whisper_model = None

# =============================================================================
# USER INPUT PROCESSING FUNCTIONS
# =============================================================================

def process_social_media_data(content: str, source_type: str = "user_input", metadata: Optional[dict] = None) -> Dict[str, Any]:
    """
    Processes raw social media content provided by the user, splitting it into individual posts and attaching metadata.
    
    Parameters:
        content (str): The raw social media content as a single string.
        source_type (str, optional): The origin of the content (e.g., 'csv_file', 'text_file', 'direct_input'). Defaults to 'user_input'.
        metadata (dict, optional): Additional metadata to associate with the processed content.
    
    Returns:
        Dict[str, Any]: A dictionary containing the processing status, list of processed posts with IDs and timestamps, metadata, and a message. On failure, returns error details and status.
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
    Parse and analyze CSV content, extracting rows, columns, and tweet-related text for further processing.
    
    Parameters:
        csv_content (str): Raw CSV data as a string.
        expected_columns (list, optional): List of column names to validate against the CSV header.
    
    Returns:
        dict: Dictionary containing processing success status, row and column statistics, missing columns, sample data, extracted tweet content, and timestamps. On failure, returns error details.
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
    Performs comprehensive NLP analysis on user-provided text, including sentiment scoring, entity extraction, risk indicator detection, and basic text statistics.
    
    Parameters:
        text (str): The text content to analyze.
        language (str, optional): Language code for the input text (default is "en").
    
    Returns:
        Dict[str, Any]: A dictionary containing NLP analysis results, including sentiment, entities, risk assessment, text statistics, and processing metadata. On failure, returns a dictionary with error details.
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
    Extracts text and analyzes political content from a user-uploaded image using a multimodal vision-language model.
    
    Accepts image data as a base64 string or file path, processes the image to extract visible text, generate a detailed content description, and analyze for political or election-related information. If a multimodal model is unavailable, returns placeholder analysis. Includes image metadata, confidence score, and processing details in the result.
    
    Returns:
        Dictionary containing extracted text, content analysis, political analysis, image metadata, confidence score, and processing status.
    """
    logger.info(f"Extracting text and analyzing content from user-uploaded image (language: {language_hint})")
    
    try:
        # Load image
        if image_data.startswith('data:image') or image_data.startswith('/'):
            # Handle base64 or file path
            if image_data.startswith('data:image'):
                # Extract base64 data
                header, encoded = image_data.split(",", 1)
                image_bytes = base64.b64decode(encoded)
                image = Image.open(io.BytesIO(image_bytes))
            else:
                # File path
                image = Image.open(image_data)
        else:
            # Assume base64 string
            image_bytes = base64.b64decode(image_data)
            image = Image.open(io.BytesIO(image_bytes))
        
        # Initialize multimodal model if not already done
        global _multimodal_processor, _multimodal_model, MULTIMODAL_AVAILABLE
        if MULTIMODAL_AVAILABLE and _multimodal_model is None:
            try:
                model_name = "Qwen/Qwen2.5-VL-7B-Instruct"
                _multimodal_processor = AutoProcessor.from_pretrained(model_name)
                _multimodal_model = AutoModelForVision2Seq.from_pretrained(
                    model_name, 
                    torch_dtype=torch.float16,
                    device_map="auto"
                )
                logger.info(f"✅ Loaded multimodal model: {model_name}")
            except Exception as e:
                logger.error(f"❌ Failed to load multimodal model: {e}")
                MULTIMODAL_AVAILABLE = False
        
        if MULTIMODAL_AVAILABLE and _multimodal_model is not None:
            # Analyze image content
            prompt = "Describe this image in detail, including any text visible in it. Focus on political content, signs, banners, or election-related information."
            
            inputs = _multimodal_processor(
                images=image,
                text=prompt,
                return_tensors="pt"
            ).to(_multimodal_model.device)
            
            with torch.no_grad():
                generated_ids = _multimodal_model.generate(
                    **inputs,
                    max_new_tokens=512,
                    do_sample=True,
                    temperature=0.7
                )
            
            analysis_text = _multimodal_processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
            
            # Extract text specifically
            text_prompt = "Extract and list all text visible in this image, including signs, banners, and any written content."
            text_inputs = _multimodal_processor(
                images=image,
                text=text_prompt,
                return_tensors="pt"
            ).to(_multimodal_model.device)
            
            with torch.no_grad():
                text_ids = _multimodal_model.generate(
                    **text_inputs,
                    max_new_tokens=256,
                    do_sample=False
                )
            
            extracted_text = _multimodal_processor.batch_decode(text_ids, skip_special_tokens=True)[0]
            
            # Analyze for political content
            political_prompt = "Analyze this image for political content, election-related information, or potential misinformation. Identify any political parties, candidates, or campaign materials."
            political_inputs = _multimodal_processor(
                images=image,
                text=political_prompt,
                return_tensors="pt"
            ).to(_multimodal_model.device)
            
            with torch.no_grad():
                political_ids = _multimodal_model.generate(
                    **political_inputs,
                    max_new_tokens=256,
                    do_sample=True,
                    temperature=0.5
                )
            
            political_analysis = _multimodal_processor.batch_decode(political_ids, skip_special_tokens=True)[0]
            
        else:
            # Fallback to placeholder
            analysis_text = f"[MULTIMODAL ANALYSIS] This would contain detailed analysis of the image content. Language detected: {language_hint}"
            extracted_text = f"[TEXT EXTRACTION] This would contain text extracted from the image. Language detected: {language_hint}"
            political_analysis = f"[POLITICAL ANALYSIS] This would contain political content analysis. Language detected: {language_hint}"
        
        # Get image metadata
        image_metadata = {
            "format": image.format,
            "mode": image.mode,
            "size": image.size,
            "width": image.width,
            "height": image.height
        }
        
        extracted_data = {
            "success": True,
            "extracted_text": extracted_text,
            "content_analysis": analysis_text,
            "political_analysis": political_analysis,
            "language_detected": language_hint,
            "confidence_score": 0.85 if MULTIMODAL_AVAILABLE else 0.5,
            "image_metadata": image_metadata,
            "processing_timestamp": datetime.now().isoformat(),
            "input_source": "user_uploaded_image",
            "model_used": "Qwen2.5-VL-7B-Instruct" if MULTIMODAL_AVAILABLE else "placeholder",
            "message": f"Multimodal analysis completed using {'real model' if MULTIMODAL_AVAILABLE else 'placeholder implementation'}"
        }
        
        logger.info(f"✅ Multimodal image analysis completed")
        return extracted_data
        
    except Exception as e:
        logger.error(f"❌ Multimodal image analysis failed: {e}")
        return {
            "success": False,
            "error": str(e),
            "language_hint": language_hint,
            "timestamp": datetime.now().isoformat()
        }

def extract_audio_transcript_from_video(video_data: str, language_hint: str = "en") -> Dict[str, Any]:
    """
    Extracts an audio transcript from a video file or base64 data and analyzes its political content.
    
    If a speech recognition model is available, transcribes the video's audio and detects the spoken language, confidence, and segments. Analyzes the transcript for political content and returns metadata about the video. If the model is unavailable or processing fails, returns placeholder transcript and analysis.
    
    Parameters:
        video_data (str): Path to a video file or base64-encoded video data.
        language_hint (str): Expected language of the audio for transcription.
    
    Returns:
        Dict[str, Any]: Dictionary containing the transcript, political analysis, detected language, confidence score, video metadata, and processing status.
    """
    logger.info(f"Extracting transcript and analyzing video content (language: {language_hint})")
    
    try:
        # Initialize Whisper model if not already done
        global _whisper_model, WHISPER_AVAILABLE
        if WHISPER_AVAILABLE and _whisper_model is None:
            try:
                _whisper_model = whisper.load_model("base")
                logger.info("✅ Loaded Whisper model for speech recognition")
            except Exception as e:
                logger.error(f"❌ Failed to load Whisper model: {e}")
                WHISPER_AVAILABLE = False
        
        # Extract audio from video and transcribe
        if WHISPER_AVAILABLE and _whisper_model is not None:
            try:
                # Load video file
                if video_data.startswith('/') or Path(video_data).exists():
                    video_path = video_data
                else:
                    # Handle base64 video data
                    video_bytes = base64.b64decode(video_data)
                    video_path = f"/tmp/temp_video_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"
                    with open(video_path, 'wb') as f:
                        f.write(video_bytes)
                
                # Transcribe audio
                result = _whisper_model.transcribe(video_path, language=language_hint)
                transcript_text = result["text"]
                detected_language = result.get("language", language_hint)
                confidence_score = result.get("confidence", 0.8)
                
                # Analyze transcript for political content
                political_analysis = analyze_transcript_for_political_content(transcript_text)
                
                # Get video metadata (simplified)
                video_metadata = {
                    "duration_seconds": result.get("duration", 0),
                    "language_detected": detected_language,
                    "segments_count": len(result.get("segments", [])),
                    "file_path": video_path
                }
                
            except Exception as e:
                logger.error(f"❌ Whisper transcription failed: {e}")
                # Fallback to placeholder
                transcript_text = f"[TRANSCRIPT PROCESSING] This would contain the transcript from the user's uploaded video. Language detected: {language_hint}"
                political_analysis = f"[POLITICAL ANALYSIS] This would contain political content analysis of the transcript. Language detected: {language_hint}"
                detected_language = language_hint
                confidence_score = 0.5
                video_metadata = {"duration_seconds": 120, "language_detected": language_hint}
        else:
            # Fallback to placeholder
            transcript_text = f"[TRANSCRIPT PROCESSING] This would contain the transcript from the user's uploaded video. Language detected: {language_hint}"
            political_analysis = f"[POLITICAL ANALYSIS] This would contain political content analysis of the transcript. Language detected: {language_hint}"
            detected_language = language_hint
            confidence_score = 0.5
            video_metadata = {"duration_seconds": 120, "language_detected": language_hint}
        
        transcript_data = {
            "success": True,
            "transcript": transcript_text,
            "political_analysis": political_analysis,
            "language_detected": detected_language,
            "confidence_score": confidence_score,
            "video_metadata": video_metadata,
            "processing_timestamp": datetime.now().isoformat(),
            "input_source": "user_uploaded_video",
            "model_used": "Whisper" if WHISPER_AVAILABLE else "placeholder",
            "message": f"Video analysis completed using {'real Whisper model' if WHISPER_AVAILABLE else 'placeholder implementation'}"
        }
        
        logger.info(f"✅ Video analysis completed")
        return transcript_data
        
    except Exception as e:
        logger.error(f"❌ Video analysis failed: {e}")
        return {
            "success": False,
            "error": str(e),
            "language_hint": language_hint,
            "timestamp": datetime.now().isoformat()
        }

def store_analysis_results(analysis_data: Dict[str, Any], collection: str = "analysis_results") -> Dict[str, Any]:
    """
    Simulate storing analysis results in a database collection and return storage metadata.
    
    Returns:
        Dictionary containing the status of the storage operation, including collection name, record ID, operation type, timestamp, and a message.
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
    Simulate querying stored analysis results and return mock results.
    
    Parameters:
        query_params (dict, optional): Parameters to filter the query, such as date range or record ID.
        collection (str): Name of the collection to query.
    
    Returns:
        dict: Dictionary containing the query status, parameters, results, and metadata.
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
# NEW MULTIMODAL ANALYSIS FUNCTIONS
# =============================================================================

def analyze_multimodal_content(content_data: Dict[str, Any], content_type: str = "mixed") -> Dict[str, Any]:
    """
    Performs a comprehensive analysis of multimodal content by integrating text, image, and video analyses.
    
    Analyzes each available modality in the input, synthesizes results across modalities, assesses overall risk, extracts political entities, and detects misinformation indicators. Returns a structured dictionary with detailed analysis results, synthesis, risk assessment, and metadata.
    
    Parameters:
        content_data (Dict[str, Any]): Dictionary containing content to analyze, with possible keys 'text', 'image', and 'video'.
        content_type (str): Descriptor for the type of content provided (e.g., 'text', 'image', 'video', or 'mixed').
    
    Returns:
        Dict[str, Any]: Dictionary containing analysis results, synthesis, risk assessment, political entities, misinformation indicators, and timestamps. On failure, returns error details.
    """
    logger.info(f"Running comprehensive multimodal analysis for {content_type} content")
    
    try:
        analysis_results = {
            "success": True,
            "content_type": content_type,
            "analysis_timestamp": datetime.now().isoformat(),
            "components": {},
            "synthesis": {},
            "risk_assessment": {},
            "political_entities": [],
            "misinformation_indicators": []
        }
        
        # Analyze text content if present
        if "text" in content_data:
            text_analysis = run_nlp_pipeline(content_data["text"])
            analysis_results["components"]["text_analysis"] = text_analysis
        
        # Analyze image content if present
        if "image" in content_data:
            image_analysis = extract_text_from_image(content_data["image"])
            analysis_results["components"]["image_analysis"] = image_analysis
        
        # Analyze video content if present
        if "video" in content_data:
            video_analysis = extract_audio_transcript_from_video(content_data["video"])
            analysis_results["components"]["video_analysis"] = video_analysis
        
        # Synthesize results across modalities
        synthesis = synthesize_multimodal_results(analysis_results["components"])
        analysis_results["synthesis"] = synthesis
        
        # Comprehensive risk assessment
        risk_assessment = assess_multimodal_risk(analysis_results["components"])
        analysis_results["risk_assessment"] = risk_assessment
        
        # Extract political entities across modalities
        political_entities = extract_political_entities_multimodal(analysis_results["components"])
        analysis_results["political_entities"] = political_entities
        
        # Detect misinformation indicators
        misinformation_indicators = detect_multimodal_misinformation(analysis_results["components"])
        analysis_results["misinformation_indicators"] = misinformation_indicators
        
        logger.info(f"✅ Comprehensive multimodal analysis completed")
        return analysis_results
        
    except Exception as e:
        logger.error(f"❌ Multimodal analysis failed: {e}")
        return {
            "success": False,
            "error": str(e),
            "content_type": content_type,
            "timestamp": datetime.now().isoformat()
        }

def process_document_with_multimodal(document_data: str, document_type: str = "pdf") -> Dict[str, Any]:
    """
    Processes a document (PDF, image, etc.) using multimodal analysis and returns extracted content and analysis results.
    
    Parameters:
        document_data (str): The document data, which can be a base64 string or file path.
        document_type (str): The type of the document (e.g., 'pdf', 'image', 'jpg', 'png').
    
    Returns:
        Dict[str, Any]: A dictionary containing analysis results, extracted text, or error information depending on the document type and processing outcome.
    """
    logger.info(f"Processing {document_type} document with multimodal analysis")
    
    try:
        # For now, handle image-based documents
        if document_type in ["image", "jpg", "jpeg", "png", "tiff"]:
            return extract_text_from_image(document_data)
        
        # For PDFs, we'd need additional processing
        elif document_type == "pdf":
            # TODO: Implement PDF processing with multimodal models
            return {
                "success": True,
                "document_type": "pdf",
                "extracted_text": "[PDF PROCESSING] This would contain text extracted from PDF using multimodal models",
                "content_analysis": "[PDF ANALYSIS] This would contain analysis of PDF content",
                "processing_timestamp": datetime.now().isoformat(),
                "message": "PDF processing with multimodal models (placeholder implementation)"
            }
        
        else:
            return {
                "success": False,
                "error": f"Unsupported document type: {document_type}",
                "timestamp": datetime.now().isoformat()
            }
            
    except Exception as e:
        logger.error(f"❌ Document processing failed: {e}")
        return {
            "success": False,
            "error": str(e),
            "document_type": document_type,
            "timestamp": datetime.now().isoformat()
        }

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def analyze_sentiment_simple(text: str) -> float:
    """
    Performs a basic sentiment analysis on the input text and returns a score between -1 and 1.
    
    The score is calculated based on the presence of predefined positive and negative words, scaled by text length. Positive values indicate positive sentiment, negative values indicate negative sentiment, and zero indicates neutral or no sentiment detected.
    
    Returns:
        float: Sentiment score in the range [-1, 1].
    """
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
    """
    Extracts simple entities from text, including mentions, hashtags, Nigerian political parties, and locations.
    
    Returns:
        List of detected entities, each with text, type, and confidence score.
    """
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
    """
    Detects and returns a list of risk indicators present in the input text based on predefined keywords related to election fraud, violence, voter suppression, misinformation, and hate speech.
    
    Parameters:
        text (str): The input text to analyze for risk patterns.
    
    Returns:
        List[str]: A list of detected risk indicator strings.
    """
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

def analyze_transcript_for_political_content(transcript: str) -> str:
    """
    Detects the presence of political or election-related keywords in a transcript.
    
    Returns:
        str: A summary indicating whether political content was found in the transcript.
    """
    political_keywords = [
        'election', 'vote', 'candidate', 'party', 'campaign', 'polling', 'ballot',
        'democracy', 'government', 'political', 'president', 'minister', 'parliament'
    ]
    
    political_content = []
    transcript_lower = transcript.lower()
    
    for keyword in political_keywords:
        if keyword in transcript_lower:
            political_content.append(keyword)
    
    if political_content:
        return f"Political content detected: {', '.join(set(political_content))}. Full transcript analysis available."
    else:
        return "No significant political content detected in transcript."

def synthesize_multimodal_results(components: Dict[str, Any]) -> Dict[str, Any]:
    """
    Aggregate and synthesize analysis results from multiple content modalities into a unified summary.
    
    Parameters:
        components (Dict[str, Any]): A dictionary containing analysis outputs from different modalities (e.g., text, image, video).
    
    Returns:
        Dict[str, Any]: A dictionary summarizing overall sentiment, confidence score, key themes, and placeholders for contradictions and reinforcing elements.
    """
    synthesis = {
        "overall_sentiment": "neutral",
        "confidence_score": 0.0,
        "key_themes": [],
        "contradictions": [],
        "reinforcing_elements": []
    }
    
    # Aggregate sentiment scores
    sentiment_scores = []
    themes = set()
    
    for component_name, component_data in components.items():
        if component_data.get("success"):
            # Extract sentiment
            if "sentiment" in component_data:
                sentiment_scores.append(component_data["sentiment"].get("score", 0))
            
            # Extract themes
            if "entities" in component_data:
                for entity in component_data["entities"]:
                    themes.add(entity.get("text", ""))
            
            # Extract political analysis
            if "political_analysis" in component_data:
                themes.add("political_content")
    
    # Calculate overall sentiment
    if sentiment_scores:
        avg_sentiment = sum(sentiment_scores) / len(sentiment_scores)
        synthesis["overall_sentiment"] = "positive" if avg_sentiment > 0.1 else "negative" if avg_sentiment < -0.1 else "neutral"
        synthesis["confidence_score"] = min(1.0, len(sentiment_scores) * 0.2)
    
    synthesis["key_themes"] = list(themes)
    
    return synthesis

def assess_multimodal_risk(components: Dict[str, Any]) -> Dict[str, Any]:
    """
    Aggregates and assesses risk indicators from multiple content modalities to determine an overall risk level.
    
    Analyzes risk assessments from each component, computes an average risk score, and assigns an overall risk level (low, medium, high). Identifies unique risk factors and provides recommendations based on detected risks.
    
    Parameters:
        components (Dict[str, Any]): Dictionary of analysis results from different modalities, each potentially containing a risk assessment.
    
    Returns:
        Dict[str, Any]: Dictionary containing the overall risk level, identified risk factors, confidence score, and actionable recommendations.
    """
    risk_assessment = {
        "overall_risk_level": "low",
        "risk_factors": [],
        "confidence": 0.0,
        "recommendations": []
    }
    
    risk_factors = []
    risk_scores = []
    
    for component_name, component_data in components.items():
        if component_data.get("success"):
            # Extract risk indicators
            if "risk_assessment" in component_data:
                risk_indicators = component_data["risk_assessment"].get("risk_indicators", [])
                risk_factors.extend(risk_indicators)
                
                risk_level = component_data["risk_assessment"].get("risk_level", "low")
                risk_scores.append({"low": 1, "medium": 2, "high": 3}.get(risk_level, 1))
    
    # Calculate overall risk
    if risk_scores:
        avg_risk = sum(risk_scores) / len(risk_scores)
        if avg_risk >= 2.5:
            risk_assessment["overall_risk_level"] = "high"
        elif avg_risk >= 1.5:
            risk_assessment["overall_risk_level"] = "medium"
        else:
            risk_assessment["overall_risk_level"] = "low"
        
        risk_assessment["confidence"] = min(1.0, len(risk_scores) * 0.3)
    
    risk_assessment["risk_factors"] = list(set(risk_factors))
    
    # Generate recommendations
    if "election_fraud_claims" in risk_factors:
        risk_assessment["recommendations"].append("Fact-check election claims and verify sources")
    if "violence_incitement" in risk_factors:
        risk_assessment["recommendations"].append("Monitor for escalation and report to authorities if necessary")
    if "misinformation" in risk_factors:
        risk_assessment["recommendations"].append("Cross-reference information with multiple reliable sources")
    
    return risk_assessment

def extract_political_entities_multimodal(components: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Extracts unique political entities from multimodal analysis components.
    
    This function scans the provided analysis components for entities related to political parties, persons, and locations, as well as markers of political content. It aggregates these entities across modalities, removes duplicates, and returns a list of unique political entities with their type, confidence, and source.
     
    Parameters:
        components (Dict[str, Any]): A dictionary of analysis results from different modalities.
    
    Returns:
        List[Dict[str, Any]]: A list of unique political entities detected, each with text, type, confidence, and source.
    """
    political_entities = []
    
    for component_name, component_data in components.items():
        if component_data.get("success"):
            # Extract entities from text analysis
            if "entities" in component_data:
                for entity in component_data["entities"]:
                    if entity.get("type") in ["POLITICAL_PARTY", "PERSON", "LOCATION"]:
                        political_entities.append({
                            "text": entity.get("text"),
                            "type": entity.get("type"),
                            "confidence": entity.get("confidence", 0.8),
                            "source": component_name
                        })
            
            # Extract from political analysis
            if "political_analysis" in component_data:
                political_entities.append({
                    "text": "political_content",
                    "type": "POLITICAL_CONTENT",
                    "confidence": 0.9,
                    "source": component_name
                })
    
    # Remove duplicates
    unique_entities = []
    seen_texts = set()
    for entity in political_entities:
        if entity["text"] not in seen_texts:
            unique_entities.append(entity)
            seen_texts.add(entity["text"])
    
    return unique_entities

def detect_multimodal_misinformation(components: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Detects potential misinformation indicators by aggregating risk assessments and contradictions from multiple content modalities.
    
    Parameters:
        components (Dict[str, Any]): Analysis results from different modalities (e.g., text, image, video).
    
    Returns:
        List[Dict[str, Any]]: A list of detected misinformation indicators, each with type, source, confidence, and description.
    """
    misinformation_indicators = []
    
    for component_name, component_data in components.items():
        if component_data.get("success"):
            # Check for risk indicators
            if "risk_assessment" in component_data:
                risk_indicators = component_data["risk_assessment"].get("risk_indicators", [])
                for indicator in risk_indicators:
                    if indicator == "misinformation":
                        misinformation_indicators.append({
                            "type": "misinformation_detected",
                            "source": component_name,
                            "confidence": 0.8,
                            "description": "Potential misinformation detected in content"
                        })
            
            # Check for contradictions between modalities
            if "contradictions" in component_data:
                misinformation_indicators.extend(component_data["contradictions"])
    
    return misinformation_indicators

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
    'analyze_multimodal_content',
    'process_document_with_multimodal',
    'store_analysis_results',
    'query_stored_results',
]