import os
import json
import requests
import tempfile
from typing import Dict, List, Any, Optional, Union
from datetime import datetime, timezone
import uuid
from pymongo import MongoClient
from google.cloud import storage, firestore, vision
from google.cloud.storage import Blob
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CloudDatabaseService:
    """
    Unified cloud database service for Election Watch ML system
    Integrates MongoDB Atlas, Google Cloud Storage, and Firestore
    """
    
    def __init__(self):
        # MongoDB Atlas connection
        self.mongo_uri = os.getenv('MONGODB_ATLAS_URI', 'mongodb+srv://ew_ml:moHsc5i6gYFrLsvL@ewcluster1.fpkzpxg.mongodb.net/')
        self.mongo_client = MongoClient(self.mongo_uri, tlsAllowInvalidCertificates=True)
        self.mongo_db = self.mongo_client["election_watch"]
        
        # Google Cloud Storage
        self.gcs_bucket_name = os.getenv('GCS_BUCKET_NAME', 'election-watch-data')
        self.gcs_client = storage.Client()
        self.gcs_bucket = self.gcs_client.bucket(self.gcs_bucket_name)
        
        # Firestore
        self.firestore_client = firestore.Client()
        
        logger.info("CloudDatabaseService initialized successfully")
    
    # ==================== ACTOR METHODS ====================
    
    def create_actor(self, actor_data: Dict[str, Any]) -> str:
        """Create a new actor in the database"""
        try:
            actor_data['uuid'] = f"actor-uuid-{str(uuid.uuid4())[:8]}"
            actor_data['created_at'] = datetime.now(timezone.utc)
            actor_data['updated_at'] = datetime.now(timezone.utc)
            
            result = self.mongo_db.actors.insert_one(actor_data)
            logger.info(f"Created actor: {actor_data['uuid']}")
            return actor_data['uuid']
        except Exception as e:
            logger.error(f"Error creating actor: {e}")
            raise
    
    def get_actor(self, actor_id: str) -> Optional[Dict[str, Any]]:
        """Get actor by UUID"""
        try:
            actor = self.mongo_db.actors.find_one({"uuid": actor_id})
            return actor
        except Exception as e:
            logger.error(f"Error getting actor {actor_id}: {e}")
            return None
    
    def search_actors(self, filters: Dict[str, Any], limit: int = 50) -> List[Dict[str, Any]]:
        """Search actors with filters"""
        try:
            cursor = self.mongo_db.actors.find(filters).limit(limit)
            return list(cursor)
        except Exception as e:
            logger.error(f"Error searching actors: {e}")
            return []
    
    def update_actor(self, actor_id: str, update_data: Dict[str, Any]) -> bool:
        """Update actor information"""
        try:
            update_data['updated_at'] = datetime.now(timezone.utc)
            result = self.mongo_db.actors.update_one(
                {"uuid": actor_id}, 
                {"$set": update_data}
            )
            return result.modified_count > 0
        except Exception as e:
            logger.error(f"Error updating actor {actor_id}: {e}")
            return False
    
    # ==================== NARRATIVE METHODS ====================
    
    def create_narrative(self, narrative_data: Dict[str, Any]) -> str:
        """Create a new narrative in the database"""
        try:
            narrative_data['uuid'] = f"narrative-uuid-{str(uuid.uuid4())[:8]}"
            narrative_data['created_at'] = datetime.now(timezone.utc)
            narrative_data['updated_at'] = datetime.now(timezone.utc)
            
            result = self.mongo_db.narratives.insert_one(narrative_data)
            logger.info(f"Created narrative: {narrative_data['uuid']}")
            return narrative_data['uuid']
        except Exception as e:
            logger.error(f"Error creating narrative: {e}")
            raise
    
    def get_narrative(self, narrative_id: str) -> Optional[Dict[str, Any]]:
        """Get narrative by UUID"""
        try:
            narrative = self.mongo_db.narratives.find_one({"uuid": narrative_id})
            return narrative
        except Exception as e:
            logger.error(f"Error getting narrative {narrative_id}: {e}")
            return None
    
    def search_narratives(self, filters: Dict[str, Any], limit: int = 50) -> List[Dict[str, Any]]:
        """Search narratives with filters"""
        try:
            cursor = self.mongo_db.narratives.find(filters).limit(limit)
            return list(cursor)
        except Exception as e:
            logger.error(f"Error searching narratives: {e}")
            return []
    
    def get_narratives_by_category(self, category: str) -> List[Dict[str, Any]]:
        """Get narratives by category"""
        return self.search_narratives({"category": category})
    
    def get_narratives_by_disarm_technique(self, technique_id: str) -> List[Dict[str, Any]]:
        """Get narratives by DISARM technique"""
        return self.search_narratives({"disarm_technique_id": technique_id})
    
    # ==================== LEXICON METHODS ====================
    
    def create_lexicon_term(self, lexicon_data: Dict[str, Any]) -> str:
        """Create a new lexicon term"""
        try:
            lexicon_data['uuid'] = f"lexicon-uuid-{str(uuid.uuid4())[:8]}"
            lexicon_data['created_at'] = datetime.now(timezone.utc)
            lexicon_data['updated_at'] = datetime.now(timezone.utc)
            
            result = self.mongo_db.lexicons.insert_one(lexicon_data)
            logger.info(f"Created lexicon term: {lexicon_data['term']}")
            return lexicon_data['uuid']
        except Exception as e:
            logger.error(f"Error creating lexicon term: {e}")
            raise
    
    def get_lexicon_term(self, term: str, language_code: str = "en") -> Optional[Dict[str, Any]]:
        """Get lexicon term by term and language"""
        try:
            lexicon = self.mongo_db.lexicons.find_one({
                "term": term, 
                "language_code": language_code
            })
            return lexicon
        except Exception as e:
            logger.error(f"Error getting lexicon term {term}: {e}")
            return None
    
    def search_lexicon_terms(self, query: str, language_code: str = "en") -> List[Dict[str, Any]]:
        """Search lexicon terms by text query"""
        try:
            # Text search in term, definition, and tags
            cursor = self.mongo_db.lexicons.find({
                "$and": [
                    {"language_code": language_code},
                    {"$or": [
                        {"term": {"$regex": query, "$options": "i"}},
                        {"definition": {"$regex": query, "$options": "i"}},
                        {"tags": {"$regex": query, "$options": "i"}}
                    ]}
                ]
            })
            return list(cursor)
        except Exception as e:
            logger.error(f"Error searching lexicon terms: {e}")
            return []
    
    def increment_lexicon_usage(self, term: str, language_code: str = "en") -> bool:
        """Increment usage count for a lexicon term"""
        try:
            result = self.mongo_db.lexicons.update_one(
                {"term": term, "language_code": language_code},
                {"$inc": {"usage_count": 1}, "$set": {"updated_at": datetime.now(timezone.utc)}}
            )
            return result.modified_count > 0
        except Exception as e:
            logger.error(f"Error incrementing usage for {term}: {e}")
            return False
    
    # ==================== CONTENT METHODS ====================
    
    def create_content_item(self, content_data: Dict[str, Any]) -> str:
        """Create a new content item"""
        try:
            content_data['uuid'] = f"content-uuid-{str(uuid.uuid4())[:8]}"
            content_data['created_at'] = datetime.now(timezone.utc)
            content_data['updated_at'] = datetime.now(timezone.utc)
            
            result = self.mongo_db.content_items.insert_one(content_data)
            logger.info(f"Created content item: {content_data['uuid']}")
            return content_data['uuid']
        except Exception as e:
            logger.error(f"Error creating content item: {e}")
            raise
    
    def get_content_item(self, content_id: str) -> Optional[Dict[str, Any]]:
        """Get content item by UUID"""
        try:
            content = self.mongo_db.content_items.find_one({"uuid": content_id})
            return content
        except Exception as e:
            logger.error(f"Error getting content item {content_id}: {e}")
            return None
    
    def search_content_items(self, filters: Dict[str, Any], limit: int = 50) -> List[Dict[str, Any]]:
        """Search content items with filters"""
        try:
            cursor = self.mongo_db.content_items.find(filters).limit(limit)
            return list(cursor)
        except Exception as e:
            logger.error(f"Error searching content items: {e}")
            return []
    
    def get_content_by_actor(self, actor_id: str) -> List[Dict[str, Any]]:
        """Get content items by actor"""
        return self.search_content_items({"author_actor_id": actor_id})
    
    def get_content_by_platform(self, platform: str) -> List[Dict[str, Any]]:
        """Get content items by platform"""
        return self.search_content_items({"source_platform": platform})
    
    # ==================== RELATIONSHIP METHODS ====================
    
    def create_relationship(self, relationship_data: Dict[str, Any]) -> str:
        """Create a new relationship"""
        try:
            relationship_data['uuid'] = f"relationship-uuid-{str(uuid.uuid4())[:8]}"
            relationship_data['created_at'] = datetime.now(timezone.utc)
            relationship_data['updated_at'] = datetime.now(timezone.utc)
            
            result = self.mongo_db.relationships.insert_one(relationship_data)
            logger.info(f"Created relationship: {relationship_data['uuid']}")
            return relationship_data['uuid']
        except Exception as e:
            logger.error(f"Error creating relationship: {e}")
            raise
    
    def get_relationships(self, source_type: str, source_id: str) -> List[Dict[str, Any]]:
        """Get relationships for a source entity"""
        try:
            cursor = self.mongo_db.relationships.find({
                "source_type": source_type,
                "source_id": source_id
            })
            return list(cursor)
        except Exception as e:
            logger.error(f"Error getting relationships: {e}")
            return []
    
    def get_related_entities(self, entity_type: str, entity_id: str, relationship_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get entities related to a given entity"""
        try:
            query = {
                "$or": [
                    {"source_type": entity_type, "source_id": entity_id},
                    {"target_type": entity_type, "target_id": entity_id}
                ]
            }
            if relationship_type:
                query["relationship_type"] = relationship_type
            
            cursor = self.mongo_db.relationships.find(query)
            return list(cursor)
        except Exception as e:
            logger.error(f"Error getting related entities: {e}")
            return []
    
    # ==================== GOOGLE CLOUD STORAGE METHODS ====================
    
    def upload_file_to_gcs(self, file_path: str, blob_name: str) -> str:
        """Upload file to Google Cloud Storage"""
        try:
            blob = self.gcs_bucket.blob(blob_name)
            blob.upload_from_filename(file_path)
            logger.info(f"Uploaded {file_path} to {blob_name}")
            return f"gs://{self.gcs_bucket_name}/{blob_name}"
        except Exception as e:
            logger.error(f"Error uploading file to GCS: {e}")
            raise
    
    def upload_data_to_gcs(self, data: Union[str, bytes], blob_name: str) -> str:
        """Upload data directly to Google Cloud Storage"""
        try:
            blob = self.gcs_bucket.blob(blob_name)
            if isinstance(data, str):
                blob.upload_from_string(data)
            else:
                blob.upload_from_string(data)
            logger.info(f"Uploaded data to {blob_name}")
            return f"gs://{self.gcs_bucket_name}/{blob_name}"
        except Exception as e:
            logger.error(f"Error uploading data to GCS: {e}")
            raise
    
    def download_from_gcs(self, blob_name: str) -> bytes:
        """Download file from Google Cloud Storage"""
        try:
            blob = self.gcs_bucket.blob(blob_name)
            return blob.download_as_bytes()
        except Exception as e:
            logger.error(f"Error downloading from GCS: {e}")
            raise
    
    def list_gcs_files(self, prefix: str = "") -> List[str]:
        """List files in Google Cloud Storage"""
        try:
            blobs = self.gcs_bucket.list_blobs(prefix=prefix)
            return [blob.name for blob in blobs]
        except Exception as e:
            logger.error(f"Error listing GCS files: {e}")
            return []
    
    # ==================== FIRESTORE METHODS ====================
    
    def cache_analysis_result(self, analysis_id: str, result_data: Dict[str, Any]) -> bool:
        """Cache analysis result in Firestore"""
        try:
            doc_ref = self.firestore_client.collection('analysis_cache').document(analysis_id)
            result_data['cached_at'] = datetime.now(timezone.utc)
            doc_ref.set(result_data)
            logger.info(f"Cached analysis result: {analysis_id}")
            return True
        except Exception as e:
            logger.error(f"Error caching analysis result: {e}")
            return False
    
    def get_cached_analysis(self, analysis_id: str) -> Optional[Dict[str, Any]]:
        """Get cached analysis result from Firestore"""
        try:
            doc_ref = self.firestore_client.collection('analysis_cache').document(analysis_id)
            doc = doc_ref.get()
            if doc.exists:
                return doc.to_dict()
            return None
        except Exception as e:
            logger.error(f"Error getting cached analysis: {e}")
            return None
    
    def store_real_time_alert(self, alert_data: Dict[str, Any]) -> str:
        """Store real-time alert in Firestore"""
        try:
            alert_data['timestamp'] = datetime.now(timezone.utc)
            alert_data['alert_id'] = f"alert-{str(uuid.uuid4())[:8]}"
            
            doc_ref = self.firestore_client.collection('alerts').document(alert_data['alert_id'])
            doc_ref.set(alert_data)
            logger.info(f"Stored alert: {alert_data['alert_id']}")
            return alert_data['alert_id']
        except Exception as e:
            logger.error(f"Error storing alert: {e}")
            raise
    
    def get_recent_alerts(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent alerts from Firestore"""
        try:
            alerts_ref = self.firestore_client.collection('alerts')
            query = alerts_ref.order_by('timestamp', direction=firestore.Query.DESCENDING).limit(limit)
            docs = query.stream()
            return [doc.to_dict() for doc in docs]
        except Exception as e:
            logger.error(f"Error getting recent alerts: {e}")
            return []
    
    # ==================== ANALYTICS METHODS ====================
    
    def get_database_stats(self) -> Dict[str, Any]:
        """Get database statistics"""
        try:
            stats = {
                "mongodb": {
                    "actors": self.mongo_db.actors.count_documents({}),
                    "narratives": self.mongo_db.narratives.count_documents({}),
                    "lexicons": self.mongo_db.lexicons.count_documents({}),
                    "content_items": self.mongo_db.content_items.count_documents({}),
                    "relationships": self.mongo_db.relationships.count_documents({})
                },
                "firestore": {
                    "cached_analyses": len(list(self.firestore_client.collection('analysis_cache').stream())),
                    "alerts": len(list(self.firestore_client.collection('alerts').stream()))
                },
                "gcs": {
                    "total_files": len(self.list_gcs_files())
                }
            }
            return stats
        except Exception as e:
            logger.error(f"Error getting database stats: {e}")
            return {}
    
    def get_narrative_categories_stats(self) -> Dict[str, int]:
        """Get narrative statistics by category"""
        try:
            pipeline = [
                {"$group": {"_id": "$category", "count": {"$sum": 1}}},
                {"$sort": {"count": -1}}
            ]
            results = list(self.mongo_db.narratives.aggregate(pipeline))
            return {result["_id"]: result["count"] for result in results}
        except Exception as e:
            logger.error(f"Error getting narrative category stats: {e}")
            return {}
    
    def get_actor_influence_stats(self) -> Dict[str, Any]:
        """Get actor influence statistics"""
        try:
            pipeline = [
                {"$group": {
                    "_id": "$actor_type",
                    "count": {"$sum": 1},
                    "avg_influence": {"$avg": "$influence_score"},
                    "max_influence": {"$max": "$influence_score"}
                }},
                {"$sort": {"avg_influence": -1}}
            ]
            results = list(self.mongo_db.actors.aggregate(pipeline))
            return {result["_id"]: {
                "count": result["count"],
                "avg_influence": round(result["avg_influence"], 2),
                "max_influence": result["max_influence"]
            } for result in results}
        except Exception as e:
            logger.error(f"Error getting actor influence stats: {e}")
            return {}
    
    # ==================== MULTIMEDIA PROCESSING METHODS ====================
    
    def extract_text_from_image(self, image_url: str, language_hint: str = "en", store_in_gcs: bool = True) -> Dict[str, Any]:
        """
        Extract text from image using Google Cloud Vision API
        Supports both local files and URLs
        """
        try:
            # Initialize Vision API client
            vision_client = vision.ImageAnnotatorClient()
            
            # Handle different input types
            if image_url.startswith(('http://', 'https://')):
                # Download image from URL
                response = requests.get(image_url)
                response.raise_for_status()
                image_content = response.content
                
                # Optionally store in GCS for archival
                if store_in_gcs:
                    blob_name = f"processed_images/{str(uuid.uuid4())[:8]}.jpg"
                    self.upload_data_to_gcs(image_content, blob_name)
                    logger.info(f"Image archived to GCS: {blob_name}")
                    
            elif image_url.startswith('gs://'):
                # Download from GCS
                blob_name = image_url.replace(f'gs://{self.gcs_bucket_name}/', '')
                image_content = self.download_from_gcs(blob_name)
                
            else:
                # Local file path
                with open(image_url, 'rb') as image_file:
                    image_content = image_file.read()
            
            # Create Vision API image object
            image = vision.Image(content=image_content)
            
            # Configure text detection with language hints
            image_context = vision.ImageContext(language_hints=[language_hint])
            
            # Perform text detection
            response = vision_client.text_detection(image=image, image_context=image_context)
            texts = response.text_annotations
            
            # Handle API errors
            if response.error.message:
                logger.error(f"Vision API error: {response.error.message}")
                return {
                    "status": "error",
                    "error": response.error.message,
                    "image_url": image_url
                }
            
            # Extract text and confidence scores
            extracted_text = ""
            text_blocks = []
            
            if texts:
                # First annotation contains the full detected text
                extracted_text = texts[0].description
                
                # Extract individual text blocks with positions
                for text in texts[1:]:  # Skip the first one as it's the full text
                    vertices = [(vertex.x, vertex.y) for vertex in text.bounding_poly.vertices]
                    text_blocks.append({
                        "text": text.description,
                        "bounding_box": vertices,
                        "confidence": getattr(text, 'confidence', 0.0)
                    })
            
            # Calculate overall confidence (simplified)
            overall_confidence = sum([block.get('confidence', 0.0) for block in text_blocks]) / len(text_blocks) if text_blocks else 0.0
            
            result = {
                "status": "success",
                "image_url": image_url,
                "language_hint": language_hint,
                "extracted_text": extracted_text,
                "text_blocks": text_blocks,
                "confidence": round(overall_confidence, 2),
                "processed_at": datetime.now(timezone.utc).isoformat()
            }
            
            # Store result in database for caching
            cache_id = f"ocr_{str(uuid.uuid4())[:8]}"
            self.cache_analysis_result(cache_id, result)
            
            logger.info(f"Successfully extracted text from image. Length: {len(extracted_text)} characters")
            return result
            
        except Exception as e:
            logger.error(f"Error extracting text from image: {e}")
            return {
                "status": "error",
                "error": str(e),
                "image_url": image_url
            }
    
    def process_multimedia_content(self, content_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process multimedia content (images, videos) and extract relevant information
        Integrates with existing content creation pipeline
        """
        try:
            content_type = content_data.get('content_type', '').lower()
            
            if content_type in ['image', 'photo', 'meme']:
                return self._process_image_content(content_data)
            elif content_type in ['video', 'video_clip']:
                return self._process_video_content(content_data)
            else:
                return {
                    "status": "error",
                    "error": f"Unsupported content type: {content_type}"
                }
                
        except Exception as e:
            logger.error(f"Error processing multimedia content: {e}")
            return {"status": "error", "error": str(e)}
    
    def _process_image_content(self, content_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process image content specifically"""
        try:
            image_url = content_data.get('media_url') or content_data.get('url')
            if not image_url:
                return {"status": "error", "error": "No image URL provided"}
            
            # Extract text using OCR
            ocr_result = self.extract_text_from_image(
                image_url, 
                language_hint=content_data.get('language_code', 'en')
            )
            
            if ocr_result['status'] == 'success':
                # Enhance content data with OCR results
                enhanced_content = content_data.copy()
                enhanced_content.update({
                    'extracted_text': ocr_result['extracted_text'],
                    'ocr_confidence': ocr_result['confidence'],
                    'text_blocks': ocr_result['text_blocks'],
                    'processed_at': datetime.now(timezone.utc)
                })
                
                # Create content item with OCR data
                content_id = self.create_content_item(enhanced_content)
                
                # Check for narrative matches
                narrative_matches = self._check_narrative_matches(ocr_result['extracted_text'])
                
                # Check for lexicon term matches
                lexicon_matches = self._check_lexicon_matches(ocr_result['extracted_text'])
                
                return {
                    "status": "success",
                    "content_id": content_id,
                    "ocr_result": ocr_result,
                    "narrative_matches": narrative_matches,
                    "lexicon_matches": lexicon_matches
                }
            else:
                return ocr_result
                
        except Exception as e:
            logger.error(f"Error processing image content: {e}")
            return {"status": "error", "error": str(e)}
    
    def _process_video_content(self, content_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process video content - placeholder for future video analysis"""
        try:
            # For now, create the content item without audio processing
            # Future: integrate with Speech-to-Text API for audio extraction
            video_url = content_data.get('media_url') or content_data.get('url')
            
            enhanced_content = content_data.copy()
            enhanced_content.update({
                'processing_note': 'Video processing not yet implemented',
                'processed_at': datetime.now(timezone.utc)
            })
            
            content_id = self.create_content_item(enhanced_content)
            
            return {
                "status": "partial_success",
                "content_id": content_id,
                "note": "Video created but audio/visual analysis not yet implemented"
            }
            
        except Exception as e:
            logger.error(f"Error processing video content: {e}")
            return {"status": "error", "error": str(e)}
    
    def _check_narrative_matches(self, text: str) -> List[Dict[str, Any]]:
        """Check extracted text against known narratives"""
        try:
            # Search for narratives that might match the content
            # Use text similarity or keyword matching
            text_lower = text.lower()
            
            narrative_matches = []
            
            # Get sample of narratives to check against
            narratives = self.search_narratives({}, limit=100)
            
            for narrative in narratives:
                # Simple keyword matching (can be enhanced with ML similarity)
                narrative_text = narrative.get('narrative_text', '').lower()
                keywords = narrative.get('keywords', [])
                
                # Check for direct text overlap or keyword matches
                if any(keyword.lower() in text_lower for keyword in keywords):
                    match_score = sum(1 for keyword in keywords if keyword.lower() in text_lower) / len(keywords)
                    narrative_matches.append({
                        "narrative_id": narrative['uuid'],
                        "category": narrative.get('category'),
                        "match_score": round(match_score, 2),
                        "matched_keywords": [kw for kw in keywords if kw.lower() in text_lower]
                    })
            
            # Sort by match score
            narrative_matches.sort(key=lambda x: x['match_score'], reverse=True)
            return narrative_matches[:5]  # Top 5 matches
            
        except Exception as e:
            logger.error(f"Error checking narrative matches: {e}")
            return []
    
    def _check_lexicon_matches(self, text: str) -> List[Dict[str, Any]]:
        """Check extracted text against lexicon terms"""
        try:
            text_lower = text.lower()
            lexicon_matches = []
            
            # Get all lexicon terms
            lexicon_terms = list(self.mongo_db.lexicons.find({}))
            
            for term_data in lexicon_terms:
                term = term_data.get('term', '').lower()
                variants = term_data.get('variants', [])
                
                # Check if term or variants appear in text
                if term in text_lower or any(variant.lower() in text_lower for variant in variants):
                    # Increment usage count
                    self.increment_lexicon_usage(term_data['term'], term_data.get('language_code', 'en'))
                    
                    lexicon_matches.append({
                        "term": term_data['term'],
                        "definition": term_data.get('definition'),
                        "language_code": term_data.get('language_code'),
                        "context": term_data.get('context_type')
                    })
            
            return lexicon_matches
            
        except Exception as e:
            logger.error(f"Error checking lexicon matches: {e}")
            return []
    
    def bulk_process_multimedia_files(self, file_list: List[str], batch_size: int = 10) -> Dict[str, Any]:
        """
        Process multiple multimedia files in batches
        Useful for bulk analysis of images/videos
        """
        try:
            results = {
                "processed": [],
                "failed": [],
                "summary": {
                    "total_files": len(file_list),
                    "success_count": 0,
                    "error_count": 0
                }
            }
            
            for i in range(0, len(file_list), batch_size):
                batch = file_list[i:i + batch_size]
                logger.info(f"Processing batch {i//batch_size + 1}: {len(batch)} files")
                
                for file_path in batch:
                    try:
                        # Determine content type based on file extension
                        content_type = "image" if file_path.lower().endswith(('.jpg', '.jpeg', '.png', '.gif')) else "video"
                        
                        content_data = {
                            "media_url": file_path,
                            "content_type": content_type,
                            "source_platform": "bulk_upload",
                            "batch_id": f"batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                        }
                        
                        result = self.process_multimedia_content(content_data)
                        
                        if result['status'] in ['success', 'partial_success']:
                            results['processed'].append({
                                "file": file_path,
                                "result": result
                            })
                            results['summary']['success_count'] += 1
                        else:
                            results['failed'].append({
                                "file": file_path,
                                "error": result.get('error', 'Unknown error')
                            })
                            results['summary']['error_count'] += 1
                            
                    except Exception as e:
                        logger.error(f"Error processing file {file_path}: {e}")
                        results['failed'].append({
                            "file": file_path,
                            "error": str(e)
                        })
                        results['summary']['error_count'] += 1
            
            logger.info(f"Bulk processing complete: {results['summary']['success_count']}/{results['summary']['total_files']} files processed successfully")
            return results
            
        except Exception as e:
            logger.error(f"Error in bulk multimedia processing: {e}")
            return {"status": "error", "error": str(e)}

    # ==================== UTILITY METHODS ====================
    
    def health_check(self) -> Dict[str, bool]:
        """Check health of all database connections"""
        health = {
            "mongodb": False,
            "firestore": False,
            "gcs": False
        }
        
        try:
            # Test MongoDB
            self.mongo_client.admin.command('ping')
            health["mongodb"] = True
        except Exception as e:
            logger.error(f"MongoDB health check failed: {e}")
        
        try:
            # Test Firestore
            self.firestore_client.collection('health_check').document('test').set({'timestamp': datetime.now(timezone.utc)})
            health["firestore"] = True
        except Exception as e:
            logger.error(f"Firestore health check failed: {e}")
        
        try:
            # Test GCS
            self.gcs_bucket.get_blob('health_check') # This will return None if not exists, but won't fail
            health["gcs"] = True
        except Exception as e:
            logger.error(f"GCS health check failed: {e}")
        
        return health
    
    def close_connections(self):
        """Close all database connections"""
        try:
            self.mongo_client.close()
            logger.info("Database connections closed")
        except Exception as e:
            logger.error(f"Error closing connections: {e}")
    
    def __del__(self):
        """Cleanup on object destruction"""
        self.close_connections()
