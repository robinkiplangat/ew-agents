from google.adk.tools import FunctionTool
import datetime
import os
import logging
from pymongo import MongoClient

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Direct MongoDB Atlas connection
def get_mongo_connection():
    """Get MongoDB Atlas connection"""
    try:
        mongo_uri = os.getenv('MONGODB_ATLAS_URI', 'mongodb+srv://ew_ml:moHsc5i6gYFrLsvL@ewcluster1.fpkzpxg.mongodb.net/')
        client = MongoClient(mongo_uri, tlsAllowInvalidCertificates=True)
        db = client["election_watch"]
        return client, db
    except Exception as e:
        print(f"[LexiconTool] Failed to connect to MongoDB Atlas: {e}")
        return None, None

def update_lexicon_term(
    term: str,
    definition: str,
    category: str,
    language_code: str,
    severity_level: str,
    tags: list,
    related_terms = None,
    source: str = "manual input"
) -> dict:
    """
    Adds or updates a term in the multilingual lexicon using MongoDB Atlas.
    If the term exists, it updates its details. Otherwise, it adds a new term.
    """
    print(f"[LexiconTool] Updating lexicon for term '{term}' in language '{language_code}'")
    
    client, db = get_mongo_connection()
    if db is None:
        return {
            "status": "error",
            "message": "Unable to connect to MongoDB Atlas",
            "term": term,
            "language_code": language_code
        }
    
    if related_terms is None:
        related_terms = []

    try:
        # Check if term already exists
        existing_term = db.lexicons.find_one({"term": term, "language_code": language_code})
        
        lexicon_data = {
            "term": term,
            "language_code": language_code,
            "definition": definition,
            "tags": list(set(tags)),  # Ensure unique tags
            "related_terms": list(set(related_terms)),  # Ensure unique related terms
            "source": source,
            "usage_count": existing_term.get("usage_count", 0) if existing_term else 0,
            "updated_at": datetime.datetime.now(datetime.timezone.utc)
        }
        
        if existing_term:
            # Update existing term
            result = db.lexicons.update_one(
                {"term": term, "language_code": language_code},
                {"$set": lexicon_data}
            )
            operation = "updated"
            lexicon_data["_id"] = existing_term["_id"]
        else:
            # Create new term
            lexicon_data["created_at"] = datetime.datetime.now(datetime.timezone.utc)
            result = db.lexicons.insert_one(lexicon_data)
            lexicon_data["_id"] = result.inserted_id
            operation = "added"

        client.close()
        
        return {
            "status": "success",
            "operation": operation,
            "term": term,
            "language_code": language_code,
            "entry": {
                "term": lexicon_data["term"],
                "definition": lexicon_data["definition"],
                "tags": lexicon_data["tags"],
                "related_terms": lexicon_data["related_terms"],
                "source": lexicon_data["source"],
                "usage_count": lexicon_data["usage_count"]
            }
        }
        
    except Exception as e:
        client.close() if client else None
        print(f"[LexiconTool] Error updating lexicon term: {e}")
        return {
            "status": "error",
            "message": f"Database error: {str(e)}",
            "term": term,
            "language_code": language_code
        }

def get_lexicon_term(term: str, language_code: str) -> dict:
    """
    Retrieves a term from the lexicon for a specific language using MongoDB Atlas.
    """
    print(f"[LexiconTool] Getting lexicon term '{term}' in language '{language_code}'")
    
    client, db = get_mongo_connection()
    if db is None:
        return {
            "status": "error",
            "message": "Unable to connect to MongoDB Atlas",
            "term": term,
            "language_code": language_code
        }
    
    try:
        result = db.lexicons.find_one({"term": term, "language_code": language_code})
        
        if result:
            # Increment usage count
            db.lexicons.update_one(
                {"term": term, "language_code": language_code},
                {"$inc": {"usage_count": 1}, "$set": {"updated_at": datetime.datetime.now(datetime.timezone.utc)}}
            )
            
            client.close()
            
            return {
                "status": "success",
                "term": term,
                "language_code": language_code,
                "entry": {
                    "term": result["term"],
                    "definition": result.get("definition", ""),
                    "tags": result.get("tags", []),
                    "related_terms": result.get("related_terms", []),
                    "source": result.get("source", ""),
                    "usage_count": result.get("usage_count", 0) + 1,
                    "created_at": result.get("created_at"),
                    "updated_at": result.get("updated_at")
                }
            }
        else:
            client.close()
            return {
                "status": "not_found",
                "term": term,
                "language_code": language_code,
                "message": f"Term '{term}' not found in language '{language_code}'."
            }
            
    except Exception as e:
        client.close() if client else None
        print(f"[LexiconTool] Error retrieving lexicon term: {e}")
        return {
            "status": "error",
            "message": f"Database error: {str(e)}",
            "term": term,
            "language_code": language_code
        }

def detect_coded_language(
    text: str,
    language_code: str = "en",
    context_keywords = None
) -> dict:
    """
    Detects new or coded language within a text sample using NLP analysis and existing lexicon.
    """
    if context_keywords is None:
        context_keywords = []
    
    print(f"[LexiconTool] Detecting coded language in sample (lang: {language_code}): '{text[:50]}...' with context: {context_keywords}")

    client, db = get_mongo_connection()
    if db is None:
        return {
            "status": "error",
            "message": "Unable to connect to MongoDB Atlas for coded language detection",
            "text_sample": text,
            "language_code": language_code
        }

    try:
        # Search existing lexicon for potential matches
        potential_terms = []
        words = text.lower().split()
        
        for word in words:
            if len(word) > 3:  # Only check words longer than 3 characters
                # Search for partial matches in lexicon
                matches = list(db.lexicons.find({
                    "language_code": language_code,
                    "$or": [
                        {"term": {"$regex": word, "$options": "i"}},
                        {"definition": {"$regex": word, "$options": "i"}},
                        {"tags": {"$regex": word, "$options": "i"}}
                    ]
                }))
                
                for match in matches:
                    potential_terms.append({
                        "term_candidate": word,
                        "existing_term": match["term"],
                        "definition": match.get("definition", ""),
                        "confidence": 0.7,  # Basic string matching confidence
                        "language_code": language_code,
                        "context_phrase": text,
                        "match_type": "lexicon_match"
                    })
        
        client.close()
        
        # TODO: Implement advanced detection algorithms here
        # - Semantic similarity analysis
        # - Anomaly detection for unusual word patterns
        # - Context-aware coded language identification
        
        if potential_terms:
            return {
                "status": "success",
                "text_sample": text,
                "language_code": language_code,
                "potential_coded_terms": potential_terms,
                "message": f"Found {len(potential_terms)} potential coded language matches in lexicon."
            }
        else:
            return {
                "status": "no_coded_language_detected",
                "text_sample": text,
                "language_code": language_code,
                "message": "No coded language patterns detected with current lexicon."
            }
            
    except Exception as e:
        client.close() if client else None
        print(f"[LexiconTool] Error in coded language detection: {e}")
        return {
            "status": "error",
            "message": f"Error in coded language detection: {str(e)}",
            "text_sample": text,
            "language_code": language_code
        }

def translate_term(term: str, source_lang: str, target_lang: str) -> dict:
    """
    Translates a term between languages using the lexicon database.
    
    TODO: Implement real translation service integration (Google Translate, Azure Translator, etc.)
    For now, this searches the lexicon for equivalent terms in different languages.
    """
    print(f"[LexiconTool] Translating '{term}' from {source_lang} to {target_lang}")
    
    client, db = get_mongo_connection()
    if db is None:
        return {
            "status": "error",
            "message": "Unable to connect to MongoDB Atlas for translation",
            "original_term": term,
            "source_language": source_lang,
            "target_language": target_lang
        }
    
    try:
        # First, get the source term to understand its meaning
        source_term = db.lexicons.find_one({"term": term, "language_code": source_lang})
        
        if not source_term:
            client.close()
            return {
                "status": "error",
                "message": f"Source term '{term}' not found in language '{source_lang}'",
                "original_term": term,
                "source_language": source_lang,
                "target_language": target_lang
            }
        
        # Search for equivalent terms in target language based on definition or tags
        definition = source_term.get("definition", "")
        tags = source_term.get("tags", [])
        
        # Search by definition keywords
        target_matches = []
        if definition:
            definition_matches = list(db.lexicons.find({
                "language_code": target_lang,
                "definition": {"$regex": definition, "$options": "i"}
            }))
            target_matches.extend(definition_matches)
        
        # Search by tags
        for tag in tags:
            tag_matches = list(db.lexicons.find({
                "language_code": target_lang,
                "tags": {"$regex": tag, "$options": "i"}
            }))
            target_matches.extend(tag_matches)
        
        # Remove duplicates
        unique_matches = {match["term"]: match for match in target_matches}.values()
        
        client.close()
        
        if unique_matches:
            # Return the best match (for now, just the first one)
            best_match = list(unique_matches)[0]
            return {
                "status": "success",
                "original_term": term,
                "source_language": source_lang,
                "target_language": target_lang,
                "translated_term": best_match["term"],
                "definition": best_match.get("definition", ""),
                "confidence": 0.8,  # Basic lexicon-based translation confidence
                "method": "lexicon_lookup"
            }
        else:
            return {
                "status": "no_translation_found",
                "message": f"No equivalent term found for '{term}' in language '{target_lang}'",
                "original_term": term,
                "source_language": source_lang,
                "target_language": target_lang
            }
            
    except Exception as e:
        client.close() if client else None
        print(f"[LexiconTool] Error in translation: {e}")
        return {
            "status": "error",
            "message": f"Translation error: {str(e)}",
            "original_term": term,
            "source_language": source_lang,
            "target_language": target_lang
        }

# Create FunctionTool instances
update_lexicon_term_tool = FunctionTool(
    func=update_lexicon_term,
)

get_lexicon_term_tool = FunctionTool(
    func=get_lexicon_term,
)

detect_coded_language_tool = FunctionTool(
    func=detect_coded_language,
)

translate_term_tool = FunctionTool(
    func=translate_term,
)
