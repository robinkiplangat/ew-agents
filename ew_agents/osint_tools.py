from google.adk.tools import FunctionTool
from typing import Optional, List
import datetime
import os
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def classify_narrative(text: str, source_platform: str = "unknown") -> dict:
    """
    Uses the fine-tuned DISARM model to classify text into misinformation themes.
    Returns a list of identified themes and their confidence scores.
    """
    print(f"[OsintTool] Classifying narrative for text from {source_platform}: '{text[:50]}...'")
    
    if not text or not text.strip():
        return {
            "status": "error", 
            "message": "Input text cannot be empty.",
            "has_content": False
        }

    # Try to use the fine-tuned DISARM model first
    try:
        logger.info("🔄 Attempting to import model_loader...")
        from .model_loader import get_disarm_model
        logger.info("✅ Model loader imported successfully")
        
        logger.info("🔄 Getting DISARM model...")
        disarm_model = get_disarm_model()
        logger.info(f"📊 DISARM model result: {disarm_model}")
        
        if disarm_model and disarm_model.is_available():
            logger.info("🔍 Using fine-tuned DISARM model for narrative classification")
            
            # Create prompt for the fine-tuned model
            prompt = f"""Analyze the following election-related content for narrative classification and actor identification:

Content: {text}

Please classify this content using the DISARM framework and identify:
1. Narrative themes and techniques
2. Political actors involved
3. Misinformation indicators
4. Risk assessment

Provide your analysis in JSON format."""

            # Generate response using fine-tuned model
            response = disarm_model.generate_response(prompt)
            
            if response:
                logger.info("✅ Fine-tuned model analysis completed")
                return {
                    "status": "success",
                    "narrative_classification": {
                        "theme": "disarm_analysis",
                        "confidence": 0.85,
                        "details": response[:500],
                        "analysis_method": "fine_tuned_disarm_model"
                    },
                    "actors_identified": [],
                    "misinformation_indicators": [],
                    "text": text[:200],
                    "source_platform": source_platform,
                    "confidence": 0.85,
                    "analysis_method": "fine_tuned_disarm_model"
                }
    except Exception as e:
        logger.warning(f"⚠️ Fine-tuned model failed, falling back to pattern matching: {e}")

    # Fallback to pattern matching - be explicit about limitations
    try:
        narrative_result = analyze_narrative_patterns_fallback(text)
        actors = extract_political_actors_fallback(text)
        misinfo_indicators = detect_misinformation_patterns_fallback(text)
        
        # Only return successful result if we actually found meaningful content
        has_meaningful_results = (
            narrative_result.get("confidence", 0) > 0.2 or 
            len(actors) > 0 or 
            len(misinfo_indicators) > 0
        )
        
        if has_meaningful_results:
            return {
                "status": "success",
                "narrative_classification": narrative_result,
                "actors_identified": actors,
                "misinformation_indicators": misinfo_indicators,
                "text": text[:200],
                "source_platform": source_platform,
                "confidence": narrative_result.get("confidence", 0.3),
                "analysis_method": "pattern_matching_fallback",
                "has_content": True,
                "limitation_note": "Analysis performed using pattern matching due to model unavailability"
            }
        else:
            return {
                "status": "limited_analysis",
                "message": "Pattern matching found minimal relevant indicators in the provided text",
                "narrative_classification": narrative_result,
                "actors_identified": actors,
                "misinformation_indicators": misinfo_indicators,
                "text": text[:200],
                "source_platform": source_platform,
                "confidence": 0.1,
                "analysis_method": "pattern_matching_fallback",
                "has_content": False,
                "limitation_note": "Limited analysis - no strong patterns detected"
            }
    except Exception as e:
        logger.error(f"Pattern matching fallback failed: {e}")
        return {
            "status": "error", 
            "message": f"All narrative classification methods failed: {str(e)}",
            "text": text[:100],
            "source_platform": source_platform,
            "has_content": False,
            "analysis_method": "failed"
        }

def track_keywords(keywords: str, platforms: str = "twitter,facebook") -> dict:
    """
    Monitors for specific keywords and hashtags across specified platforms.
    
    Args:
        keywords: Comma-separated list of keywords to track
        platforms: Comma-separated list of platforms to monitor
    
    TODO: Implement real social media API integrations for keyword tracking.
    """
    # Convert comma-separated strings to lists
    keywords_list = [k.strip() for k in keywords.split(',') if k.strip()]
    platforms_list = [p.strip() for p in platforms.split(',') if p.strip()]
    
    print(f"[OsintTool] Tracking keywords: {keywords_list} on platforms: {platforms_list}")

    # TODO: Implement real platform API integrations
    # Examples:
    # - Twitter API v2 for tweet monitoring
    # - Facebook Graph API for post tracking
    # - Reddit API for subreddit monitoring
    # - TikTok Research API for content analysis
    
    # Each platform would require:
    # 1. API authentication
    # 2. Real-time or batch query mechanisms
    # 3. Rate limiting handling
    # 4. Data normalization and storage

    return {
        "status": "error",
        "message": "Real keyword tracking not yet implemented",
        "keywords": keywords_list,
        "platforms": platforms_list
    }

def calculate_influence_metrics(actor_id: str, platform: str) -> dict:
    """
    Analyzes graph data and other metrics to calculate influence scores for actors.
    
    TODO: Implement real influence calculation using:
    - Social network analysis
    - Engagement rate calculations  
    - Reach and virality metrics
    - Historical influence tracking
    """
    print(f"[OsintTool] Calculating influence metrics for actor: {actor_id} on {platform}")

    # TODO: Implement real influence metrics calculation
    # This should:
    # 1. Fetch actor data from platform APIs
    # 2. Calculate network centrality metrics
    # 3. Analyze engagement patterns
    # 4. Generate composite influence scores
    # 5. Store results in analytical database

    return {
        "status": "error",
        "message": "Real influence metrics calculation not yet implemented",
        "actor_id": actor_id,
        "platform": platform
    }

def detect_coordinated_behavior(account_ids: str, platform: str, time_window_hours: int = 24) -> dict:
    """
    Implements algorithms to identify inauthentic network activity among a list of accounts.
    
    Args:
        account_ids: Comma-separated list of account IDs
        platform: Platform name (twitter, facebook, etc.)
        time_window_hours: Time window for analysis in hours
    
    TODO: Implement real coordinated behavior detection using:
    - Temporal analysis of posting patterns
    - Content similarity analysis
    - Network analysis for unusual connections
    - Behavioral anomaly detection
    """
    # Convert comma-separated string to list
    account_list = [a.strip() for a in account_ids.split(',') if a.strip()]
    
    print(f"[OsintTool] Detecting coordinated behavior for accounts: {account_list} on {platform} within {time_window_hours}h.")
    
    if not account_list:
        return {"status": "error", "message": "Account IDs list cannot be empty."}

    # TODO: Implement real coordinated behavior detection
    # This should use algorithms like:
    # 1. Temporal clustering of activities
    # 2. Content fingerprinting and similarity matching
    # 3. Network analysis for suspicious connections
    # 4. Machine learning models for behavior classification

    return {
        "status": "error",
        "message": "Real coordinated behavior detection not yet implemented",
        "platform": platform,
        "accounts_analyzed": account_list,
        "time_window_hours": time_window_hours
    }

def generate_actor_profile(actor_id: str, platform: str) -> dict:
    """
    Creates detailed profiles of actors including their activity history, known associations, and narrative involvement.
    
    TODO: Implement real actor profiling using:
    - Platform API data collection
    - Historical activity analysis
    - Network relationship mapping
    - Risk assessment scoring
    """
    print(f"[OsintTool] Generating profile for actor: {actor_id} on {platform}")

    # TODO: Implement real actor profiling
    # This should:
    # 1. Collect comprehensive data from platform APIs
    # 2. Analyze historical posting patterns
    # 3. Map network relationships
    # 4. Calculate risk and influence scores
    # 5. Store profile data for future reference

    return {
        "status": "error",
        "message": "Real actor profiling not yet implemented",
        "actor_id": actor_id,
        "platform": platform
    }

def classify_image_content_theme(image_url: str, associated_text: Optional[str] = None) -> dict:
    """
    Classifies an image's content theme, potentially aided by associated text.
    
    TODO: Implement real image classification using:
    - Computer vision models for object detection
    - OCR for text extraction from images
    - Visual similarity matching against known misinformation imagery
    - Multi-modal analysis combining image and text
    """
    print(f"[OsintTool-Multimedia] Image Classification: Analyzing image: {image_url}. Associated text: '{associated_text[:50] if associated_text else 'None'}...'")

    # TODO: Implement real image classification
    # This should use:
    # 1. Computer vision models (CLIP, ResNet, etc.)
    # 2. OCR engines for text extraction
    # 3. Hash-based similarity matching
    # 4. Multi-modal ML models for content understanding

    return {
        "status": "error",
        "message": "Real image content classification not yet implemented",
        "image_url": image_url,
        "associated_text": associated_text
    }

# =============================================================================
# Batch classification helper
# =============================================================================

def classify_narratives_batch(texts: List[str]) -> dict:
    """Batch wrapper applying classify_narrative over multiple texts and aggregating a best pick.
    Reduces upstream agent call overhead by allowing a single batched call.
    """
    results = []
    for t in texts or []:
        results.append(classify_narrative(t))
    # choose best by confidence when available
    best = None
    best_conf = -1.0
    for r in results:
        if r.get("status") == "success":
            conf = r.get("narrative_classification", {}).get("confidence", 0.0)
            if conf > best_conf:
                best_conf = conf
                best = r
    return {
        "status": "success" if results else "empty",
        "count": len(results),
        "best": best,
        "items": results,
    }

# =============================================================================
# Fallback Analysis Functions
# =============================================================================

def analyze_narrative_patterns_fallback(text: str) -> dict:
    """Enhanced fallback narrative pattern analysis using comprehensive keyword matching"""
    import re
    
    text_lower = text.lower()
    words = text_lower.split()
    
    # Enhanced keyword categories for African elections
    narrative_patterns = {
        "election_fraud": {
            "keywords": ['rigged', 'fraud', 'manipulation', 'fake', 'ghost', 'stolen', 'cheating', 
                        'ballot stuffing', 'vote buying', 'result manipulation', 'inec', 'electoral'],
            "weight": 1.0
        },
        "violence_incitement": {
            "keywords": ['violence', 'fight', 'war', 'attack', 'threat', 'intimidation', 'clash',
                        'kill', 'destroy', 'burn', 'militia', 'thugs', 'cultists'],
            "weight": 1.2
        },
        "ethnic_religious_division": {
            "keywords": ['ethnic', 'religious', 'tribal', 'christian', 'muslim', 'yoruba', 'igbo', 'hausa',
                        'fulani', 'sectarian', 'indigene', 'settler', 'kafir', 'infidel'],
            "weight": 1.1
        },
        "misinformation": {
            "keywords": ['fake news', 'false', 'lie', 'hoax', 'propaganda', 'misinformation',
                        'disinformation', 'conspiracy', 'rumor', 'unverified'],
            "weight": 0.9
        },
        "governance_criticism": {
            "keywords": ['corruption', 'incompetent', 'failure', 'disappointment', 'bad governance',
                        'mismanagement', 'nepotism', 'cabalism', 'one man government'],
            "weight": 0.8
        }
    }
    
    # Calculate scores for each narrative
    narrative_scores = {}
    detected_keywords = {}
    
    for narrative, config in narrative_patterns.items():
        score = 0
        found_keywords = []
        
        for keyword in config["keywords"]:
            if keyword in text_lower:
                score += config["weight"]
                found_keywords.append(keyword)
        
        narrative_scores[narrative] = score
        detected_keywords[narrative] = found_keywords
    
    # Find primary narrative
    primary_theme = max(narrative_scores, key=narrative_scores.get) if any(narrative_scores.values()) else "general_political"
    max_score = narrative_scores.get(primary_theme, 0)
    confidence = min(max_score * 0.15, 0.85)  # Cap confidence for fallback method
    
    # Generate threat indicators
    threat_indicators = []
    if narrative_scores.get("violence_incitement", 0) > 1:
        threat_indicators.append("Violence incitement detected")
    if narrative_scores.get("ethnic_religious_division", 0) > 1:
        threat_indicators.append("Ethnic/religious division detected")
    if narrative_scores.get("election_fraud", 0) > 2:
        threat_indicators.append("Election fraud allegations detected")
    
    # Determine threat level
    total_threat_score = (narrative_scores.get("violence_incitement", 0) * 2 + 
                         narrative_scores.get("ethnic_religious_division", 0) * 1.5 +
                         narrative_scores.get("election_fraud", 0))
    
    if total_threat_score > 4:
        threat_level = "High"
    elif total_threat_score > 2:
        threat_level = "Medium"
    elif total_threat_score > 0:
        threat_level = "Low"
    else:
        threat_level = "Minimal"
    
    return {
        "theme": f"{primary_theme.replace('_', ' ').title()}: {', '.join(detected_keywords.get(primary_theme, [])[:3])}",
        "confidence": confidence,
        "details": f"Pattern matching identified {len([k for keywords in detected_keywords.values() for k in keywords])} relevant indicators. Primary theme: {primary_theme}. Detected patterns: {dict((k, len(v)) for k, v in detected_keywords.items() if v)}",
        "narrative_scores": narrative_scores,
        "detected_keywords": detected_keywords,
        "threat_indicators": threat_indicators,
        "threat_level": threat_level
    }

def extract_political_actors_fallback(text: str) -> list:
    """Enhanced political actors extraction using comprehensive pattern matching"""
    import re
    
    actors = []
    text_lower = text.lower()
    
    # Enhanced political figures database with variations
    political_figures = {
        "Bola Ahmed Tinubu": ["tinubu", "bola", "jagaban", "bat", "bola ahmed tinubu"],
        "Peter Obi": ["peter obi", "obi", "peter gregory obi"],
        "Atiku Abubakar": ["atiku", "abubakar", "atiku abubakar", "waziri"],
        "Rabiu Kwankwaso": ["kwankwaso", "rabiu", "rabiu kwankwaso"],
        "Muhammadu Buhari": ["buhari", "muhammadu buhari", "pmb"],
        "Yemi Osinbajo": ["osinbajo", "yemi", "yemi osinbajo", "vp"],
        "Kashim Shettima": ["shettima", "kashim", "kashim shettima"],
        "Yusuf Datti": ["datti", "yusuf datti", "baba-ahmed"],
        "Ifeanyi Okowa": ["okowa", "ifeanyi", "ifeanyi okowa"]
    }
    
    for full_name, variations in political_figures.items():
        for variation in variations:
            if variation in text_lower:
                # Determine context based on surrounding text
                context = "mentioned"
                if any(word in text_lower for word in ["support", "vote", "endorse"]):
                    context = "supported"
                elif any(word in text_lower for word in ["criticize", "attack", "against"]):
                    context = "criticized"
                elif any(word in text_lower for word in ["president", "governor", "senator"]):
                    context = "political_role"
                
                actors.append({
                    "name": full_name,
                    "role": "Political figure/candidate",
                    "type": "politician",
                    "context": context,
                    "confidence": 0.9,
                    "matched_term": variation
                })
                break  # Only match once per person
    
    # Enhanced political parties
    political_parties = {
        "APC": ["apc", "all progressives congress", "ruling party"],
        "PDP": ["pdp", "peoples democratic party"],
        "Labour Party": ["labour", "labour party", "lp"],
        "NNPP": ["nnpp", "new nigeria peoples party"],
        "SDP": ["sdp", "social democratic party"],
        "ADC": ["adc", "african democratic congress"],
        "APGA": ["apga", "all progressives grand alliance"]
    }
    
    for party_name, variations in political_parties.items():
        for variation in variations:
            if variation in text_lower:
                actors.append({
                    "name": party_name,
                    "role": "Political party",
                    "type": "political_party",
                    "context": "mentioned",
                    "confidence": 0.8,
                    "matched_term": variation
                })
                break
    
    # Enhanced institutions and electoral bodies
    institutions = {
        "INEC": ["inec", "independent national electoral commission", "electoral commission"],
        "BVAS": ["bvas", "bimodal voter accreditation system"],
        "IREV": ["irev", "inec result viewing portal"],
        "Federal High Court": ["federal high court", "election tribunal"],
        "Supreme Court": ["supreme court", "apex court"],
        "Police": ["nigeria police", "police force", "law enforcement"],
        "DSS": ["dss", "department of state services", "state security"]
    }
    
    for inst_name, variations in institutions.items():
        for variation in variations:
            if variation in text_lower:
                actors.append({
                    "name": inst_name,
                    "role": "Institution/Electoral body",
                    "type": "institution",
                    "context": "mentioned",
                    "confidence": 0.7,
                    "matched_term": variation
                })
                break
    
    return actors

def detect_misinformation_patterns_fallback(text: str) -> list:
    """Detect misinformation patterns using heuristics"""
    patterns = []
    text_lower = text.lower()
    
    # Unverified claims patterns
    if any(phrase in text_lower for phrase in ['breaking:', 'confirmed:', 'reports confirm']):
        if any(word in text_lower for word in ['million', 'thousand', 'massive', 'widespread']):
            patterns.append({
                "type": "unverified_statistics",
                "confidence": 0.6,
                "description": "Unverified numerical claims"
            })
    
    # Emotional manipulation
    emotion_words = ['shocking', 'devastating', 'outrageous', 'betrayal', 'disaster']
    emotion_count = sum(1 for word in emotion_words if word in text_lower)
    if emotion_count >= 2:
        patterns.append({
            "type": "emotional_manipulation",
            "confidence": min(emotion_count * 0.2, 0.8),
            "description": "High emotional language"
        })
    
    # Conspiracy patterns
    conspiracy_phrases = ['they don\'t want you to know', 'hidden truth', 'cover up', 'exposed']
    if any(phrase in text_lower for phrase in conspiracy_phrases):
        patterns.append({
            "type": "conspiracy_narrative",
            "confidence": 0.7,
            "description": "Conspiracy-style language"
        })
    
    return patterns

# Create FunctionTool instances
classify_narrative_tool = FunctionTool(
    func=classify_narrative,
)
track_keywords_tool = FunctionTool(
    func=track_keywords,
)
calculate_influence_metrics_tool = FunctionTool(
    func=calculate_influence_metrics,
)
detect_coordinated_behavior_tool = FunctionTool(
    func=detect_coordinated_behavior,
)
generate_actor_profile_tool = FunctionTool(
    func=generate_actor_profile,
)
classify_image_content_theme_tool = FunctionTool(
    func=classify_image_content_theme,
)