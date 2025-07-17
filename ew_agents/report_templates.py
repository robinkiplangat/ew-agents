"""
ElectionWatch Agent Report Templates

This module contains standardized JSON templates that agents can use to structure
their analysis reports consistently.
"""

import json
from datetime import datetime
from typing import Dict, Any, List, Optional

class ElectionWatchReportTemplate:
    """
    Standardized templates for ElectionWatch agent reports
    """
    
    @staticmethod
    def get_comprehensive_analysis_template() -> Dict[str, Any]:
        """
        Returns a comprehensive analysis report template for election monitoring
        """
        return {
            "report_metadata": {
                "report_id": "",
                "analysis_timestamp": datetime.now().isoformat(),
                "analyzing_agent": "",
                "report_version": "1.0",
                "confidence_level": ""  # HIGH, MEDIUM, LOW
            },
            "content_analysis": {
                "source_content": {
                    "original_text": "",
                    "content_type": "",  # text, image, video, audio
                    "source_platform": "",  # Twitter, Facebook, WhatsApp, etc.
                    "source_url": "",
                    "author_handle": "",
                    "publication_date": "",
                    "language_detected": ""
                },
                "narrative_classification": {
                    "primary_theme": "",
                    "secondary_themes": [],
                    "threat_level": "",  # CRITICAL, HIGH, MEDIUM, LOW, MINIMAL
                    "classification_confidence": 0.0,  # 0.0 to 1.0
                    "misinformation_type": "",  # vote_buying, voter_intimidation, false_results, etc.
                    "narrative_details": "",
                    "targeting_demographics": []
                }
            },
            "actors_identified": [
                {
                    "actor_id": "",
                    "name": "",
                    "role": "",  # politician, influencer, organization, bot, etc.
                    "affiliation": "",  # political party, ethnic group, etc.
                    "influence_level": "",  # HIGH, MEDIUM, LOW
                    "verification_status": "",  # verified, unverified, suspicious
                    "follower_count": 0,
                    "account_creation_date": "",
                    "previous_violations": []
                }
            ],
            "lexicon_analysis": {
                "terms_detected": [
                    {
                        "term": "",
                        "language": "",
                        "category": "",  # ethnic_slur, vote_buying, intimidation, etc.
                        "severity": "",  # CRITICAL, HIGH, MEDIUM, LOW
                        "context_usage": "",
                        "frequency_in_content": 0,
                        "translation": "",
                        "related_terms": []
                    }
                ],
                "coded_language_detected": False,
                "linguistic_patterns": [],
                "sentiment_analysis": {
                    "overall_sentiment": "",  # positive, negative, neutral
                    "emotional_tone": [],  # anger, fear, hope, etc.
                    "polarization_indicators": []
                }
            },
            "geographic_analysis": {
                "target_locations": [],
                "mentioned_constituencies": [],
                "ethnic_regions_referenced": [],
                "geographic_spread": ""  # local, regional, national
            },
            "trend_analysis": {
                "similar_content_volume": 0,
                "trending_status": "",  # viral, growing, stable, declining
                "time_pattern": "",  # election_period, pre_election, post_election
                "coordinated_behavior_indicators": [],
                "historical_context": ""
            },
            "risk_assessment": {
                "overall_risk_score": 0.0,  # 0.0 to 10.0
                "violence_potential": "",  # HIGH, MEDIUM, LOW
                "electoral_impact": "",  # HIGH, MEDIUM, LOW
                "social_cohesion_threat": "",  # HIGH, MEDIUM, LOW
                "urgency_level": "",  # IMMEDIATE, WITHIN_24H, WITHIN_WEEK, MONITORING
                "recommended_actions": []
            },
            "evidence_chain": {
                "supporting_evidence": [],
                "cross_references": [],
                "verification_sources": [],
                "multimedia_evidence": [],
                "witness_accounts": []
            },
            "recommendations": {
                "immediate_actions": [],
                "monitoring_suggestions": [],
                "stakeholder_notifications": [],
                "follow_up_required": False,
                "escalation_needed": False
            },
            "technical_metadata": {
                "processing_time_seconds": 0.0,
                "models_used": [],
                "tool_chain": [],
                "data_sources_accessed": [],
                "api_calls_made": 0
            }
        }
    
    @staticmethod
    def get_quick_analysis_template() -> Dict[str, Any]:
        """
        Returns a simplified template for quick content analysis
        """
        return {
            "report_metadata": {
                "report_id": "",
                "analysis_timestamp": datetime.now().isoformat(),
                "report_type": "quick_analysis"
            },
            "narrative_classification": {
                "theme": "",
                "threat_level": "",
                "details": ""
            },
            "actors": [
                {
                    "name": "",
                    "affiliation": "",
                    "role": ""
                }
            ],
            "lexicon_terms": [
                {
                    "term": "",
                    "category": "",
                    "context": ""
                }
            ],
            "risk_level": "",
            "date_analyzed": "",
            "recommendations": []
        }
    
    @staticmethod
    def get_multimedia_analysis_template() -> Dict[str, Any]:
        """
        Returns a template specifically for multimedia content analysis
        """
        return {
            "report_metadata": {
                "report_id": "",
                "analysis_timestamp": datetime.now().isoformat(),
                "content_type": "multimedia"
            },
            "media_analysis": {
                "media_url": "",
                "media_type": "",  # image, video, audio
                "extracted_text": "",
                "visual_elements": [],
                "audio_transcript": "",
                "metadata_analysis": {}
            },
            "content_classification": {
                "narrative_theme": "",
                "manipulation_indicators": [],
                "authenticity_score": 0.0,
                "deepfake_probability": 0.0
            },
            "actors_in_media": [],
            "lexicon_terms_found": [],
            "contextual_analysis": {
                "associated_text": "",
                "social_context": "",
                "temporal_context": ""
            },
            "risk_assessment": {
                "viral_potential": "",
                "harm_likelihood": "",
                "recommended_actions": []
            }
        }
    
    @staticmethod
    def get_trend_monitoring_template() -> Dict[str, Any]:
        """
        Returns a template for trend analysis and monitoring reports
        """
        return {
            "report_metadata": {
                "report_id": "",
                "analysis_timestamp": datetime.now().isoformat(),
                "report_type": "trend_monitoring",
                "time_period_analyzed": ""
            },
            "trend_metrics": {
                "narrative_volume": 0,
                "growth_rate": 0.0,
                "peak_activity_times": [],
                "geographic_distribution": {},
                "platform_distribution": {}
            },
            "narrative_evolution": {
                "original_narrative": "",
                "variants_detected": [],
                "mutation_patterns": [],
                "adaptation_strategies": []
            },
            "actor_network": {
                "key_amplifiers": [],
                "influence_network": {},
                "coordination_indicators": [],
                "bot_activity_detected": False
            },
            "early_warning_indicators": {
                "escalation_signals": [],
                "threshold_breaches": [],
                "anomaly_detection": [],
                "intervention_points": []
            },
            "forecast": {
                "predicted_trajectory": "",
                "confidence_interval": "",
                "scenario_projections": [],
                "timeline_estimates": {}
            }
        }
    
    @staticmethod
    def create_report_with_template(template_type: str = "comprehensive", **kwargs) -> Dict[str, Any]:
        """
        Creates a report using the specified template with provided data
        
        Args:
            template_type: Type of template to use (comprehensive, quick, multimedia, trend)
            **kwargs: Data to populate in the template
        
        Returns:
            Dict containing the populated report structure
        """
        templates = {
            "comprehensive": ElectionWatchReportTemplate.get_comprehensive_analysis_template(),
            "quick": ElectionWatchReportTemplate.get_quick_analysis_template(),
            "multimedia": ElectionWatchReportTemplate.get_multimedia_analysis_template(),
            "trend": ElectionWatchReportTemplate.get_trend_monitoring_template()
        }
        
        if template_type not in templates:
            raise ValueError(f"Unknown template type: {template_type}")
        
        template = templates[template_type]
        
        # Update template with provided data
        def update_nested_dict(d: Dict, updates: Dict):
            for key, value in updates.items():
                if key in d:
                    if isinstance(d[key], dict) and isinstance(value, dict):
                        update_nested_dict(d[key], value)
                    else:
                        d[key] = value
        
        update_nested_dict(template, kwargs)
        return template
    
    @staticmethod
    def validate_report_structure(report: Dict[str, Any], template_type: str = "comprehensive") -> bool:
        """
        Validates if a report follows the expected structure
        
        Args:
            report: The report to validate
            template_type: The template type to validate against
        
        Returns:
            bool: True if valid, False otherwise
        """
        try:
            template = ElectionWatchReportTemplate.create_report_with_template(template_type)
            required_keys = set(template.keys())
            report_keys = set(report.keys())
            return required_keys.issubset(report_keys)
        except Exception:
            return False
    
    @staticmethod
    def export_report_as_json(report_data: Dict[str, Any], filename: Optional[str] = None) -> Dict[str, Any]:
        """
        Exports a report as a JSON file
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"electionwatch_report_{timestamp}.json"
        
        try:
            import os
            output_dir = os.path.join(os.path.dirname(__file__), "..", "data", "outputs")
            os.makedirs(output_dir, exist_ok=True)
            
            filepath = os.path.join(output_dir, filename)
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(report_data, f, indent=2, ensure_ascii=False)
            
            return {
                "status": "success",
                "filename": filename,
                "filepath": filepath,
                "message": f"Report exported successfully to {filename}"
            }
        except Exception as e:
            return {
                "status": "error",
                "message": f"Failed to export report: {str(e)}"
            }

# Example usage and utility functions
def generate_sample_report():
    """Generate a sample report for testing purposes"""
    sample_data = {
        "report_metadata": {
            "report_id": "EW-2025-001",
            "analyzing_agent": "CoordinatorAgent"
        },
        "content_analysis": {
            "source_content": {
                "original_text": "Sample election-related content for analysis",
                "source_platform": "Twitter",
                "language_detected": "English"
            },
            "narrative_classification": {
                "primary_theme": "voter_intimidation",
                "threat_level": "MEDIUM",
                "classification_confidence": 0.75
            }
        }
    }
    
    return ElectionWatchReportTemplate.create_report_with_template("comprehensive", **sample_data)

# Template instruction for agents
AGENT_TEMPLATE_INSTRUCTION = """
When generating analysis reports, use the ElectionWatchReportTemplate class to ensure consistent structure.

Examples:
1. For comprehensive analysis: ElectionWatchReportTemplate.get_comprehensive_analysis_template()
2. For quick analysis: ElectionWatchReportTemplate.get_quick_analysis_template()
3. For multimedia: ElectionWatchReportTemplate.get_multimedia_analysis_template()
4. For trend monitoring: ElectionWatchReportTemplate.get_trend_monitoring_template()

Always populate the following key fields:
- report_metadata (with unique report_id and timestamp)
- narrative_classification (with theme, threat_level, confidence)
- actors_identified (with name, role, affiliation)
- lexicon_analysis (with detected terms and categories)
- risk_assessment (with overall risk score and recommendations)

Use the create_report_with_template() method to populate templates with your analysis data.
Export final reports using export_report_as_json() for persistent storage.
""" 

# Tool functions for coordinator access
from google.adk.tools import FunctionTool

def get_comprehensive_analysis_template() -> Dict[str, Any]:
    """Get comprehensive analysis template for detailed election monitoring reports"""
    return ElectionWatchReportTemplate.get_comprehensive_analysis_template()

def get_quick_analysis_template() -> Dict[str, Any]:
    """Get quick analysis template for rapid assessments"""
    return ElectionWatchReportTemplate.get_quick_analysis_template()

def get_multimedia_analysis_template() -> Dict[str, Any]:
    """Get multimedia analysis template for image/video content analysis"""
    return ElectionWatchReportTemplate.get_multimedia_analysis_template()

def get_trend_monitoring_template() -> Dict[str, Any]:
    """Get trend monitoring template for tracking narrative patterns"""
    return ElectionWatchReportTemplate.get_trend_monitoring_template()

def export_report_json(report_data: Dict[str, Any], filename: Optional[str] = None) -> Dict[str, Any]:
    """Export analysis report as JSON file"""
    return ElectionWatchReportTemplate.export_report_as_json(report_data, filename)

# Create FunctionTool instances for coordinator use
get_comprehensive_analysis_template_tool = FunctionTool(
    func=get_comprehensive_analysis_template
)

get_quick_analysis_template_tool = FunctionTool(
    func=get_quick_analysis_template
)

get_multimedia_analysis_template_tool = FunctionTool(
    func=get_multimedia_analysis_template
)

get_trend_monitoring_template_tool = FunctionTool(
    func=get_trend_monitoring_template
)

export_report_json_tool = FunctionTool(
    func=export_report_json
) 