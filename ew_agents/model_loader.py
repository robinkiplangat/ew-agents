#!/usr/bin/env python3
"""
Model Loader for Fine-tuned DISARM Model
========================================

Handles loading and configuration of the fine-tuned DISARM model for OSINT analysis.
"""

import logging
import os
from typing import Optional, Dict, Any
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

logger = logging.getLogger(__name__)

class DISARMModelLoader:
    """Loader for the fine-tuned DISARM model."""
    
    def __init__(self, model_name: str = "fourbic/disarm-ew-llama3-finetuned"):
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
    def load_model(self) -> bool:
        """Load the fine-tuned DISARM model."""
        try:
            logger.info(f"🔄 Loading fine-tuned DISARM model: {self.model_name}")
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                trust_remote_code=True,
                use_fast=False,
                token=os.getenv("HUGGING_FACE_HUB_TOKEN")  # Add token for gated models
            )
            
            # Load model with quantization for efficiency
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True,
                load_in_8bit=True if self.device == "cuda" else False,
                token=os.getenv("HUGGING_FACE_HUB_TOKEN")  # Add token for gated models
            )
            
            logger.info(f"✅ Fine-tuned DISARM model loaded successfully on {self.device}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to load fine-tuned model: {e}")
            return False
    
    def generate_response(self, prompt: str, max_length: int = 512) -> str:
        """Generate response using the fine-tuned model."""
        if not self.model or not self.tokenizer:
            logger.warning("⚠️ Model not loaded, cannot generate response")
            return ""
        
        try:
            # Tokenize input
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            
            # Generate response
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_length=max_length,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            # Decode response
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Remove the input prompt from response
            response = response[len(prompt):].strip()
            
            return response
            
        except Exception as e:
            logger.error(f"❌ Error generating response: {e}")
            return ""
    
    def is_available(self) -> bool:
        """Check if the fine-tuned model is available."""
        return self.model is not None and self.tokenizer is not None

# Global model loader instance
disarm_model_loader = DISARMModelLoader()

def get_disarm_model() -> Optional[DISARMModelLoader]:
    """Get the DISARM model loader instance."""
    logger.info("🔍 Attempting to get DISARM model...")
    try:
        if not disarm_model_loader.is_available():
            logger.info("🔄 Model not available, attempting to load...")
            success = disarm_model_loader.load_model()
            logger.info(f"📊 Model loading result: {success}")
        else:
            logger.info("✅ Model already available")
        
        if disarm_model_loader.is_available():
            logger.info("✅ DISARM model is available and ready")
            return disarm_model_loader
        else:
            logger.warning("⚠️ DISARM model is not available")
            return None
    except Exception as e:
        logger.error(f"❌ Error getting DISARM model: {e}")
        return None 