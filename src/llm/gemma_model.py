"""Gemma 3n E2B-it model integration"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import List, Dict, Any
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

class GemmaModel:
    """Gemma 3n E2B-it model for therapeutic conversations"""
    
    def __init__(self, model_id: str = "google/gemma-3n-e2b-it",
                 device: str = "cuda", max_length: int = 8192,
                 temperature: float = 0.7, top_p: float = 0.9, top_k: int = 40):
        self.model_id = model_id
        self.device = device if torch.cuda.is_available() else "cpu"
        self.max_length = max_length
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        
        # Clear CUDA cache before loading model
        if self.device == "cuda" and torch.cuda.is_available():
            torch.cuda.empty_cache()
            logger.info("Cleared CUDA cache before loading Gemma model")
        
        logger.info(f"Loading Gemma model: {model_id} on {self.device}")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        
        # Load model
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            device_map="auto" if self.device == "cuda" else None,
            low_cpu_mem_usage=True
        )
        
        if self.device == "cpu":
            self.model = self.model.to(self.device)
        
        self.model.eval()
        logger.info("Gemma model loaded successfully")
    
    def generate_response(self, messages: List[Dict[str, str]], 
                         max_new_tokens: int = 512) -> str:
        """Generate response using chat format"""
        
        # Clear CUDA cache before generation
        if self.device == "cuda" and torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Format messages for Gemma
        formatted_prompt = self._format_chat(messages)
        
        # Tokenize
        inputs = self.tokenizer(
            formatted_prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length - max_new_tokens
        ).to(self.device)
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=self.temperature,
                top_p=self.top_p,
                top_k=self.top_k,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
        
        # Decode
        response = self.tokenizer.decode(
            outputs[0][inputs['input_ids'].shape[1]:],
            skip_special_tokens=True
        )
        
        # Clear cache after generation
        if self.device == "cuda" and torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        return response.strip()
    
    def _format_chat(self, messages: List[Dict[str, str]]) -> str:
        """Format messages for Gemma chat template"""
        formatted = ""
        
        for message in messages:
            role = message['role']
            content = message['content']
            
            if role == 'system':
                formatted += f"<start_of_turn>user\n{content}<end_of_turn>\n"
            elif role == 'user':
                formatted += f"<start_of_turn>user\n{content}<end_of_turn>\n"
            elif role == 'assistant':
                formatted += f"<start_of_turn>model\n{content}<end_of_turn>\n"
        
        # Add model turn
        formatted += "<start_of_turn>model\n"
        
        return formatted
    
    def count_tokens(self, text: str) -> int:
        """Count tokens in text"""
        return len(self.tokenizer.encode(text))
    
    def get_max_context_length(self) -> int:
        """Get maximum context length"""
        return self.max_length
