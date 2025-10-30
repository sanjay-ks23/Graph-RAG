"""Gemma 3n E2B-it model integration"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import List, Dict
from src.utils.logger import setup_logger
from src.utils.memory_utils import with_memory_cleanup, clear_cuda_cache

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
        
        clear_cuda_cache()
        logger.info(f"Loading Gemma model: {model_id} on {self.device}")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        
        # Load model with proper device handling
        if self.device == "cuda":
            try:
                # Try 8-bit quantization first (requires bitsandbytes)
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_id,
                    load_in_8bit=True,
                    device_map="auto",
                    low_cpu_mem_usage=True
                )
                logger.info("Loaded model with 8-bit quantization")
            except Exception as e:
                logger.warning(f"8-bit loading failed: {e}. Falling back to float16")
                # Fallback to float16 with memory limit
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_id,
                    dtype=torch.float16,
                    device_map="auto",
                    low_cpu_mem_usage=True,
                    max_memory={0: "6GB", "cpu": "8GB"}
                )
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_id,
                dtype=torch.float32,
                low_cpu_mem_usage=True
            )
            self.model = self.model.to(self.device)
        
        self.model.eval()
        logger.info("Gemma model loaded successfully")
    
    @with_memory_cleanup
    def generate_response(self, messages: List[Dict[str, str]], 
                         max_new_tokens: int = 512) -> str:
        """Generate response using chat format"""
        
        # Format messages for Gemma
        formatted_prompt = self._format_chat(messages)
        
        # Tokenize
        inputs = self.tokenizer(
            formatted_prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length - max_new_tokens
        ).to(self.device)
        
        # Generate with memory-efficient settings
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=self.temperature,
                top_p=self.top_p,
                top_k=self.top_k,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                use_cache=True  # Enable KV cache for memory efficiency
            )
        
        # Decode
        response = self.tokenizer.decode(
            outputs[0][inputs['input_ids'].shape[1]:],
            skip_special_tokens=True
        )
        
        del inputs, outputs
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
