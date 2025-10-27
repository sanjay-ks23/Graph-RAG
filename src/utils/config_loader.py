"""Configuration loader and manager"""

import yaml
import os
from pathlib import Path
from typing import Dict, Any
from dotenv import load_dotenv

class ConfigLoader:
    """Load and manage configuration"""
    
    def __init__(self, config_path: str = "config.yaml"):
        load_dotenv()
        self.config_path = config_path
        self.config = self._load_config()
        self._setup_directories()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file"""
        with open(self.config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Override with environment variables if present
        if os.getenv('CUDA_VISIBLE_DEVICES'):
            config['models']['llm']['device'] = 'cuda'
            config['models']['embedding']['device'] = 'cuda'
        
        return config
    
    def _setup_directories(self):
        """Create necessary directories"""
        dirs = [
            self.config['paths']['data_dir'],
            self.config['paths']['models_dir'],
            self.config['paths']['logs_dir'],
            self.config['paths']['cache_dir'],
            self.config['data']['books_directory']
        ]
        
        for dir_path in dirs:
            Path(dir_path).mkdir(parents=True, exist_ok=True)
    
    def get(self, key_path: str, default=None):
        """Get configuration value using dot notation"""
        keys = key_path.split('.')
        value = self.config
        
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default
        
        return value
    
    def __getitem__(self, key):
        return self.config[key]
