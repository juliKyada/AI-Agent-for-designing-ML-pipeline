"""
Utility functions for configuration management
"""
import yaml
from pathlib import Path
from typing import Dict, Any
from dotenv import load_dotenv

# Load environment variables at the module level
load_dotenv()


class Config:
    """Configuration manager for MetaFlow"""
    
    def __init__(self, config_path: str = None):
        """
        Initialize configuration
        
        Args:
            config_path: Path to configuration file
        """
        if config_path is None:
            config_path = Path(__file__).parent.parent.parent / "config" / "config.yaml"
        
        self.config_path = Path(config_path)
        self._config = self._load_config()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file"""
        if not self.config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")
        
        with open(self.config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        return config
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value by key (supports nested keys with dot notation)
        
        Args:
            key: Configuration key (e.g., 'pipeline.max_iterations')
            default: Default value if key not found
            
        Returns:
            Configuration value
        """
        keys = key.split('.')
        value = self._config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value
    
    def __getitem__(self, key: str) -> Any:
        """Get configuration value using dictionary notation"""
        return self.get(key)
    
    def to_dict(self) -> Dict[str, Any]:
        """Return full configuration as dictionary"""
        return self._config.copy()


# Global configuration instance
_config = None

def get_config(config_path: str = None) -> Config:
    """
    Get global configuration instance
    
    Args:
        config_path: Path to configuration file (only used for first call)
        
    Returns:
        Config instance
    """
    global _config
    if _config is None:
        _config = Config(config_path)
    return _config
