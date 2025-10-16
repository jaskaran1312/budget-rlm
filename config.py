"""
Configuration management for RLM Document Analyzer
Provides robust configuration handling with validation and defaults.
"""

import os
import json
import logging
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, asdict
from pathlib import Path

# Try to import python-dotenv, but don't fail if it's not available
try:
    from dotenv import load_dotenv
    DOTENV_AVAILABLE = True
except ImportError:
    DOTENV_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class RLMConfig:
    """Configuration for the RLM system."""
    # Core RLM settings
    max_iterations: int = 10
    max_code_executions: int = 20
    timeout_seconds: int = 300
    enable_recursive_calls: bool = True
    max_recursion_depth: int = 3
    
    # Model settings
    model_name: str = "gemini-2.5-flash"
    api_key: Optional[str] = None
    
    # REPL settings
    max_output_length: int = 10000
    enable_file_operations: bool = False
    allowed_imports: List[str] = None
    
    # Document processing settings
    max_document_size: int = 10 * 1024 * 1024  # 10MB
    supported_extensions: List[str] = None
    encoding_fallback: List[str] = None
    
    # Logging settings
    log_level: str = "DEBUG"
    log_file: Optional[str] = None
    
    # Performance settings
    cache_analysis_results: bool = True
    max_cache_size: int = 1000
    
    def __post_init__(self):
        """Set default values for list fields."""
        if self.allowed_imports is None:
            self.allowed_imports = [
                're', 'json', 'math', 'random', 'datetime', 
                'collections', 'itertools', 'functools'
            ]
        
        if self.supported_extensions is None:
            self.supported_extensions = [
                '.txt', '.md', '.py', '.js', '.ts', '.html', '.css', 
                '.json', '.xml', '.csv', '.log', '.sql', '.yaml', 
                '.yml', '.ini', '.cfg', '.rst', '.tex'
            ]
        
        if self.encoding_fallback is None:
            self.encoding_fallback = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']


class ConfigManager:
    """Manages configuration loading, validation, and saving."""
    
    def __init__(self, config_file: Optional[str] = None):
        self.config_file = config_file or "rlm_config.json"
        self.config = RLMConfig()
        self._load_env_file()
        self._load_config()
    
    def _load_env_file(self):
        """Load environment variables from .env file if it exists."""
        if DOTENV_AVAILABLE:
            # Look for .env file in current directory and parent directories
            env_paths = ['.env', '../.env', '../../.env']
            
            for env_path in env_paths:
                if Path(env_path).exists():
                    load_dotenv(env_path)
                    logger.info(f"Loaded environment variables from {env_path}")
                    break
            else:
                logger.debug("No .env file found, using system environment variables")
        else:
            logger.warning("python-dotenv not available, .env files will not be loaded")
    
    def _load_config(self):
        """Load configuration from file and environment variables."""
        # Load from file if it exists
        if Path(self.config_file).exists():
            try:
                with open(self.config_file, 'r') as f:
                    file_config = json.load(f)
                
                # Update config with file values
                for key, value in file_config.items():
                    if hasattr(self.config, key):
                        setattr(self.config, key, value)
                
                logger.info(f"Loaded configuration from {self.config_file}")
            except Exception as e:
                logger.warning(f"Failed to load config file {self.config_file}: {e}")
        
        # Override with environment variables
        self._load_from_env()
    
    def _load_from_env(self):
        """Load configuration from environment variables."""
        env_mappings = {
            'GEMINI_API_KEY': 'api_key',
            'RLM_MAX_ITERATIONS': ('max_iterations', int),
            'RLM_MAX_RECURSION_DEPTH': ('max_recursion_depth', int),
            'RLM_MODEL_NAME': 'model_name',
            'RLM_LOG_LEVEL': 'log_level',
            'RLM_LOG_FILE': 'log_file',
            'RLM_MAX_DOCUMENT_SIZE': ('max_document_size', int),
            'RLM_TIMEOUT_SECONDS': ('timeout_seconds', int),
        }
        
        for env_var, config_attr in env_mappings.items():
            value = os.environ.get(env_var)
            if value is not None:
                if isinstance(config_attr, tuple):
                    attr_name, type_func = config_attr
                    try:
                        value = type_func(value)
                    except ValueError:
                        logger.warning(f"Invalid value for {env_var}: {value}")
                        continue
                else:
                    attr_name = config_attr
                
                setattr(self.config, attr_name, value)
                logger.debug(f"Set {attr_name} from environment variable {env_var}")
    
    def save_config(self, file_path: Optional[str] = None):
        """Save current configuration to file."""
        save_path = file_path or self.config_file
        
        try:
            config_dict = asdict(self.config)
            # Remove None values
            config_dict = {k: v for k, v in config_dict.items() if v is not None}
            
            with open(save_path, 'w') as f:
                json.dump(config_dict, f, indent=2)
            
            logger.info(f"Configuration saved to {save_path}")
        except Exception as e:
            logger.error(f"Failed to save configuration: {e}")
    
    def validate_config(self) -> List[str]:
        """Validate the current configuration and return any issues."""
        issues = []
        
        # Check required settings
        if not self.config.api_key:
            issues.append("API key is required (set GEMINI_API_KEY environment variable)")
        
        # Check numeric ranges
        if self.config.max_iterations < 1:
            issues.append("max_iterations must be at least 1")
        
        if self.config.max_recursion_depth < 0:
            issues.append("max_recursion_depth must be non-negative")
        
        if self.config.timeout_seconds < 1:
            issues.append("timeout_seconds must be at least 1")
        
        if self.config.max_document_size < 1024:
            issues.append("max_document_size must be at least 1KB")
        
        # Check log level
        valid_log_levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
        if self.config.log_level.upper() not in valid_log_levels:
            issues.append(f"log_level must be one of: {valid_log_levels}")
        
        return issues
    
    def get_config(self) -> RLMConfig:
        """Get the current configuration."""
        return self.config
    
    def update_config(self, **kwargs):
        """Update configuration with new values."""
        for key, value in kwargs.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
                logger.debug(f"Updated config: {key} = {value}")
            else:
                logger.warning(f"Unknown configuration key: {key}")


def setup_logging(config: RLMConfig):
    """Set up logging based on configuration."""
    log_level = getattr(logging, config.log_level.upper(), logging.INFO)
    
    # Configure root logger
    handlers = [logging.StreamHandler()]
    
    if config.log_file:
        handlers.append(logging.FileHandler(config.log_file))
    
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=handlers
    )
    
    logger.info(f"Logging configured: level={config.log_level}, file={config.log_file}")


def create_default_config_file(file_path: str = "rlm_config.json"):
    """Create a default configuration file."""
    config = RLMConfig()
    config_dict = asdict(config)
    
    # Remove None values and add comments
    config_dict = {k: v for k, v in config_dict.items() if v is not None}
    
    with open(file_path, 'w') as f:
        json.dump(config_dict, f, indent=2)
    
    logger.info(f"Created default configuration file: {file_path}")


# Global config manager instance
config_manager = ConfigManager()


def get_config() -> RLMConfig:
    """Get the global configuration instance."""
    return config_manager.get_config()


def update_config(**kwargs):
    """Update the global configuration."""
    config_manager.update_config(**kwargs)


def validate_and_setup():
    """Validate configuration and set up logging."""
    issues = config_manager.validate_config()
    
    if issues:
        logger.error("Configuration validation failed:")
        for issue in issues:
            logger.error(f"  - {issue}")
        return False
    
    setup_logging(config_manager.get_config())
    return True
