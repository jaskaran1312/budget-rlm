"""
Utility functions for RLM Document Analyzer
Provides common utilities for error handling, validation, and data processing.
"""

import os
import sys
import time
import hashlib
import logging
from typing import Any, Dict, List, Optional, Union, Callable
from functools import wraps
from pathlib import Path
import json
import re

logger = logging.getLogger(__name__)


def retry_on_failure(max_retries: int = 3, delay: float = 1.0, backoff: float = 2.0):
    """
    Decorator to retry a function on failure.
    
    Args:
        max_retries: Maximum number of retry attempts
        delay: Initial delay between retries in seconds
        backoff: Multiplier for delay after each retry
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            current_delay = delay
            last_exception = None
            
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    if attempt < max_retries:
                        logger.warning(f"Attempt {attempt + 1} failed for {func.__name__}: {e}")
                        logger.info(f"Retrying in {current_delay} seconds...")
                        time.sleep(current_delay)
                        current_delay *= backoff
                    else:
                        logger.error(f"All {max_retries + 1} attempts failed for {func.__name__}")
            
            raise last_exception
        
        return wrapper
    return decorator


def safe_execute(func: Callable, *args, **kwargs) -> tuple[bool, Any, Optional[str]]:
    """
    Safely execute a function and return success status, result, and error.
    
    Args:
        func: Function to execute
        *args: Function arguments
        **kwargs: Function keyword arguments
        
    Returns:
        Tuple of (success, result, error_message)
    """
    try:
        result = func(*args, **kwargs)
        return True, result, None
    except Exception as e:
        error_msg = f"{type(e).__name__}: {str(e)}"
        logger.error(f"Error in {func.__name__}: {error_msg}")
        return False, None, error_msg


def validate_file_path(file_path: Union[str, Path], must_exist: bool = True) -> bool:
    """
    Validate a file path.
    
    Args:
        file_path: Path to validate
        must_exist: Whether the file must exist
        
    Returns:
        True if valid, False otherwise
    """
    try:
        path = Path(file_path)
        
        if must_exist and not path.exists():
            logger.error(f"File does not exist: {file_path}")
            return False
        
        if must_exist and not path.is_file():
            logger.error(f"Path is not a file: {file_path}")
            return False
        
        return True
    except Exception as e:
        logger.error(f"Invalid file path {file_path}: {e}")
        return False


def validate_directory_path(dir_path: Union[str, Path], must_exist: bool = True) -> bool:
    """
    Validate a directory path.
    
    Args:
        dir_path: Path to validate
        must_exist: Whether the directory must exist
        
    Returns:
        True if valid, False otherwise
    """
    try:
        path = Path(dir_path)
        
        if must_exist and not path.exists():
            logger.error(f"Directory does not exist: {dir_path}")
            return False
        
        if must_exist and not path.is_dir():
            logger.error(f"Path is not a directory: {dir_path}")
            return False
        
        return True
    except Exception as e:
        logger.error(f"Invalid directory path {dir_path}: {e}")
        return False


def calculate_file_hash(file_path: Union[str, Path], algorithm: str = 'md5') -> Optional[str]:
    """
    Calculate hash of a file.
    
    Args:
        file_path: Path to the file
        algorithm: Hash algorithm to use ('md5', 'sha1', 'sha256')
        
    Returns:
        Hash string or None if error
    """
    try:
        path = Path(file_path)
        if not path.exists() or not path.is_file():
            return None
        
        hash_obj = hashlib.new(algorithm)
        
        with open(path, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_obj.update(chunk)
        
        return hash_obj.hexdigest()
    except Exception as e:
        logger.error(f"Error calculating hash for {file_path}: {e}")
        return None


def sanitize_filename(filename: str) -> str:
    """
    Sanitize a filename by removing or replacing invalid characters.
    
    Args:
        filename: Original filename
        
    Returns:
        Sanitized filename
    """
    # Remove or replace invalid characters
    sanitized = re.sub(r'[<>:"/\\|?*]', '_', filename)
    
    # Remove leading/trailing spaces and dots
    sanitized = sanitized.strip(' .')
    
    # Ensure it's not empty
    if not sanitized:
        sanitized = "unnamed_file"
    
    # Limit length
    if len(sanitized) > 255:
        name, ext = os.path.splitext(sanitized)
        sanitized = name[:255-len(ext)] + ext
    
    return sanitized


def format_file_size(size_bytes: int) -> str:
    """
    Format file size in human-readable format.
    
    Args:
        size_bytes: Size in bytes
        
    Returns:
        Formatted size string
    """
    if size_bytes == 0:
        return "0 B"
    
    size_names = ["B", "KB", "MB", "GB", "TB"]
    i = 0
    size = float(size_bytes)
    
    while size >= 1024.0 and i < len(size_names) - 1:
        size /= 1024.0
        i += 1
    
    return f"{size:.1f} {size_names[i]}"


def format_duration(seconds: float) -> str:
    """
    Format duration in human-readable format.
    
    Args:
        seconds: Duration in seconds
        
    Returns:
        Formatted duration string
    """
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f}m"
    else:
        hours = seconds / 3600
        return f"{hours:.1f}h"


def truncate_text(text: str, max_length: int = 100, suffix: str = "...") -> str:
    """
    Truncate text to a maximum length.
    
    Args:
        text: Text to truncate
        max_length: Maximum length
        suffix: Suffix to add if truncated
        
    Returns:
        Truncated text
    """
    if len(text) <= max_length:
        return text
    
    return text[:max_length - len(suffix)] + suffix


def extract_text_snippet(text: str, pattern: str, context_lines: int = 3) -> List[str]:
    """
    Extract snippets of text around pattern matches.
    
    Args:
        text: Text to search in
        pattern: Regex pattern to search for
        context_lines: Number of context lines around matches
        
    Returns:
        List of text snippets
    """
    lines = text.split('\n')
    snippets = []
    
    for i, line in enumerate(lines):
        if re.search(pattern, line, re.IGNORECASE):
            start = max(0, i - context_lines)
            end = min(len(lines), i + context_lines + 1)
            snippet = '\n'.join(lines[start:end])
            snippets.append(snippet)
    
    return snippets


def merge_dictionaries(*dicts: Dict[str, Any]) -> Dict[str, Any]:
    """
    Merge multiple dictionaries, with later ones taking precedence.
    
    Args:
        *dicts: Dictionaries to merge
        
    Returns:
        Merged dictionary
    """
    result = {}
    for d in dicts:
        if d:
            result.update(d)
    return result


def deep_merge_dicts(dict1: Dict[str, Any], dict2: Dict[str, Any]) -> Dict[str, Any]:
    """
    Deep merge two dictionaries.
    
    Args:
        dict1: First dictionary
        dict2: Second dictionary
        
    Returns:
        Deep merged dictionary
    """
    result = dict1.copy()
    
    for key, value in dict2.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge_dicts(result[key], value)
        else:
            result[key] = value
    
    return result


def ensure_directory_exists(dir_path: Union[str, Path]) -> bool:
    """
    Ensure a directory exists, creating it if necessary.
    
    Args:
        dir_path: Path to the directory
        
    Returns:
        True if directory exists or was created, False otherwise
    """
    try:
        path = Path(dir_path)
        path.mkdir(parents=True, exist_ok=True)
        return True
    except Exception as e:
        logger.error(f"Failed to create directory {dir_path}: {e}")
        return False


def get_system_info() -> Dict[str, Any]:
    """
    Get system information.
    
    Returns:
        Dictionary with system information
    """
    import platform
    import psutil
    
    try:
        return {
            'platform': platform.platform(),
            'python_version': platform.python_version(),
            'architecture': platform.architecture(),
            'processor': platform.processor(),
            'memory_total': psutil.virtual_memory().total,
            'memory_available': psutil.virtual_memory().available,
            'cpu_count': psutil.cpu_count(),
            'disk_usage': psutil.disk_usage('/').percent if os.name != 'nt' else psutil.disk_usage('C:').percent
        }
    except ImportError:
        # psutil not available
        return {
            'platform': platform.platform(),
            'python_version': platform.python_version(),
            'architecture': platform.architecture(),
            'processor': platform.processor(),
        }


def create_progress_bar(current: int, total: int, width: int = 50) -> str:
    """
    Create a text-based progress bar.
    
    Args:
        current: Current progress
        total: Total items
        width: Width of the progress bar
        
    Returns:
        Progress bar string
    """
    if total == 0:
        return "[" + " " * width + "] 0/0 (0%)"
    
    progress = current / total
    filled = int(width * progress)
    bar = "█" * filled + "░" * (width - filled)
    
    return f"[{bar}] {current}/{total} ({progress:.1%})"


class Timer:
    """Context manager for timing operations."""
    
    def __init__(self, description: str = "Operation"):
        self.description = description
        self.start_time = None
        self.end_time = None
    
    def __enter__(self):
        self.start_time = time.time()
        logger.info(f"Starting {self.description}")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = time.time()
        duration = self.end_time - self.start_time
        logger.info(f"Completed {self.description} in {format_duration(duration)}")
    
    @property
    def elapsed(self) -> float:
        """Get elapsed time in seconds."""
        if self.start_time is None:
            return 0.0
        end = self.end_time or time.time()
        return end - self.start_time


def validate_json_schema(data: Any, schema: Dict[str, Any]) -> List[str]:
    """
    Simple JSON schema validation.
    
    Args:
        data: Data to validate
        schema: Schema definition
        
    Returns:
        List of validation errors
    """
    errors = []
    
    for key, expected_type in schema.items():
        if key not in data:
            errors.append(f"Missing required field: {key}")
        elif not isinstance(data[key], expected_type):
            errors.append(f"Field '{key}' must be of type {expected_type.__name__}")
    
    return errors


def chunk_list(lst: List[Any], chunk_size: int) -> List[List[Any]]:
    """
    Split a list into chunks of specified size.
    
    Args:
        lst: List to chunk
        chunk_size: Size of each chunk
        
    Returns:
        List of chunks
    """
    return [lst[i:i + chunk_size] for i in range(0, len(lst), chunk_size)]


def flatten_list(nested_list: List[Any]) -> List[Any]:
    """
    Flatten a nested list.
    
    Args:
        nested_list: List that may contain nested lists
        
    Returns:
        Flattened list
    """
    result = []
    for item in nested_list:
        if isinstance(item, list):
            result.extend(flatten_list(item))
        else:
            result.append(item)
    return result
