"""
Python REPL Environment for Recursive Language Models
Implements a secure Python execution environment that can store variables
and execute code blocks, similar to a Jupyter notebook environment.
"""

import ast
import sys
import io
import traceback
import types
import builtins
from typing import Dict, Any, List, Optional, Tuple
import re
import json
import logging
from text_toolkit import TextProcessor, DocumentTextProcessor

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


class SecureREPL:
    """
    A secure Python REPL environment that can execute code and store variables.
    Designed to work with Recursive Language Models for document analysis.
    """
    
    def __init__(self, max_output_length: int = 10000):
        self.max_output_length = max_output_length
        self.namespace = {}
        self.execution_history = []
        self.output_buffer = io.StringIO()
        
        # Initialize with safe builtins
        self.namespace.update({
            '__builtins__': {
                'print': self._safe_print,
                'len': len,
                'str': str,
                'int': int,
                'float': float,
                'bool': bool,
                'list': list,
                'dict': dict,
                'tuple': tuple,
                'set': set,
                'range': range,
                'enumerate': enumerate,
                'zip': zip,
                'map': map,
                'filter': filter,
                'sum': sum,
                'max': max,
                'min': min,
                'abs': abs,
                'round': round,
                'sorted': sorted,
                'reversed': reversed,
                'any': any,
                'all': all,
                'isinstance': isinstance,
                'type': type,
                'hasattr': hasattr,
                'getattr': getattr,
                'setattr': setattr,
                'dir': dir,
                'open': self._safe_open,
                're': re,
                'json': json,
            }
        })
        
        # Add common modules
        self.namespace.update({
            're': re,
            'json': json,
            'sys': sys,
            'collections': __import__('collections'),
            'itertools': __import__('itertools'),
            'functools': __import__('functools'),
        })
    
    def _safe_print(self, *args, **kwargs):
        """Safe print function that captures output."""
        print(*args, file=self.output_buffer, **kwargs)
    
    def _safe_open(self, *args, **kwargs):
        """Safe open function - disabled for security."""
        raise PermissionError("File operations are disabled in this REPL environment")
    
    def execute_code(self, code: str) -> Tuple[bool, str, str]:
        """
        Execute Python code in the REPL environment.
        
        Args:
            code: Python code to execute
            
        Returns:
            Tuple of (success, output, error)
        """
        logger.info(f"Executing code: {len(code)} characters")
        logger.debug(f"Code preview: {code[:200]}...")
        
        # Clear output buffer
        self.output_buffer = io.StringIO()
        
        try:
            # Parse and compile the code
            logger.debug("Parsing and compiling code...")
            tree = ast.parse(code, mode='exec')
            
            # Check for dangerous operations
            logger.debug("Checking AST safety...")
            self._check_ast_safety(tree)
            
            # Execute the code
            logger.debug("Executing compiled code...")
            exec(compile(tree, '<repl>', 'exec'), self.namespace)
            
            # Get output
            output = self.output_buffer.getvalue()
            logger.debug(f"Code execution completed, output length: {len(output)}")
            
            # Truncate output if too long
            if len(output) > self.max_output_length:
                output = output[:self.max_output_length] + "\n... (output truncated)"
                logger.warning(f"Output truncated to {self.max_output_length} characters")
            
            # Record execution
            self.execution_history.append({
                'code': code,
                'success': True,
                'output': output,
                'error': None
            })
            
            logger.info("Code execution successful")
            return True, output, ""
            
        except Exception as e:
            error_msg = f"{type(e).__name__}: {str(e)}"
            if hasattr(e, '__traceback__'):
                error_msg += f"\n{traceback.format_exc()}"
            
            logger.error(f"Code execution failed: {error_msg}")
            
            # Record failed execution
            self.execution_history.append({
                'code': code,
                'success': False,
                'output': "",
                'error': error_msg
            })
            
            return False, "", error_msg
    
    def _check_ast_safety(self, tree: ast.AST):
        """Check AST for potentially dangerous operations."""
        logger.debug("Checking AST for dangerous operations...")
        dangerous_nodes = (
            ast.Import, ast.ImportFrom, ast.Call
        )
        
        for node in ast.walk(tree):
            if isinstance(node, dangerous_nodes):
                if isinstance(node, ast.Call):
                    # Check for dangerous function calls
                    if isinstance(node.func, ast.Name):
                        if node.func.id in ['__import__', 'eval', 'exec', 'compile']:
                            logger.error(f"Dangerous function call detected: {node.func.id}")
                            raise PermissionError(f"Dangerous function call: {node.func.id}")
                elif isinstance(node, (ast.Import, ast.ImportFrom)):
                    # Allow only safe imports
                    if isinstance(node, ast.Import):
                        for alias in node.names:
                            if alias.name not in ['re', 'json', 'math', 'random', 'datetime']:
                                logger.error(f"Import not allowed: {alias.name}")
                                raise PermissionError(f"Import not allowed: {alias.name}")
                    elif isinstance(node, ast.ImportFrom):
                        if node.module not in ['re', 'json', 'math', 'random', 'datetime']:
                            logger.error(f"Import from not allowed: {node.module}")
                            raise PermissionError(f"Import from not allowed: {node.module}")
        
        logger.debug("AST safety check passed")
    
    def get_variable(self, name: str) -> Any:
        """Get a variable from the namespace."""
        return self.namespace.get(name)
    
    def set_variable(self, name: str, value: Any):
        """Set a variable in the namespace."""
        self.namespace[name] = value
    
    def list_variables(self) -> Dict[str, str]:
        """List all variables in the namespace with their types."""
        variables = {}
        for name, value in self.namespace.items():
            if not name.startswith('__'):
                variables[name] = f"{type(value).__name__}: {str(value)[:100]}..."
        return variables
    
    def get_execution_history(self) -> List[Dict]:
        """Get the execution history."""
        return self.execution_history
    
    def clear_history(self):
        """Clear execution history."""
        self.execution_history = []
    
    def reset_namespace(self):
        """Reset the namespace to initial state."""
        self.__init__(self.max_output_length)


class DocumentAnalyzer:
    """
    Document analysis utilities for the REPL environment.
    Provides tools for analyzing large collections of documents.
    """
    
    def __init__(self, repl: SecureREPL):
        self.repl = repl
        self.documents = {}
        self.document_metadata = {}
        self.text_processor = None
        self.document_processor = None
    
    def load_documents(self, documents: Dict[str, str], metadata: Optional[Dict[str, Dict]] = None):
        """
        Load documents into the REPL environment.
        
        Args:
            documents: Dictionary of {doc_id: content}
            metadata: Optional metadata for each document
        """
        logger.info(f"Loading {len(documents)} documents into REPL environment")
        logger.debug(f"Document IDs: {list(documents.keys())[:10]}...")  # Show first 10 IDs
        
        self.documents = documents
        self.document_metadata = metadata or {}
        
        # Initialize text processors
        self.text_processor = TextProcessor()
        self.document_processor = DocumentTextProcessor(self.documents)
        
        # Store in REPL
        logger.debug("Setting variables in REPL namespace...")
        self.repl.set_variable('documents', self.documents)
        self.repl.set_variable('document_metadata', self.document_metadata)
        self.repl.set_variable('num_documents', len(documents))
        self.repl.set_variable('text_processor', self.text_processor)
        self.repl.set_variable('document_processor', self.document_processor)
        
        logger.info(f"Successfully loaded {len(documents)} documents into REPL environment")
    
    def search_documents(self, pattern: str, case_sensitive: bool = False) -> Dict[str, List[str]]:
        """
        Search for patterns across all documents.
        
        Args:
            pattern: Regex pattern to search for
            case_sensitive: Whether search should be case sensitive
            
        Returns:
            Dictionary of {doc_id: [matching_lines]}
        """
        flags = 0 if case_sensitive else re.IGNORECASE
        results = {}
        
        for doc_id, content in self.documents.items():
            lines = content.split('\n')
            matches = []
            
            for i, line in enumerate(lines):
                if re.search(pattern, line, flags):
                    matches.append(f"Line {i+1}: {line.strip()}")
            
            if matches:
                results[doc_id] = matches
        
        return results
    
    def analyze_document_stats(self) -> Dict[str, Any]:
        """Analyze basic statistics of all documents."""
        stats = {
            'total_documents': len(self.documents),
            'total_characters': sum(len(content) for content in self.documents.values()),
            'total_words': sum(len(content.split()) for content in self.documents.values()),
            'total_lines': sum(len(content.split('\n')) for content in self.documents.values()),
            'avg_doc_length': 0,
            'doc_lengths': {}
        }
        
        if self.documents:
            stats['avg_doc_length'] = stats['total_characters'] / len(self.documents)
            stats['doc_lengths'] = {doc_id: len(content) for doc_id, content in self.documents.items()}
        
        return stats
    
    def get_document_sample(self, doc_id: str, start_line: int = 0, num_lines: int = 10) -> str:
        """Get a sample of lines from a specific document."""
        if doc_id not in self.documents:
            return f"Document {doc_id} not found"
        
        lines = self.documents[doc_id].split('\n')
        end_line = min(start_line + num_lines, len(lines))
        
        return '\n'.join(lines[start_line:end_line])


def create_repl_environment() -> Tuple[SecureREPL, DocumentAnalyzer]:
    """Create and initialize a REPL environment with document analysis capabilities."""
    repl = SecureREPL()
    analyzer = DocumentAnalyzer(repl)
    
    # Add some utility functions to the REPL
    utility_code = """
def grep(pattern, text, case_sensitive=False):
    '''Search for pattern in text using regex.'''
    flags = 0 if case_sensitive else re.IGNORECASE
    matches = re.findall(pattern, text, flags)
    return matches

def count_occurrences(pattern, text, case_sensitive=False):
    '''Count occurrences of pattern in text.'''
    flags = 0 if case_sensitive else re.IGNORECASE
    return len(re.findall(pattern, text, flags))

def extract_sections(text, start_pattern, end_pattern):
    '''Extract sections between start and end patterns.'''
    pattern = f"{re.escape(start_pattern)}(.*?){re.escape(end_pattern)}"
    matches = re.findall(pattern, text, re.DOTALL)
    return matches

def summarize_text(text, max_length=500):
    '''Create a simple summary of text.'''
    words = text.split()
    if len(words) <= max_length:
        return text
    return ' '.join(words[:max_length]) + '...'

def get_recent_executions(n=5):
    '''Get the last n code executions with their results and errors.'''
    try:
        # Access the REPL instance from the namespace
        repl_instance = _repl_instance
        if hasattr(repl_instance, 'get_execution_history'):
            all_history = repl_instance.get_execution_history()
            return all_history[-n:] if len(all_history) > n else all_history
    except:
        pass
    return []

def get_execution_errors(n=5):
    '''Get recent execution errors to help with debugging.'''
    recent = get_recent_executions(n)
    errors = []
    for exec_info in recent:
        if not exec_info['success'] and exec_info['error']:
            errors.append({
                'code': exec_info['code'][:200] + '...' if len(exec_info['code']) > 200 else exec_info['code'],
                'error': exec_info['error']
            })
    return errors

# Advanced text processing functions
def advanced_search(pattern, context_lines=3, case_sensitive=False, doc_ids=None):
    '''Advanced search across documents with context extraction.'''
    try:
        return document_processor.execute_tool('advanced_search_docs', pattern, context_lines, case_sensitive, doc_ids)
    except:
        return {}

def extract_sections_docs(start_pattern, end_pattern, include_delimiters=False, doc_ids=None):
    '''Extract sections between patterns from documents.'''
    try:
        return document_processor.execute_tool('extract_sections_docs', start_pattern, end_pattern, include_delimiters, doc_ids)
    except:
        return {}

def slice_docs(start_pattern=None, end_pattern=None, start_line=None, end_line=None, start_char=None, end_char=None, doc_ids=None):
    '''Slice documents using various methods (patterns, lines, characters).'''
    try:
        return document_processor.execute_tool('slice_docs', start_pattern, end_pattern, start_line, end_line, start_char, end_char, doc_ids)
    except:
        return {}

def analyze_structure(doc_ids=None):
    '''Analyze text structure of documents.'''
    try:
        return document_processor.execute_tool('analyze_structure_docs', doc_ids)
    except:
        return {}

def semantic_analysis(doc_ids=None):
    '''Perform semantic analysis on documents.'''
    try:
        return document_processor.execute_tool('semantic_analysis_docs', doc_ids)
    except:
        return {}

def find_patterns(patterns, case_sensitive=False, doc_ids=None):
    '''Find multiple patterns in documents.'''
    try:
        return document_processor.execute_tool('find_patterns_docs', patterns, case_sensitive, doc_ids)
    except:
        return {}

def create_custom_tool(name, code, description=""):
    '''Create a custom text processing tool.'''
    try:
        return document_processor.create_custom_tool(name, code, description)
    except:
        return False

def create_text_pipeline(name, steps, description=""):
    '''Create a text processing pipeline from a list of steps.'''
    try:
        return document_processor.create_text_pipeline(name, steps, description)
    except:
        return False

def get_available_tools():
    '''Get list of available text processing tools.'''
    try:
        return document_processor.get_available_tools()
    except:
        return {}

def execute_tool(name, *args, **kwargs):
    '''Execute a text processing tool.'''
    try:
        return document_processor.execute_tool(name, *args, **kwargs)
    except:
        return None

# Text analysis helpers
def analyze_single_text(text, analysis_type='structure'):
    '''Analyze a single text string.'''
    try:
        if analysis_type == 'structure':
            return text_processor.analyze_text_structure(text)
        elif analysis_type == 'semantic':
            return text_processor.semantic_analysis(text)
        elif analysis_type == 'patterns':
            return text_processor._analyze_pattern_frequency(text)
    except:
        pass
    return {}

def search_single_text(text, pattern, context_lines=3, case_sensitive=False):
    '''Search a single text with context.'''
    try:
        return text_processor.advanced_search(text, pattern, context_lines, case_sensitive)
    except:
        return []

def extract_sections_single(text, start_pattern, end_pattern, include_delimiters=False):
    '''Extract sections from a single text.'''
    try:
        return text_processor.extract_sections(text, start_pattern, end_pattern, include_delimiters)
    except:
        return []

def slice_single_text(text, start_pattern=None, end_pattern=None, start_line=None, end_line=None, start_char=None, end_char=None):
    '''Slice a single text using various methods.'''
    try:
        return text_processor.slice_text(text, start_pattern, end_pattern, start_line, end_line, start_char, end_char)
    except:
        return text

"""
    
    success, output, error = repl.execute_code(utility_code)
    if not success:
        logger.warning(f"Failed to load utility functions: {error}")
    
    # Make REPL instance accessible to helper functions
    repl.set_variable('_repl_instance', repl)
    
    return repl, analyzer
