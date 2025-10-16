"""
Advanced Text Processing Toolkit for RLM
Provides comprehensive text manipulation, analysis, and processing capabilities
that can be dynamically created and used by the LLM.
"""

import re
import json
import logging
from typing import Dict, List, Optional, Any, Tuple, Callable, Union
from dataclasses import dataclass
from collections import Counter, defaultdict
import itertools
import functools
import math

logger = logging.getLogger(__name__)


@dataclass
class TextMatch:
    """Represents a text match with context."""
    text: str
    start: int
    end: int
    line_number: int
    context_before: str
    context_after: str
    full_line: str


@dataclass
class TextSection:
    """Represents a section of text with metadata."""
    content: str
    start_line: int
    end_line: int
    section_type: str
    metadata: Dict[str, Any]


class TextProcessor:
    """Core text processing engine with advanced capabilities."""
    
    def __init__(self):
        self.custom_functions = {}
        self.pattern_cache = {}
        self.analysis_cache = {}
    
    def register_custom_function(self, name: str, func: Callable, description: str = ""):
        """Register a custom text processing function."""
        self.custom_functions[name] = {
            'function': func,
            'description': description
        }
        logger.info(f"Registered custom function: {name}")
    
    def create_dynamic_function(self, name: str, code: str, description: str = ""):
        """Create a dynamic function from code string."""
        try:
            # Create a safe namespace for the function
            namespace = {
                're': re,
                'json': json,
                'Counter': Counter,
                'defaultdict': defaultdict,
                'itertools': itertools,
                'functools': functools,
                'math': math,
                'len': len,
                'str': str,
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
                # Add access to this processor's methods
                'advanced_search': self.advanced_search,
                'extract_sections': self.extract_sections,
                'slice_text': self.slice_text,
                'find_patterns': self.find_patterns,
                'analyze_text_structure': self.analyze_text_structure,
                'semantic_analysis': self.semantic_analysis,
                'create_text_filter': self.create_text_filter,
                'batch_process': self.batch_process,
            }
            
            # Compile and execute the function code
            exec(code, namespace)
            
            if name in namespace:
                self.register_custom_function(name, namespace[name], description)
                return True
            else:
                logger.error(f"Function {name} not found in compiled code")
                return False
                
        except Exception as e:
            logger.error(f"Failed to create dynamic function {name}: {e}")
            return False
    
    def create_text_pipeline(self, name: str, steps: List[Dict[str, Any]], description: str = ""):
        """Create a text processing pipeline from a list of steps."""
        def pipeline_function(text, **kwargs):
            result = text
            for step in steps:
                step_type = step.get('type')
                step_params = step.get('params', {})
                
                # Ensure result is a string for text processing operations
                if step_type in ['search', 'extract', 'slice'] and not isinstance(result, str):
                    if isinstance(result, list):
                        # If result is a list of matches, extract the text from the first match
                        if result and hasattr(result[0], 'text'):
                            result = result[0].text
                        elif result and isinstance(result[0], str):
                            result = result[0]
                        else:
                            result = str(result[0]) if result else ""
                    else:
                        result = str(result)
                
                if step_type == 'search':
                    pattern = step_params.get('pattern', '')
                    context_lines = step_params.get('context_lines', 3)
                    matches = self.advanced_search(result, pattern, context_lines)
                    result = matches
                
                elif step_type == 'extract':
                    start_pattern = step_params.get('start_pattern', '')
                    end_pattern = step_params.get('end_pattern', '')
                    sections = self.extract_sections(result, start_pattern, end_pattern)
                    result = sections
                
                elif step_type == 'slice':
                    start_line = step_params.get('start_line')
                    end_line = step_params.get('end_line')
                    start_char = step_params.get('start_char')
                    end_char = step_params.get('end_char')
                    result = self.slice_text(result, start_line=start_line, end_line=end_line, 
                                           start_char=start_char, end_char=end_char)
                
                elif step_type == 'filter':
                    filter_type = step_params.get('filter_type', 'length')
                    # Remove filter_type from step_params to avoid duplicate argument
                    filter_params = {k: v for k, v in step_params.items() if k != 'filter_type'}
                    filter_func = self.create_text_filter(filter_type, **filter_params)
                    if isinstance(result, list):
                        result = [item for item in result if filter_func(str(item))]
                    else:
                        if not filter_func(str(result)):
                            result = ""
                
                elif step_type == 'transform':
                    transform_code = step_params.get('code', '')
                    if transform_code:
                        try:
                            namespace = {'text': result, 're': re, 'json': json}
                            exec(transform_code, namespace)
                            if 'result' in namespace:
                                result = namespace['result']
                        except Exception as e:
                            logger.error(f"Transform step failed: {e}")
                
                elif step_type == 'custom':
                    func_name = step_params.get('function')
                    if func_name in self.custom_functions:
                        result = self.custom_functions[func_name]['function'](result, **step_params)
            
            return result
        
        self.register_custom_function(name, pipeline_function, description)
        return True
    
    def advanced_search(self, text: str, pattern: str, context_lines: int = 3, 
                       case_sensitive: bool = False, multiline: bool = False) -> List[TextMatch]:
        """Advanced search with context extraction."""
        flags = 0 if case_sensitive else re.IGNORECASE
        if multiline:
            flags |= re.MULTILINE | re.DOTALL
        
        lines = text.split('\n')
        matches = []
        
        for line_num, line in enumerate(lines):
            for match in re.finditer(pattern, line, flags):
                # Extract context
                start_context = max(0, line_num - context_lines)
                end_context = min(len(lines), line_num + context_lines + 1)
                
                context_before = '\n'.join(lines[start_context:line_num])
                context_after = '\n'.join(lines[line_num + 1:end_context])
                
                matches.append(TextMatch(
                    text=match.group(),
                    start=match.start(),
                    end=match.end(),
                    line_number=line_num + 1,
                    context_before=context_before,
                    context_after=context_after,
                    full_line=line
                ))
        
        return matches
    
    def extract_sections(self, text: str, start_pattern: str, end_pattern: str, 
                        include_delimiters: bool = False) -> List[TextSection]:
        """Extract sections between patterns."""
        pattern = f"({re.escape(start_pattern)})(.*?)({re.escape(end_pattern)})"
        matches = re.findall(pattern, text, re.DOTALL)
        
        sections = []
        current_pos = 0
        
        for match in matches:
            start_delim, content, end_delim = match
            
            # Find line numbers
            start_line = text[:text.find(start_delim, current_pos)].count('\n') + 1
            end_line = start_line + content.count('\n')
            
            section_content = content
            if include_delimiters:
                section_content = start_delim + content + end_delim
            
            sections.append(TextSection(
                content=section_content,
                start_line=start_line,
                end_line=end_line,
                section_type="extracted",
                metadata={
                    'start_delimiter': start_delim,
                    'end_delimiter': end_delim,
                    'include_delimiters': include_delimiters
                }
            ))
            
            current_pos = text.find(end_delim, current_pos) + len(end_delim)
        
        return sections
    
    def slice_text(self, text: str, start_pattern: str = None, end_pattern: str = None,
                   start_line: int = None, end_line: int = None, 
                   start_char: int = None, end_char: int = None) -> str:
        """Slice text using various methods."""
        if start_pattern and end_pattern:
            # Pattern-based slicing
            start_match = re.search(re.escape(start_pattern), text)
            end_match = re.search(re.escape(end_pattern), text)
            
            if start_match and end_match:
                return text[start_match.start():end_match.end()]
        
        elif start_line is not None and end_line is not None:
            # Line-based slicing
            lines = text.split('\n')
            start_idx = max(0, start_line - 1)
            end_idx = min(len(lines), end_line)
            return '\n'.join(lines[start_idx:end_idx])
        
        elif start_char is not None and end_char is not None:
            # Character-based slicing
            return text[start_char:end_char]
        
        return text
    
    def find_patterns(self, text: str, patterns: List[str], 
                     case_sensitive: bool = False) -> Dict[str, List[TextMatch]]:
        """Find multiple patterns in text."""
        results = {}
        
        for pattern in patterns:
            matches = self.advanced_search(text, pattern, case_sensitive=case_sensitive)
            if matches:
                results[pattern] = matches
        
        return results
    
    def analyze_text_structure(self, text: str) -> Dict[str, Any]:
        """Analyze the structure of text."""
        lines = text.split('\n')
        
        structure = {
            'total_lines': len(lines),
            'non_empty_lines': len([line for line in lines if line.strip()]),
            'indentation_analysis': self._analyze_indentation(lines),
            'paragraph_analysis': self._analyze_paragraphs(text),
            'code_analysis': self._analyze_code_blocks(text),
            'header_analysis': self._analyze_headers(text),
            'list_analysis': self._analyze_lists(text),
            'table_analysis': self._analyze_tables(text),
            'link_analysis': self._analyze_links(text),
            'pattern_frequency': self._analyze_pattern_frequency(text)
        }
        
        return structure
    
    def _analyze_indentation(self, lines: List[str]) -> Dict[str, Any]:
        """Analyze indentation patterns."""
        indent_levels = defaultdict(int)
        indent_chars = defaultdict(int)
        
        for line in lines:
            if line.strip():
                indent = len(line) - len(line.lstrip())
                indent_levels[indent] += 1
                
                if indent > 0:
                    indent_char = line[len(line) - len(line.lstrip()) - 1]
                    indent_chars[indent_char] += 1
        
        return {
            'levels': dict(indent_levels),
            'characters': dict(indent_chars),
            'max_level': max(indent_levels.keys()) if indent_levels else 0
        }
    
    def _analyze_paragraphs(self, text: str) -> Dict[str, Any]:
        """Analyze paragraph structure."""
        paragraphs = re.split(r'\n\s*\n', text)
        paragraph_lengths = [len(p.split()) for p in paragraphs if p.strip()]
        
        return {
            'count': len(paragraphs),
            'avg_length': sum(paragraph_lengths) / len(paragraph_lengths) if paragraph_lengths else 0,
            'max_length': max(paragraph_lengths) if paragraph_lengths else 0,
            'min_length': min(paragraph_lengths) if paragraph_lengths else 0
        }
    
    def _analyze_code_blocks(self, text: str) -> Dict[str, Any]:
        """Analyze code blocks in text."""
        # Markdown code blocks
        markdown_blocks = re.findall(r'```[\s\S]*?```', text)
        
        # Inline code
        inline_code = re.findall(r'`[^`]+`', text)
        
        # Indented code blocks
        lines = text.split('\n')
        indented_blocks = []
        current_block = []
        
        for line in lines:
            if line.startswith('    ') or line.startswith('\t'):
                current_block.append(line)
            else:
                if current_block:
                    indented_blocks.append('\n'.join(current_block))
                    current_block = []
        
        if current_block:
            indented_blocks.append('\n'.join(current_block))
        
        return {
            'markdown_blocks': len(markdown_blocks),
            'inline_code': len(inline_code),
            'indented_blocks': len(indented_blocks),
            'total_code_lines': sum(len(block.split('\n')) for block in markdown_blocks + indented_blocks)
        }
    
    def _analyze_headers(self, text: str) -> Dict[str, Any]:
        """Analyze headers in text."""
        # Markdown headers
        markdown_headers = re.findall(r'^(#{1,6})\s+(.+)$', text, re.MULTILINE)
        
        # Common header patterns
        title_patterns = [
            r'^([A-Z][A-Z\s]+)$',  # ALL CAPS
            r'^([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)$',  # Title Case
            r'^(\d+\.\s+[A-Z].*)$',  # Numbered sections
        ]
        
        other_headers = []
        for pattern in title_patterns:
            matches = re.findall(pattern, text, re.MULTILINE)
            other_headers.extend(matches)
        
        return {
            'markdown_headers': len(markdown_headers),
            'other_headers': len(other_headers),
            'header_levels': dict(Counter([len(level) for level, _ in markdown_headers]))
        }
    
    def _analyze_lists(self, text: str) -> Dict[str, Any]:
        """Analyze list structures."""
        # Bullet lists
        bullet_lists = re.findall(r'^[\s]*[-*+]\s+', text, re.MULTILINE)
        
        # Numbered lists
        numbered_lists = re.findall(r'^[\s]*\d+\.\s+', text, re.MULTILINE)
        
        return {
            'bullet_items': len(bullet_lists),
            'numbered_items': len(numbered_lists),
            'total_list_items': len(bullet_lists) + len(numbered_lists)
        }
    
    def _analyze_tables(self, text: str) -> Dict[str, Any]:
        """Analyze table structures."""
        # Markdown tables
        table_lines = re.findall(r'^\|.*\|$', text, re.MULTILINE)
        separator_lines = re.findall(r'^\|[\s\-\|:]+\|$', text, re.MULTILINE)
        
        return {
            'table_lines': len(table_lines),
            'separator_lines': len(separator_lines),
            'estimated_tables': len(separator_lines)
        }
    
    def _analyze_links(self, text: str) -> Dict[str, Any]:
        """Analyze links in text."""
        # Markdown links
        markdown_links = re.findall(r'\[([^\]]+)\]\(([^)]+)\)', text)
        
        # URL patterns
        urls = re.findall(r'https?://[^\s]+', text)
        
        return {
            'markdown_links': len(markdown_links),
            'urls': len(urls),
            'link_texts': [text for text, _ in markdown_links]
        }
    
    def _analyze_pattern_frequency(self, text: str) -> Dict[str, int]:
        """Analyze frequency of common patterns."""
        patterns = {
            'email': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
            'phone': r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b',
            'url': r'https?://[^\s]+',
            'ip_address': r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b',
            'date': r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b',
            'time': r'\b\d{1,2}:\d{2}(?::\d{2})?\s*(?:AM|PM)?\b',
            'currency': r'\$\d+(?:,\d{3})*(?:\.\d{2})?',
            'percentage': r'\d+(?:\.\d+)?%',
            'version': r'\b\d+\.\d+(?:\.\d+)?\b'
        }
        
        frequency = {}
        for name, pattern in patterns.items():
            matches = re.findall(pattern, text, re.IGNORECASE)
            frequency[name] = len(matches)
        
        return frequency
    
    def semantic_analysis(self, text: str) -> Dict[str, Any]:
        """Perform semantic analysis on text."""
        words = re.findall(r'\b[a-zA-Z]+\b', text.lower())
        word_freq = Counter(words)
        
        # Remove common stop words
        stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with',
            'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had', 'do', 'does',
            'did', 'will', 'would', 'could', 'should', 'may', 'might', 'must', 'can', 'this',
            'that', 'these', 'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they', 'me', 'him',
            'her', 'us', 'them', 'my', 'your', 'his', 'her', 'its', 'our', 'their', 'not',
            'no', 'yes', 'so', 'if', 'then', 'else', 'when', 'where', 'why', 'how', 'what',
            'who', 'which', 'there', 'here', 'now', 'then', 'up', 'down', 'out', 'off',
            'over', 'under', 'again', 'further', 'once', 'more', 'most', 'other', 'some',
            'very', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few',
            'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own',
            'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don',
            'should', 'now'
        }
        
        filtered_words = {word: count for word, count in word_freq.items() 
                         if word not in stop_words and len(word) > 2}
        
        # Calculate text statistics
        total_words = len(words)
        unique_words = len(set(words))
        vocabulary_richness = unique_words / total_words if total_words > 0 else 0
        
        # Find key phrases (2-3 word combinations)
        phrases = []
        for i in range(len(words) - 1):
            phrase = f"{words[i]} {words[i+1]}"
            phrases.append(phrase)
        
        phrase_freq = Counter(phrases)
        key_phrases = dict(phrase_freq.most_common(20))
        
        return {
            'total_words': total_words,
            'unique_words': unique_words,
            'vocabulary_richness': vocabulary_richness,
            'top_words': dict(word_freq.most_common(20)),
            'filtered_top_words': dict(Counter(filtered_words).most_common(20)),
            'key_phrases': key_phrases,
            'word_length_distribution': self._analyze_word_lengths(words),
            'sentence_analysis': self._analyze_sentences(text)
        }
    
    def _analyze_word_lengths(self, words: List[str]) -> Dict[int, int]:
        """Analyze distribution of word lengths."""
        length_dist = defaultdict(int)
        for word in words:
            length_dist[len(word)] += 1
        return dict(length_dist)
    
    def _analyze_sentences(self, text: str) -> Dict[str, Any]:
        """Analyze sentence structure."""
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        sentence_lengths = [len(s.split()) for s in sentences]
        
        return {
            'count': len(sentences),
            'avg_length': sum(sentence_lengths) / len(sentence_lengths) if sentence_lengths else 0,
            'max_length': max(sentence_lengths) if sentence_lengths else 0,
            'min_length': min(sentence_lengths) if sentence_lengths else 0
        }
    
    def create_text_filter(self, filter_type: str, **kwargs) -> Callable:
        """Create a text filter function."""
        if filter_type == "length":
            min_length = kwargs.get('min_length', 0)
            max_length = kwargs.get('max_length', float('inf'))
            return lambda text: min_length <= len(text) <= max_length
        
        elif filter_type == "pattern":
            pattern = kwargs.get('pattern', '')
            case_sensitive = kwargs.get('case_sensitive', False)
            flags = 0 if case_sensitive else re.IGNORECASE
            return lambda text: bool(re.search(pattern, text, flags))
        
        elif filter_type == "word_count":
            min_words = kwargs.get('min_words', 0)
            max_words = kwargs.get('max_words', float('inf'))
            return lambda text: min_words <= len(text.split()) <= max_words
        
        elif filter_type == "line_count":
            min_lines = kwargs.get('min_lines', 0)
            max_lines = kwargs.get('max_lines', float('inf'))
            return lambda text: min_lines <= len(text.split('\n')) <= max_lines
        
        else:
            raise ValueError(f"Unknown filter type: {filter_type}")
    
    def batch_process(self, texts: List[str], processor: Callable, 
                     filter_func: Optional[Callable] = None) -> List[Any]:
        """Process multiple texts with optional filtering."""
        results = []
        
        for text in texts:
            if filter_func is None or filter_func(text):
                try:
                    result = processor(text)
                    results.append(result)
                except Exception as e:
                    logger.error(f"Error processing text: {e}")
                    results.append(None)
        
        return results
    
    def get_available_functions(self) -> Dict[str, str]:
        """Get list of available custom functions."""
        return {name: info['description'] for name, info in self.custom_functions.items()}
    
    def execute_custom_function(self, name: str, *args, **kwargs) -> Any:
        """Execute a custom function."""
        if name not in self.custom_functions:
            raise ValueError(f"Custom function '{name}' not found")
        
        try:
            return self.custom_functions[name]['function'](*args, **kwargs)
        except Exception as e:
            logger.error(f"Error executing custom function '{name}': {e}")
            raise


class DocumentTextProcessor:
    """High-level document text processing interface."""
    
    def __init__(self, documents: Dict[str, str]):
        self.documents = documents
        self.processor = TextProcessor()
        self._setup_default_functions()
    
    def _setup_default_functions(self):
        """Set up default text processing functions."""
        
        # Advanced search function
        def advanced_search_docs(pattern: str, context_lines: int = 3, 
                               case_sensitive: bool = False, doc_ids: List[str] = None):
            """Search across documents with context."""
            if doc_ids is None:
                doc_ids = list(self.documents.keys())
            
            results = {}
            for doc_id in doc_ids:
                if doc_id in self.documents:
                    matches = self.processor.advanced_search(
                        self.documents[doc_id], pattern, context_lines, case_sensitive
                    )
                    if matches:
                        results[doc_id] = matches
            
            return results
        
        # Extract sections function
        def extract_sections_docs(start_pattern: str, end_pattern: str, 
                                include_delimiters: bool = False, doc_ids: List[str] = None):
            """Extract sections from documents."""
            if doc_ids is None:
                doc_ids = list(self.documents.keys())
            
            results = {}
            for doc_id in doc_ids:
                if doc_id in self.documents:
                    sections = self.processor.extract_sections(
                        self.documents[doc_id], start_pattern, end_pattern, include_delimiters
                    )
                    if sections:
                        results[doc_id] = sections
            
            return results
        
        # Slice documents function
        def slice_docs(start_pattern: str = None, end_pattern: str = None,
                      start_line: int = None, end_line: int = None,
                      start_char: int = None, end_char: int = None,
                      doc_ids: List[str] = None):
            """Slice documents using various methods."""
            if doc_ids is None:
                doc_ids = list(self.documents.keys())
            
            results = {}
            for doc_id in doc_ids:
                if doc_id in self.documents:
                    sliced = self.processor.slice_text(
                        self.documents[doc_id], start_pattern, end_pattern,
                        start_line, end_line, start_char, end_char
                    )
                    results[doc_id] = sliced
            
            return results
        
        # Analyze structure function
        def analyze_structure_docs(doc_ids: List[str] = None):
            """Analyze structure of documents."""
            if doc_ids is None:
                doc_ids = list(self.documents.keys())
            
            results = {}
            for doc_id in doc_ids:
                if doc_id in self.documents:
                    structure = self.processor.analyze_text_structure(self.documents[doc_id])
                    results[doc_id] = structure
            
            return results
        
        # Semantic analysis function
        def semantic_analysis_docs(doc_ids: List[str] = None):
            """Perform semantic analysis on documents."""
            if doc_ids is None:
                doc_ids = list(self.documents.keys())
            
            results = {}
            for doc_id in doc_ids:
                if doc_id in self.documents:
                    analysis = self.processor.semantic_analysis(self.documents[doc_id])
                    results[doc_id] = analysis
            
            return results
        
        # Find patterns function
        def find_patterns_docs(patterns: List[str], case_sensitive: bool = False,
                             doc_ids: List[str] = None):
            """Find multiple patterns in documents."""
            if doc_ids is None:
                doc_ids = list(self.documents.keys())
            
            results = {}
            for doc_id in doc_ids:
                if doc_id in self.documents:
                    matches = self.processor.find_patterns(
                        self.documents[doc_id], patterns, case_sensitive
                    )
                    if matches:
                        results[doc_id] = matches
            
            return results
        
        # Register default functions
        self.processor.register_custom_function(
            "advanced_search_docs", advanced_search_docs,
            "Search across documents with context extraction"
        )
        self.processor.register_custom_function(
            "extract_sections_docs", extract_sections_docs,
            "Extract sections between patterns from documents"
        )
        self.processor.register_custom_function(
            "slice_docs", slice_docs,
            "Slice documents using various methods (patterns, lines, characters)"
        )
        self.processor.register_custom_function(
            "analyze_structure_docs", analyze_structure_docs,
            "Analyze text structure of documents"
        )
        self.processor.register_custom_function(
            "semantic_analysis_docs", semantic_analysis_docs,
            "Perform semantic analysis on documents"
        )
        self.processor.register_custom_function(
            "find_patterns_docs", find_patterns_docs,
            "Find multiple patterns in documents"
        )
    
    def create_custom_tool(self, name: str, code: str, description: str = ""):
        """Create a custom text processing tool."""
        return self.processor.create_dynamic_function(name, code, description)
    
    def create_text_pipeline(self, name: str, steps: List[Dict[str, Any]], description: str = ""):
        """Create a text processing pipeline."""
        return self.processor.create_text_pipeline(name, steps, description)
    
    def get_available_tools(self) -> Dict[str, str]:
        """Get list of available tools."""
        return self.processor.get_available_functions()
    
    def execute_tool(self, name: str, *args, **kwargs):
        """Execute a text processing tool."""
        return self.processor.execute_custom_function(name, *args, **kwargs)
