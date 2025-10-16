"""
Document Analysis Module for RLM
Provides utilities for loading, processing, and analyzing large collections of documents.
"""

import os
import json
import logging
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
import hashlib
import time
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class DocumentMetadata:
    """Metadata for a document."""
    doc_id: str
    filename: str
    file_size: int
    created_time: float
    modified_time: float
    content_hash: str
    file_type: str
    encoding: str = "utf-8"


class DocumentLoader:
    """Loads documents from various sources."""
    
    def __init__(self):
        self.supported_extensions = {
            '.txt', '.md', '.py', '.js', '.ts', '.html', '.css', '.json', 
            '.xml', '.csv', '.log', '.sql', '.yaml', '.yml', '.ini', '.cfg'
        }
    
    def load_from_directory(self, directory_path: str, recursive: bool = True) -> Tuple[Dict[str, str], Dict[str, DocumentMetadata]]:
        """
        Load all supported documents from a directory.
        
        Args:
            directory_path: Path to the directory
            recursive: Whether to search recursively
            
        Returns:
            Tuple of (documents_dict, metadata_dict)
        """
        documents = {}
        metadata = {}
        
        directory = Path(directory_path)
        if not directory.exists():
            raise FileNotFoundError(f"Directory not found: {directory_path}")
        
        pattern = "**/*" if recursive else "*"
        
        for file_path in directory.glob(pattern):
            if file_path.is_file() and file_path.suffix.lower() in self.supported_extensions:
                try:
                    doc_id = str(file_path.relative_to(directory))
                    content = self._read_file(file_path)
                    
                    if content:  # Only include non-empty files
                        documents[doc_id] = content
                        metadata[doc_id] = self._create_metadata(doc_id, file_path, content)
                        
                except Exception as e:
                    logger.warning(f"Failed to load {file_path}: {e}")
        
        logger.info(f"Loaded {len(documents)} documents from {directory_path}")
        return documents, metadata
    
    def load_from_files(self, file_paths: List[str]) -> Tuple[Dict[str, str], Dict[str, DocumentMetadata]]:
        """
        Load documents from a list of file paths.
        
        Args:
            file_paths: List of file paths to load
            
        Returns:
            Tuple of (documents_dict, metadata_dict)
        """
        documents = {}
        metadata = {}
        
        for file_path in file_paths:
            try:
                path = Path(file_path)
                if path.exists() and path.is_file():
                    content = self._read_file(path)
                    if content:
                        doc_id = path.name
                        documents[doc_id] = content
                        metadata[doc_id] = self._create_metadata(doc_id, path, content)
                else:
                    logger.warning(f"File not found: {file_path}")
                    
            except Exception as e:
                logger.warning(f"Failed to load {file_path}: {e}")
        
        logger.info(f"Loaded {len(documents)} documents from file list")
        return documents, metadata
    
    def load_from_json(self, json_path: str) -> Tuple[Dict[str, str], Dict[str, DocumentMetadata]]:
        """
        Load documents from a JSON file.
        Expected format: {"doc_id": "content", ...}
        
        Args:
            json_path: Path to JSON file
            
        Returns:
            Tuple of (documents_dict, metadata_dict)
        """
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        documents = {}
        metadata = {}
        
        for doc_id, content in data.items():
            if isinstance(content, str) and content.strip():
                documents[doc_id] = content
                metadata[doc_id] = DocumentMetadata(
                    doc_id=doc_id,
                    filename=doc_id,
                    file_size=len(content.encode('utf-8')),
                    created_time=time.time(),
                    modified_time=time.time(),
                    content_hash=hashlib.md5(content.encode('utf-8')).hexdigest(),
                    file_type='json'
                )
        
        logger.info(f"Loaded {len(documents)} documents from JSON")
        return documents, metadata
    
    def _read_file(self, file_path: Path) -> Optional[str]:
        """Read a file with proper encoding detection."""
        encodings = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']
        
        for encoding in encodings:
            try:
                with open(file_path, 'r', encoding=encoding) as f:
                    return f.read()
            except UnicodeDecodeError:
                continue
        
        logger.warning(f"Could not decode {file_path} with any supported encoding")
        return None
    
    def _create_metadata(self, doc_id: str, file_path: Path, content: str) -> DocumentMetadata:
        """Create metadata for a document."""
        stat = file_path.stat()
        return DocumentMetadata(
            doc_id=doc_id,
            filename=file_path.name,
            file_size=stat.st_size,
            created_time=stat.st_ctime,
            modified_time=stat.st_mtime,
            content_hash=hashlib.md5(content.encode('utf-8')).hexdigest(),
            file_type=file_path.suffix.lower()
        )


class DocumentAnalyzer:
    """
    Advanced document analysis capabilities for RLM.
    Provides tools for analyzing large collections of documents.
    """
    
    def __init__(self, documents: Dict[str, str], metadata: Dict[str, DocumentMetadata]):
        self.documents = documents
        self.metadata = metadata
        self.analysis_cache = {}
    
    def get_document_stats(self) -> Dict[str, Any]:
        """Get comprehensive statistics about the document collection."""
        if 'stats' in self.analysis_cache:
            return self.analysis_cache['stats']
        
        stats = {
            'total_documents': len(self.documents),
            'total_characters': 0,
            'total_words': 0,
            'total_lines': 0,
            'file_types': {},
            'size_distribution': {
                'small': 0,    # < 1KB
                'medium': 0,   # 1KB - 100KB
                'large': 0,    # 100KB - 1MB
                'huge': 0      # > 1MB
            },
            'avg_doc_length': 0,
            'longest_doc': None,
            'shortest_doc': None,
            'doc_lengths': {}
        }
        
        longest_length = 0
        shortest_length = float('inf')
        
        for doc_id, content in self.documents.items():
            content_length = len(content)
            word_count = len(content.split())
            line_count = len(content.split('\n'))
            
            stats['total_characters'] += content_length
            stats['total_words'] += word_count
            stats['total_lines'] += line_count
            
            # File type distribution
            file_type = self.metadata.get(doc_id, DocumentMetadata(doc_id, '', 0, 0, 0, '', '')).file_type
            stats['file_types'][file_type] = stats['file_types'].get(file_type, 0) + 1
            
            # Size distribution
            if content_length < 1024:
                stats['size_distribution']['small'] += 1
            elif content_length < 102400:
                stats['size_distribution']['medium'] += 1
            elif content_length < 1048576:
                stats['size_distribution']['large'] += 1
            else:
                stats['size_distribution']['huge'] += 1
            
            # Track longest and shortest
            if content_length > longest_length:
                longest_length = content_length
                stats['longest_doc'] = doc_id
            
            if content_length < shortest_length:
                shortest_length = content_length
                stats['shortest_doc'] = doc_id
            
            stats['doc_lengths'][doc_id] = content_length
        
        if stats['total_documents'] > 0:
            stats['avg_doc_length'] = stats['total_characters'] / stats['total_documents']
        
        self.analysis_cache['stats'] = stats
        return stats
    
    def search_across_documents(self, pattern: str, case_sensitive: bool = False, 
                              file_types: Optional[List[str]] = None) -> Dict[str, List[Dict]]:
        """
        Search for patterns across all documents.
        
        Args:
            pattern: Regex pattern to search for
            case_sensitive: Whether search should be case sensitive
            file_types: Optional list of file types to search in
            
        Returns:
            Dictionary of {doc_id: [match_info]}
        """
        import re
        
        flags = 0 if case_sensitive else re.IGNORECASE
        results = {}
        
        for doc_id, content in self.documents.items():
            # Filter by file type if specified
            if file_types:
                file_type = self.metadata.get(doc_id, DocumentMetadata(doc_id, '', 0, 0, 0, '', '')).file_type
                if file_type not in file_types:
                    continue
            
            matches = []
            lines = content.split('\n')
            
            for line_num, line in enumerate(lines, 1):
                for match in re.finditer(pattern, line, flags):
                    matches.append({
                        'line_number': line_num,
                        'line_content': line.strip(),
                        'match_text': match.group(),
                        'start_pos': match.start(),
                        'end_pos': match.end()
                    })
            
            if matches:
                results[doc_id] = matches
        
        return results
    
    def find_similar_documents(self, doc_id: str, threshold: float = 0.8) -> List[Tuple[str, float]]:
        """
        Find documents similar to the given document using content similarity.
        
        Args:
            doc_id: ID of the reference document
            threshold: Similarity threshold (0-1)
            
        Returns:
            List of (doc_id, similarity_score) tuples
        """
        if doc_id not in self.documents:
            return []
        
        reference_content = self.documents[doc_id]
        reference_words = set(reference_content.lower().split())
        
        similarities = []
        
        for other_doc_id, other_content in self.documents.items():
            if other_doc_id == doc_id:
                continue
            
            other_words = set(other_content.lower().split())
            
            # Calculate Jaccard similarity
            intersection = len(reference_words & other_words)
            union = len(reference_words | other_words)
            
            if union > 0:
                similarity = intersection / union
                if similarity >= threshold:
                    similarities.append((other_doc_id, similarity))
        
        return sorted(similarities, key=lambda x: x[1], reverse=True)
    
    def extract_keywords(self, top_n: int = 50) -> List[Tuple[str, int]]:
        """
        Extract most common keywords from all documents.
        
        Args:
            top_n: Number of top keywords to return
            
        Returns:
            List of (keyword, frequency) tuples
        """
        import re
        from collections import Counter
        
        # Simple keyword extraction (can be enhanced with NLP libraries)
        all_text = ' '.join(self.documents.values()).lower()
        
        # Remove common stop words and extract words
        stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with',
            'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had', 'do', 'does',
            'did', 'will', 'would', 'could', 'should', 'may', 'might', 'must', 'can', 'this',
            'that', 'these', 'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they', 'me', 'him',
            'her', 'us', 'them', 'my', 'your', 'his', 'her', 'its', 'our', 'their'
        }
        
        # Extract words (alphanumeric sequences)
        words = re.findall(r'\b[a-zA-Z]{3,}\b', all_text)
        words = [word for word in words if word not in stop_words]
        
        # Count frequencies
        word_counts = Counter(words)
        
        return word_counts.most_common(top_n)
    
    def get_document_sample(self, doc_id: str, start_line: int = 0, num_lines: int = 10) -> str:
        """Get a sample of lines from a specific document."""
        if doc_id not in self.documents:
            return f"Document {doc_id} not found"
        
        lines = self.documents[doc_id].split('\n')
        end_line = min(start_line + num_lines, len(lines))
        
        return '\n'.join(lines[start_line:end_line])
    
    def analyze_document_structure(self, doc_id: str) -> Dict[str, Any]:
        """
        Analyze the structure of a specific document.
        
        Args:
            doc_id: ID of the document to analyze
            
        Returns:
            Dictionary with structure analysis
        """
        if doc_id not in self.documents:
            return {"error": f"Document {doc_id} not found"}
        
        content = self.documents[doc_id]
        lines = content.split('\n')
        
        structure = {
            'total_lines': len(lines),
            'non_empty_lines': len([line for line in lines if line.strip()]),
            'indentation_levels': {},
            'common_patterns': {},
            'sections': []
        }
        
        # Analyze indentation
        for line in lines:
            if line.strip():
                indent = len(line) - len(line.lstrip())
                structure['indentation_levels'][indent] = structure['indentation_levels'].get(indent, 0) + 1
        
        # Look for common patterns (headers, code blocks, etc.)
        import re
        
        # Headers (markdown style)
        headers = re.findall(r'^#+\s+(.+)$', content, re.MULTILINE)
        structure['sections'].extend([{'type': 'header', 'content': h} for h in headers])
        
        # Code blocks
        code_blocks = re.findall(r'```[\s\S]*?```', content)
        structure['sections'].extend([{'type': 'code_block', 'content': cb[:100] + '...'} for cb in code_blocks])
        
        return structure
    
    def generate_summary_report(self) -> str:
        """Generate a comprehensive summary report of the document collection."""
        stats = self.get_document_stats()
        keywords = self.extract_keywords(10)
        
        report = f"""
Document Collection Summary Report
=================================

Overview:
- Total Documents: {stats['total_documents']}
- Total Characters: {stats['total_characters']:,}
- Total Words: {stats['total_words']:,}
- Total Lines: {stats['total_lines']:,}
- Average Document Length: {stats['avg_doc_length']:.0f} characters

File Type Distribution:
"""
        
        for file_type, count in stats['file_types'].items():
            report += f"- {file_type}: {count} documents\n"
        
        report += f"""
Size Distribution:
- Small (< 1KB): {stats['size_distribution']['small']} documents
- Medium (1KB - 100KB): {stats['size_distribution']['medium']} documents
- Large (100KB - 1MB): {stats['size_distribution']['large']} documents
- Huge (> 1MB): {stats['size_distribution']['huge']} documents

Longest Document: {stats['longest_doc']} ({stats['doc_lengths'].get(stats['longest_doc'], 0):,} characters)
Shortest Document: {stats['shortest_doc']} ({stats['doc_lengths'].get(stats['shortest_doc'], 0):,} characters)

Top Keywords:
"""
        
        for keyword, frequency in keywords:
            report += f"- {keyword}: {frequency} occurrences\n"
        
        return report
