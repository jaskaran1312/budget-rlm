#!/usr/bin/env python3
"""
Advanced Text Processing Example
Demonstrates the comprehensive text processing capabilities of the RLM system.
"""

import sys
import os
from pathlib import Path

# Add the current directory to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from main import RLMDocumentAnalyzer
from config import get_config

def create_sample_documents():
    """Create sample documents for demonstration."""
    sample_docs = {
        "python_code.py": '''
"""
Sample Python module for demonstration.
"""

import re
import json
from typing import List, Dict, Optional

class TextProcessor:
    """A class for processing text data."""
    
    def __init__(self, config: Dict[str, str]):
        self.config = config
        self.patterns = []
    
    def add_pattern(self, pattern: str) -> None:
        """Add a regex pattern for text matching."""
        self.patterns.append(pattern)
    
    def process_text(self, text: str) -> List[str]:
        """Process text using configured patterns."""
        results = []
        for pattern in self.patterns:
            matches = re.findall(pattern, text)
            results.extend(matches)
        return results
    
    def save_results(self, results: List[str], filename: str) -> bool:
        """Save results to a JSON file."""
        try:
            with open(filename, 'w') as f:
                json.dump(results, f, indent=2)
            return True
        except Exception as e:
            print(f"Error saving results: {e}")
            return False

def main():
    """Main function."""
    config = {"debug": "true", "output_dir": "./output"}
    processor = TextProcessor(config)
    
    # Add some patterns
    processor.add_pattern(r'\\b\\w+@\\w+\\.\\w+\\b')  # Email pattern
    processor.add_pattern(r'\\b\\d{3}-\\d{3}-\\d{4}\\b')  # Phone pattern
    
    # Process some sample text
    sample_text = """
    Contact us at support@example.com or call 555-123-4567.
    For sales inquiries, email sales@company.com or call 555-987-6543.
    """
    
    results = processor.process_text(sample_text)
    print(f"Found {len(results)} matches: {results}")
    
    # Save results
    success = processor.save_results(results, "matches.json")
    print(f"Results saved: {success}")

if __name__ == "__main__":
    main()
''',
        
        "markdown_doc.md": '''
# Advanced Text Processing Documentation

## Overview

This document describes the advanced text processing capabilities of the RLM system.

## Features

### 1. Pattern Matching
- **Regex Support**: Full regular expression support
- **Context Extraction**: Extract surrounding context for matches
- **Multi-pattern Search**: Search for multiple patterns simultaneously

### 2. Text Analysis
- **Structure Analysis**: Analyze document structure (headers, lists, code blocks)
- **Semantic Analysis**: Word frequency, vocabulary richness, key phrases
- **Pattern Frequency**: Count common patterns (emails, phones, URLs, etc.)

### 3. Text Manipulation
- **Section Extraction**: Extract sections between delimiters
- **Text Slicing**: Slice text by lines, characters, or patterns
- **Custom Filters**: Create custom filtering functions

## Examples

### Code Block Extraction
```python
def extract_code_blocks(text):
    """Extract Python code blocks from text."""
    pattern = r'```python\\s*(.*?)\\s*```'
    return re.findall(pattern, text, re.DOTALL)
```

### Email Extraction
```python
def extract_emails(text):
    """Extract email addresses from text."""
    pattern = r'\\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\\.[A-Z|a-z]{2,}\\b'
    return re.findall(pattern, text)
```

## Configuration

| Setting | Description | Default |
|---------|-------------|---------|
| `max_context_lines` | Maximum context lines around matches | 3 |
| `case_sensitive` | Whether searches are case sensitive | false |
| `include_delimiters` | Include delimiters in extracted sections | false |

## Contact

For questions or support, contact:
- Email: support@example.com
- Phone: 555-123-4567
- Website: https://example.com
''',
        
        "config_file.json": '''
{
  "application": {
    "name": "RLM Text Processor",
    "version": "2.0.0",
    "description": "Advanced text processing system"
  },
  "processing": {
    "max_iterations": 50,
    "timeout_seconds": 300,
    "enable_recursive_calls": true,
    "max_recursion_depth": 3
  },
  "text_analysis": {
    "context_lines": 3,
    "case_sensitive": false,
    "include_delimiters": false,
    "max_output_length": 10000
  },
  "output": {
    "format": "json",
    "include_metadata": true,
    "compress": false
  },
  "logging": {
    "level": "INFO",
    "file": "rlm_processing.log",
    "max_size": "10MB"
  }
}
''',
        
        "log_file.txt": '''
2024-01-15 10:30:15 INFO Starting RLM text processing
2024-01-15 10:30:15 DEBUG Loading configuration from config.json
2024-01-15 10:30:16 INFO Loaded 3 documents for processing
2024-01-15 10:30:16 DEBUG Initializing text processor
2024-01-15 10:30:16 INFO Text processor initialized successfully
2024-01-15 10:30:17 DEBUG Processing document: python_code.py
2024-01-15 10:30:17 INFO Found 2 function definitions in python_code.py
2024-01-15 10:30:17 DEBUG Processing document: markdown_doc.md
2024-01-15 10:30:18 INFO Found 3 code blocks in markdown_doc.md
2024-01-15 10:30:18 DEBUG Processing document: config_file.json
2024-01-15 10:30:18 INFO JSON structure analysis completed
2024-01-15 10:30:19 INFO Text processing completed successfully
2024-01-15 10:30:19 INFO Results saved to output/results.json
2024-01-15 10:30:19 INFO Processing time: 4.2 seconds
'''
    }
    
    return sample_docs

def demonstrate_basic_analysis():
    """Demonstrate basic text analysis capabilities."""
    print("="*80)
    print("BASIC TEXT ANALYSIS DEMONSTRATION")
    print("="*80)
    
    # Create sample documents
    sample_docs = create_sample_documents()
    
    # Initialize analyzer
    config = get_config()
    analyzer = RLMDocumentAnalyzer(config=config)
    
    # Load documents
    analyzer.documents = sample_docs
    analyzer._update_analyzer()
    
    # Get document statistics
    stats = analyzer.get_document_stats()
    print(f"Document Statistics:")
    print(f"- Total documents: {stats['total_documents']}")
    print(f"- Total characters: {stats['total_characters']:,}")
    print(f"- Total words: {stats['total_words']:,}")
    print(f"- Total lines: {stats['total_lines']:,}")
    print(f"- Average document length: {stats['avg_doc_length']:.0f} characters")
    
    # Search for patterns
    print(f"\nSearch Results:")
    search_results = analyzer.search_documents(r"def \w+")
    for doc_id, matches in search_results.items():
        print(f"- {doc_id}: {len(matches)} function definitions")
    
    # Generate summary report
    print(f"\nSummary Report:")
    report = analyzer.generate_summary_report()
    print(report[:500] + "..." if len(report) > 500 else report)

def demonstrate_advanced_processing():
    """Demonstrate advanced text processing capabilities."""
    print("\n" + "="*80)
    print("ADVANCED TEXT PROCESSING DEMONSTRATION")
    print("="*80)
    
    # Create sample documents
    sample_docs = create_sample_documents()
    
    # Initialize analyzer
    config = get_config()
    analyzer = RLMDocumentAnalyzer(config=config)
    
    # Load documents
    analyzer.documents = sample_docs
    analyzer._update_analyzer()
    
    # Demonstrate advanced search with context
    print("1. Advanced Search with Context:")
    query = """
    # Advanced search for function definitions with context
    results = advanced_search(r"def \\w+", context_lines=2)
    print(f"Found function definitions in {len(results)} documents:")
    for doc_id, matches in results.items():
        print(f"\\n{doc_id}:")
        for match in matches:
            print(f"  Line {match.line_number}: {match.text}")
            print(f"  Context: {match.context_before[-50:]}...{match.context_after[:50]}")
    """
    
    result = analyzer.analyze_with_rlm(query)
    print(result)
    
    # Demonstrate section extraction
    print("\n2. Section Extraction:")
    query = """
    # Extract code blocks from markdown document
    sections = extract_sections_docs("```python", "```", include_delimiters=True)
    print(f"Found code blocks in {len(sections)} documents:")
    for doc_id, doc_sections in sections.items():
        print(f"\\n{doc_id}:")
        for i, section in enumerate(doc_sections):
            print(f"  Block {i+1} (lines {section.start_line}-{section.end_line}):")
            print(f"  {section.content[:100]}...")
    """
    
    result = analyzer.analyze_with_rlm(query)
    print(result)
    
    # Demonstrate semantic analysis
    print("\n3. Semantic Analysis:")
    query = """
    # Perform semantic analysis on all documents
    analysis = semantic_analysis()
    print(f"Semantic analysis results:")
    for doc_id, doc_analysis in analysis.items():
        print(f"\\n{doc_id}:")
        print(f"  Total words: {doc_analysis['total_words']}")
        print(f"  Unique words: {doc_analysis['unique_words']}")
        print(f"  Vocabulary richness: {doc_analysis['vocabulary_richness']:.3f}")
        print(f"  Top words: {list(doc_analysis['filtered_top_words'].keys())[:5]}")
    """
    
    result = analyzer.analyze_with_rlm(query)
    print(result)

def demonstrate_dynamic_tools():
    """Demonstrate dynamic tool creation capabilities."""
    print("\n" + "="*80)
    print("DYNAMIC TOOL CREATION DEMONSTRATION")
    print("="*80)
    
    # Create sample documents
    sample_docs = create_sample_documents()
    
    # Initialize analyzer
    config = get_config()
    analyzer = RLMDocumentAnalyzer(config=config)
    
    # Load documents
    analyzer.documents = sample_docs
    analyzer._update_analyzer()
    
    # Demonstrate custom tool creation
    print("1. Custom Tool Creation:")
    query = """
    # Create a custom tool to extract email addresses
    success = create_custom_tool("extract_emails", 
        "def extract_emails(text): return re.findall(r'\\\\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\\\\.[A-Z|a-z]{2,}\\\\b', text)",
        "Extract email addresses from text")
    print(f"Custom tool creation: {success}")
    
    if success:
        # Use the custom tool on all documents
        print("\\nEmail addresses found:")
        for doc_id, content in documents.items():
            emails = execute_tool('extract_emails', content)
            if emails:
                print(f"  {doc_id}: {emails}")
    """
    
    result = analyzer.analyze_with_rlm(query)
    print(result)
    
    # Demonstrate pipeline creation
    print("\n2. Text Processing Pipeline:")
    query = """
    # Create a pipeline to extract and analyze code blocks
    pipeline_steps = [
        {"type": "search", "params": {"pattern": "```python", "context_lines": 0}},
        {"type": "extract", "params": {"start_pattern": "```python", "end_pattern": "```"}},
        {"type": "filter", "params": {"filter_type": "length", "min_length": 50}}
    ]
    
    success = create_text_pipeline("extract_code_blocks", pipeline_steps, "Extract and filter Python code blocks")
    print(f"Pipeline creation: {success}")
    
    if success:
        # Use the pipeline on the markdown document
        markdown_content = documents.get('markdown_doc.md', '')
        if markdown_content:
            result = execute_tool('extract_code_blocks', markdown_content)
            print(f"\\nPipeline result: {len(result)} code blocks extracted")
            for i, block in enumerate(result):
                print(f"  Block {i+1}: {str(block)[:100]}...")
    """
    
    result = analyzer.analyze_with_rlm(query)
    print(result)

def demonstrate_comprehensive_analysis():
    """Demonstrate comprehensive document analysis."""
    print("\n" + "="*80)
    print("COMPREHENSIVE DOCUMENT ANALYSIS DEMONSTRATION")
    print("="*80)
    
    # Create sample documents
    sample_docs = create_sample_documents()
    
    # Initialize analyzer
    config = get_config()
    analyzer = RLMDocumentAnalyzer(config=config)
    
    # Load documents
    analyzer.documents = sample_docs
    analyzer._update_analyzer()
    
    # Comprehensive analysis query
    query = """
    # Comprehensive analysis of all documents
    
    # 1. Get available tools
    tools = get_available_tools()
    print(f"Available tools: {list(tools.keys())}")
    
    # 2. Analyze structure of all documents
    structure = analyze_structure()
    print(f"\\nDocument Structure Analysis:")
    for doc_id, doc_structure in structure.items():
        print(f"\\n{doc_id}:")
        print(f"  Total lines: {doc_structure['total_lines']}")
        print(f"  Non-empty lines: {doc_structure['non_empty_lines']}")
        print(f"  Code blocks: {doc_structure['code_analysis']['markdown_blocks']}")
        print(f"  Headers: {doc_structure['header_analysis']['markdown_headers']}")
        print(f"  Lists: {doc_structure['list_analysis']['total_list_items']}")
    
    # 3. Find patterns across documents
    patterns = [r'\\\\b\\\\w+@\\\\w+\\\\.\\\\w+\\\\b', r'\\\\b\\\\d{3}-\\\\d{3}-\\\\d{4}\\\\b', r'https?://[^\\\\s]+']
    pattern_results = find_patterns(patterns)
    print(f"\\nPattern Analysis:")
    for doc_id, doc_patterns in pattern_results.items():
        print(f"\\n{doc_id}:")
        for pattern, matches in doc_patterns.items():
            print(f"  {pattern}: {len(matches)} matches")
    
    # 4. Create a custom analysis tool
    analysis_code = '''
def analyze_document_complexity(text):
    """Analyze document complexity."""
    lines = text.split('\\\\n')
    non_empty_lines = [line for line in lines if line.strip()]
    
    # Count different elements
    code_blocks = len(re.findall(r'```', text)) // 2
    functions = len(re.findall(r'def \\\\w+', text))
    classes = len(re.findall(r'class \\\\w+', text))
    comments = len(re.findall(r'#', text))
    
    # Calculate complexity score
    complexity_score = (functions * 2 + classes * 3 + code_blocks * 1.5 + comments * 0.5) / len(non_empty_lines) if non_empty_lines else 0
    
    return {
        'total_lines': len(lines),
        'non_empty_lines': len(non_empty_lines),
        'code_blocks': code_blocks,
        'functions': functions,
        'classes': classes,
        'comments': comments,
        'complexity_score': complexity_score
    }
'''
    
    success = create_custom_tool("analyze_complexity", analysis_code, "Analyze document complexity")
    print(f"\\nCustom complexity analysis tool: {success}")
    
    if success:
        print("\\nDocument Complexity Analysis:")
        for doc_id, content in documents.items():
            complexity = execute_tool('analyze_complexity', content)
            print(f"\\n{doc_id}:")
            print(f"  Complexity score: {complexity['complexity_score']:.2f}")
            print(f"  Functions: {complexity['functions']}")
            print(f"  Classes: {complexity['classes']}")
            print(f"  Code blocks: {complexity['code_blocks']}")
    """
    
    result = analyzer.analyze_with_rlm(query)
    print(result)

def main():
    """Main demonstration function."""
    print("ADVANCED TEXT PROCESSING CAPABILITIES DEMONSTRATION")
    print("This example shows the comprehensive text processing capabilities")
    print("that are now available to the LLM in the RLM system.")
    print()
    
    try:
        # Run demonstrations
        demonstrate_basic_analysis()
        demonstrate_advanced_processing()
        demonstrate_dynamic_tools()
        demonstrate_comprehensive_analysis()
        
        print("\n" + "="*80)
        print("DEMONSTRATION COMPLETED SUCCESSFULLY!")
        print("="*80)
        print()
        print("The RLM system now has comprehensive text processing capabilities:")
        print("✓ Advanced search with context extraction")
        print("✓ Section extraction and text slicing")
        print("✓ Structure and semantic analysis")
        print("✓ Pattern frequency analysis")
        print("✓ Dynamic tool creation")
        print("✓ Text processing pipelines")
        print("✓ Custom function execution")
        print("✓ Multi-document processing")
        print()
        print("The LLM can now create any sort of text processing tools it needs")
        print("to slice and dice context around found matches and answer all")
        print("queries about documents, whether semantic or syntactic.")
        
    except Exception as e:
        print(f"Demonstration failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
