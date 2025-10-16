# Advanced Text Processing Capabilities

The RLM system now includes comprehensive text processing capabilities that allow the LLM to create any sort of text tools it wants to slice and dice context around found matches and answer all queries about documents, whether semantic or syntactic.

## Overview

The system provides a powerful text processing toolkit that includes:

- **Advanced Search**: Context-aware pattern matching with surrounding text extraction
- **Section Extraction**: Extract text sections between delimiters
- **Text Slicing**: Slice text by lines, characters, or patterns
- **Structure Analysis**: Analyze document structure (headers, lists, code blocks, etc.)
- **Semantic Analysis**: Word frequency, vocabulary richness, key phrases
- **Pattern Frequency**: Count common patterns (emails, phones, URLs, etc.)
- **Dynamic Tool Creation**: Create custom text processing functions on the fly
- **Text Processing Pipelines**: Chain multiple processing steps together
- **Multi-document Processing**: Process multiple documents simultaneously

## Core Components

### 1. TextProcessor Class

The core text processing engine with advanced capabilities:

```python
from text_toolkit import TextProcessor

processor = TextProcessor()

# Advanced search with context
matches = processor.advanced_search(text, pattern, context_lines=3)

# Extract sections
sections = processor.extract_sections(text, start_pattern, end_pattern)

# Analyze structure
structure = processor.analyze_text_structure(text)

# Semantic analysis
semantic = processor.semantic_analysis(text)
```

### 2. DocumentTextProcessor Class

High-level interface for processing multiple documents:

```python
from text_toolkit import DocumentTextProcessor

processor = DocumentTextProcessor(documents)

# Search across all documents
results = processor.execute_tool('advanced_search_docs', pattern, context_lines=3)

# Extract sections from all documents
sections = processor.execute_tool('extract_sections_docs', start_pattern, end_pattern)

# Analyze structure of all documents
structure = processor.execute_tool('analyze_structure_docs')
```

## Available Functions in RLM Environment

### Basic Text Processing
- `grep(pattern, text)`: Search for patterns in text using regex
- `count_occurrences(pattern, text)`: Count pattern occurrences
- `extract_sections(text, start, end)`: Extract sections between patterns
- `summarize_text(text, max_length)`: Create text summaries

### Advanced Text Processing
- `advanced_search(pattern, context_lines=3, case_sensitive=False, doc_ids=None)`: Search across documents with context extraction
- `extract_sections_docs(start_pattern, end_pattern, include_delimiters=False, doc_ids=None)`: Extract sections between patterns from documents
- `slice_docs(start_pattern=None, end_pattern=None, start_line=None, end_line=None, start_char=None, end_char=None, doc_ids=None)`: Slice documents using various methods
- `analyze_structure(doc_ids=None)`: Analyze text structure of documents
- `semantic_analysis(doc_ids=None)`: Perform semantic analysis on documents
- `find_patterns(patterns, case_sensitive=False, doc_ids=None)`: Find multiple patterns in documents

### Single Text Analysis
- `analyze_single_text(text, analysis_type='structure')`: Analyze a single text string
- `search_single_text(text, pattern, context_lines=3, case_sensitive=False)`: Search a single text with context
- `extract_sections_single(text, start_pattern, end_pattern, include_delimiters=False)`: Extract sections from a single text
- `slice_single_text(text, start_pattern=None, end_pattern=None, start_line=None, end_line=None, start_char=None, end_char=None)`: Slice a single text

### Dynamic Tool Creation
- `create_custom_tool(name, code, description="")`: Create a custom text processing tool
- `create_text_pipeline(name, steps, description="")`: Create a text processing pipeline
- `get_available_tools()`: Get list of available text processing tools
- `execute_tool(name, *args, **kwargs)`: Execute a text processing tool

## Dynamic Tool Creation Examples

### 1. Custom Function Creation

```python
# Create a custom email extractor
success = create_custom_tool("extract_emails", 
    "def extract_emails(text): return re.findall(r'\\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\\.[A-Z|a-z]{2,}\\b', text)",
    "Extract email addresses from text")

# Use the custom tool
emails = execute_tool('extract_emails', document_text)
```

### 2. Text Processing Pipeline

```python
# Create a pipeline to extract and filter code blocks
pipeline_steps = [
    {"type": "search", "params": {"pattern": "```python", "context_lines": 0}},
    {"type": "extract", "params": {"start_pattern": "```python", "end_pattern": "```"}},
    {"type": "filter", "params": {"filter_type": "length", "min_length": 50}}
]

success = create_text_pipeline("extract_code_blocks", pipeline_steps, "Extract and filter Python code blocks")

# Use the pipeline
code_blocks = execute_tool('extract_code_blocks', markdown_text)
```

## Pipeline Step Types

### Search Step
```python
{"type": "search", "params": {"pattern": r"def \\w+", "context_lines": 3}}
```

### Extract Step
```python
{"type": "extract", "params": {"start_pattern": "def ", "end_pattern": "return"}}
```

### Slice Step
```python
{"type": "slice", "params": {"start_line": 10, "end_line": 20}}
```

### Filter Step
```python
{"type": "filter", "params": {"filter_type": "length", "min_length": 100}}
```

### Transform Step
```python
{"type": "transform", "params": {"code": "result = text.upper()"}}
```

### Custom Step
```python
{"type": "custom", "params": {"function": "my_custom_function", "param1": "value1"}}
```

## Text Analysis Capabilities

### Structure Analysis
- Total lines and non-empty lines
- Indentation analysis
- Paragraph analysis
- Code block detection (markdown and indented)
- Header analysis (markdown and common patterns)
- List analysis (bullet and numbered)
- Table analysis
- Link analysis

### Semantic Analysis
- Word frequency analysis
- Vocabulary richness calculation
- Key phrase extraction
- Word length distribution
- Sentence analysis
- Stop word filtering

### Pattern Frequency Analysis
- Email addresses
- Phone numbers
- URLs
- IP addresses
- Dates
- Times
- Currency amounts
- Percentages
- Version numbers

## Usage Examples

### 1. Find All Function Definitions with Context

```python
# Search for function definitions with 3 lines of context
results = advanced_search(r"def \\w+", context_lines=3)

for doc_id, matches in results.items():
    print(f"Document: {doc_id}")
    for match in matches:
        print(f"Line {match.line_number}: {match.text}")
        print(f"Context: {match.context_before}...{match.context_after}")
```

### 2. Extract Code Blocks from Markdown

```python
# Extract Python code blocks from markdown documents
sections = extract_sections_docs("```python", "```", include_delimiters=True)

for doc_id, doc_sections in sections.items():
    print(f"Document: {doc_id}")
    for section in doc_sections:
        print(f"Code block (lines {section.start_line}-{section.end_line}):")
        print(section.content)
```

### 3. Analyze Document Complexity

```python
# Create a custom complexity analyzer
complexity_code = '''
def analyze_complexity(text):
    functions = len(re.findall(r'def \\\\w+', text))
    classes = len(re.findall(r'class \\\\w+', text))
    lines = len(text.split('\\\\n'))
    return {'functions': functions, 'classes': classes, 'lines': lines}
'''

create_custom_tool("analyze_complexity", complexity_code, "Analyze document complexity")

# Use it on all documents
for doc_id, content in documents.items():
    complexity = execute_tool('analyze_complexity', content)
    print(f"{doc_id}: {complexity['functions']} functions, {complexity['classes']} classes")
```

### 4. Multi-Pattern Search

```python
# Search for multiple patterns across documents
patterns = [r'\\\\b\\\\w+@\\\\w+\\\\.\\\\w+\\\\b', r'\\\\b\\\\d{3}-\\\\d{3}-\\\\d{4}\\\\b', r'https?://[^\\\\s]+']
results = find_patterns(patterns)

for doc_id, doc_patterns in results.items():
    print(f"Document: {doc_id}")
    for pattern, matches in doc_patterns.items():
        print(f"  {pattern}: {len(matches)} matches")
```

## Integration with RLM System

The text processing capabilities are fully integrated into the RLM system:

1. **REPL Environment**: All functions are available in the Python REPL environment
2. **System Instructions**: The LLM is informed about all available capabilities
3. **Dynamic Creation**: The LLM can create custom tools and pipelines on the fly
4. **Error Handling**: Robust error handling and recovery mechanisms
5. **Performance**: Optimized for processing large document collections

## Benefits

1. **Comprehensive Analysis**: Can analyze documents from multiple angles (syntactic and semantic)
2. **Context Awareness**: Maintains context around matches for better understanding
3. **Flexibility**: Can create custom tools for specific analysis needs
4. **Scalability**: Efficiently processes large document collections
5. **Integration**: Seamlessly integrated with the existing RLM framework

## Conclusion

The advanced text processing capabilities give the LLM the ability to create any sort of text tools it wants to slice and dice context around found matches and answer all queries about documents, whether semantic or syntactic. This makes the RLM system a powerful tool for comprehensive document analysis and understanding.
