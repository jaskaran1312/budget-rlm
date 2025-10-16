# RLM Document Analyzer

A robust implementation of Recursive Language Models (RLMs) for analyzing large collections of documents. This system allows language models to recursively call themselves through a Python REPL environment, enabling analysis of essentially unbounded context lengths.

## Features

- **Recursive Language Model Framework**: Implements the RLM architecture from the research paper
- **Python REPL Environment**: Secure code execution environment for LLM interactions
- **Document Analysis**: Process and analyze 1000+ documents efficiently
- **Gemini Integration**: Uses Google's Gemini API for LLM calls
- **Robust Error Handling**: Comprehensive error handling and logging
- **Interactive Mode**: Command-line interface for interactive document analysis
- **Batch Processing**: Analyze multiple documents in batch operations

## Installation

1. Clone or download this repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Set up your Gemini API key:
   
   **Option A: Using .env file (Recommended)**
   ```bash
   python setup_env.py
   # Edit the created .env file and add your actual API key
   ```
   
   **Option B: Environment variable**
   ```bash
   export GEMINI_API_KEY="your_api_key_here"
   ```

## Quick Start

### Setup (First Time)
```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Set up environment file
python setup_env.py

# 3. Edit .env file and add your GEMINI_API_KEY
# Get your API key from: https://ai.google.dev/

# 4. Test your setup
python quick_start.py
```

### Basic Usage

```bash
# Analyze documents from default Documents directory
python main.py --query "What are the main topics in these documents?"

# Interactive mode with default Documents directory
python main.py --interactive

# Analyze documents from a specific directory
python main.py --documents /path/to/documents --query "What are the main topics in these documents?"

# Analyze specific files
python main.py --files file1.txt file2.md file3.py --query "Find all TODO comments"
```

### Interactive Mode

```bash
# Interactive mode with default Documents directory
python main.py --interactive

# Interactive mode with specific directory
python main.py --documents /path/to/documents --interactive
```

Available commands in interactive mode:
- `help`: Show available commands
- `stats`: Show document collection statistics
- `summary`: Generate summary report
- `search <pattern>`: Search for patterns across documents
- `<query>`: Any other input is treated as an RLM analysis query

## Architecture

### Core Components

1. **Python REPL Environment** (`python_repl.py`)
   - Secure code execution environment
   - Variable storage and management
   - Built-in utility functions for document analysis

2. **RLM Wrapper** (`rlm_wrapper.py`)
   - Manages recursive LLM calls
   - Handles conversation flow and context management
   - Integrates with Gemini API

3. **Document Analyzer** (`document_analyzer.py`)
   - Loads documents from various sources
   - Provides analysis utilities and statistics
   - Handles large document collections

4. **Configuration Management** (`config.py`)
   - Robust configuration handling
   - Environment variable support
   - Validation and defaults

5. **Utilities** (`utils.py`)
   - Common utility functions
   - Error handling and validation
   - Performance monitoring

## How RLM Works

The Recursive Language Model framework works as follows:

1. **Context Loading**: Documents are loaded into a Python REPL environment as variables
2. **Query Processing**: The root LLM receives the user query and can interact with the REPL
3. **Code Execution**: The LLM can write and execute Python code to analyze the documents
4. **Recursive Calls**: The LLM can make recursive calls to itself for sub-problems
5. **Final Answer**: The LLM provides a final answer using `FINAL()` or `FINAL_VAR()` tags

### Example RLM Interaction

```
User: "What are the main topics discussed in these documents?"

RLM: I'll analyze the documents to identify the main topics.

```python
# First, let me get an overview of the documents
print(f"Total documents: {len(documents)}")
print(f"Total characters: {sum(len(doc) for doc in documents.values())}")

# Let me search for common topic indicators
import re
from collections import Counter

all_text = ' '.join(documents.values()).lower()
words = re.findall(r'\b[a-zA-Z]{4,}\b', all_text)
word_counts = Counter(words)

# Get top keywords
top_keywords = word_counts.most_common(20)
print("Top keywords:", top_keywords)
```

Code executed successfully. Output:
Total documents: 150
Total characters: 2,450,000
Top keywords: [('machine', 1250), ('learning', 980), ('data', 850), ...]

```python
# Now let me analyze specific documents for topic patterns
topics = {}
for doc_id, content in documents.items():
    # Look for topic indicators
    if 'machine learning' in content.lower():
        topics[doc_id] = 'Machine Learning'
    elif 'data analysis' in content.lower():
        topics[doc_id] = 'Data Analysis'
    # ... more topic detection logic

topic_summary = Counter(topics.values())
print("Topic distribution:", topic_summary)
```

FINAL(The main topics in these documents are: Machine Learning (45 documents), Data Analysis (32 documents), Software Development (28 documents), Research Papers (25 documents), and Documentation (20 documents). The documents primarily focus on technical subjects with a strong emphasis on machine learning and data analysis methodologies.)
```

## Configuration

Create a `rlm_config.json` file to customize the system:

```json
{
  "max_iterations": 10,
  "max_recursion_depth": 3,
  "model_name": "gemini-2.5-flash",
  "max_output_length": 10000,
  "log_level": "INFO",
  "max_document_size": 10485760
}
```

Or use environment variables:
- `GEMINI_API_KEY`: Your Gemini API key
- `RLM_MAX_ITERATIONS`: Maximum RLM iterations
- `RLM_MAX_RECURSION_DEPTH`: Maximum recursion depth
- `RLM_LOG_LEVEL`: Logging level

## Supported Document Types

The system supports various document types:
- Text files (`.txt`, `.md`)
- Code files (`.py`, `.js`, `.ts`, `.html`, `.css`)
- Data files (`.json`, `.xml`, `.csv`, `.yaml`)
- Configuration files (`.ini`, `.cfg`)
- And more...

## Advanced Usage

### Custom Analysis Queries

```bash
# Find all TODO comments across code files (using default Documents directory)
python main.py --query "Find all TODO and FIXME comments in the codebase"

# Analyze error patterns in log files
python main.py --documents /path/to/logs --query "What are the most common error patterns in these logs?"

# Summarize research papers
python main.py --documents /path/to/papers --query "Provide a summary of the key findings from these research papers"
```

### Batch Processing

```python
from main import RLMDocumentAnalyzer

analyzer = RLMDocumentAnalyzer()

# Load documents
analyzer.load_documents_from_directory("/path/to/documents")

# Run multiple analyses
queries = [
    "What are the main topics?",
    "Find all code examples",
    "Summarize the key findings"
]

for query in queries:
    result = analyzer.analyze_with_rlm(query)
    print(f"Query: {query}")
    print(f"Result: {result}\n")
```

## Performance Considerations

- **Document Size**: Large documents (>10MB) may impact performance
- **Recursion Depth**: Higher recursion depths increase processing time
- **API Limits**: Be mindful of Gemini API rate limits
- **Memory Usage**: Large document collections require sufficient memory

## Error Handling

The system includes comprehensive error handling:
- API failures with retry logic
- Invalid document formats
- Code execution errors in REPL
- Configuration validation
- Graceful degradation

## Logging

Enable detailed logging:
```bash
python main.py --verbose --documents /path/to/documents --query "your query"
```

Logs include:
- RLM call history
- Code execution results
- Performance metrics
- Error details

## Contributing

This implementation is based on the research paper "Recursive Language Models" by Alex Zhang. The system is designed to be extensible and robust for real-world document analysis tasks.

## License

This project is provided as-is for educational and research purposes.

## References

- [Recursive Language Models Research Paper](https://alexzhang13.github.io/blog/2025/rlm/)
- [Google Gemini API Documentation](https://ai.google.dev/docs)
