# RLM Document Analyzer - System Overview

## 🎯 Project Summary

This project implements a **Recursive Language Model (RLM)** system based on the research paper by Alex Zhang. The system allows language models to recursively call themselves through a Python REPL environment, enabling analysis of essentially unbounded context lengths and processing of 1000+ documents efficiently.

## 🏗️ Architecture

### Core Components

1. **Python REPL Environment** (`python_repl.py`)
   - Secure code execution environment
   - Variable storage and management
   - Built-in utility functions (grep, regex, text processing)
   - Safety checks and sandboxing

2. **RLM Wrapper** (`rlm_wrapper.py`)
   - Manages recursive LLM calls
   - Handles conversation flow and context management
   - Integrates with Gemini API
   - Supports FINAL() and FINAL_VAR() response types

3. **Document Analyzer** (`document_analyzer.py`)
   - Loads documents from directories, files, or JSON
   - Provides comprehensive analysis utilities
   - Handles large document collections (1000+ documents)
   - Generates statistics and reports

4. **Configuration Management** (`config.py`)
   - Robust configuration handling
   - Environment variable support
   - Validation and defaults
   - JSON configuration files

5. **Utilities** (`utils.py`)
   - Common utility functions
   - Error handling and validation
   - Performance monitoring
   - File operations and formatting

6. **Main Entry Point** (`main.py`)
   - Command-line interface
   - Interactive mode
   - Batch processing capabilities
   - Comprehensive logging

## 🚀 Key Features

### RLM Framework Implementation
- **Recursive Calls**: LLMs can call themselves for sub-problems
- **Context Management**: Documents stored as variables in REPL
- **Code Execution**: LLMs can write and execute Python code
- **Flexible Responses**: FINAL() and FINAL_VAR() response types

### Document Processing
- **Multiple Formats**: Supports .txt, .md, .py, .js, .json, .xml, .csv, etc.
- **Large Scale**: Handles 1000+ documents efficiently
- **Search & Analysis**: Pattern matching, keyword extraction, similarity analysis
- **Statistics**: Comprehensive document collection statistics

### Robustness Features
- **Error Handling**: Comprehensive error handling and recovery
- **Security**: Sandboxed code execution environment
- **Logging**: Detailed logging and monitoring
- **Configuration**: Flexible configuration management
- **Testing**: Comprehensive test suite

## 📁 File Structure

```
budget-rlm-v2/
├── main.py                 # Main entry point and CLI
├── python_repl.py          # Python REPL environment
├── rlm_wrapper.py          # RLM framework implementation
├── document_analyzer.py    # Document loading and analysis
├── config.py              # Configuration management
├── utils.py               # Utility functions
├── test_system.py         # Test suite
├── example_usage.py       # Usage examples
├── requirements.txt       # Dependencies
├── README.md             # Documentation
└── SYSTEM_OVERVIEW.md    # This file
```

## 🔧 Installation & Setup

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Set API Key**:
   ```bash
   export GEMINI_API_KEY="your_api_key_here"
   ```

3. **Run Tests**:
   ```bash
   python test_system.py
   ```

## 💡 Usage Examples

### Basic Analysis
```bash
python main.py --documents /path/to/documents --query "What are the main topics?"
```

### Interactive Mode
```bash
python main.py --documents /path/to/documents --interactive
```

### Code Analysis
```bash
python main.py --documents /path/to/code --query "Find all TODO comments"
```

### Batch Processing
```python
from main import RLMDocumentAnalyzer

analyzer = RLMDocumentAnalyzer()
analyzer.load_documents_from_directory("/path/to/documents")

result = analyzer.analyze_with_rlm("Analyze these documents comprehensively")
```

## 🧪 Testing

The system includes comprehensive tests:
- Python REPL functionality
- Document loading and processing
- Analysis capabilities
- Configuration management
- Integration between components

Run tests with:
```bash
python test_system.py
```

## 🔒 Security Features

- **Sandboxed Execution**: Code runs in restricted environment
- **Import Restrictions**: Only safe modules allowed
- **File Operations**: Disabled by default for security
- **Input Validation**: Comprehensive input validation
- **Error Isolation**: Errors don't crash the system

## 📊 Performance

- **Scalability**: Handles 1000+ documents
- **Memory Efficient**: Streaming and chunking for large files
- **Caching**: Analysis results caching
- **Parallel Processing**: Support for concurrent operations
- **Resource Monitoring**: Built-in performance monitoring

## 🔄 RLM Workflow

1. **Context Loading**: Documents loaded into REPL as variables
2. **Query Processing**: Root LLM receives user query
3. **Code Execution**: LLM writes Python code to analyze documents
4. **Recursive Calls**: LLM can call itself for sub-problems
5. **Final Answer**: LLM provides final answer using FINAL() tags

## 🎯 Research Implementation

This implementation follows the RLM research paper principles:
- **Context-Centric View**: Focus on context management rather than problem decomposition
- **REPL Environment**: Python notebook-like environment for LLM interaction
- **Recursive Calls**: LLMs can recursively call themselves
- **Unbounded Context**: Process essentially unlimited context lengths
- **Tool Integration**: Built-in tools like grep, regex, and text processing

## 🚀 Future Enhancements

- **Additional LLM Providers**: Support for OpenAI, Anthropic, etc.
- **Advanced NLP**: Integration with spaCy, NLTK, transformers
- **Visualization**: Document analysis visualization tools
- **API Server**: REST API for remote access
- **Web Interface**: Browser-based interface
- **Performance Optimization**: GPU acceleration and caching

## 📚 Documentation

- **README.md**: Complete usage documentation
- **Example Usage**: Comprehensive examples in `example_usage.py`
- **API Documentation**: Inline documentation in all modules
- **Configuration Guide**: Configuration options and examples

## ✅ Quality Assurance

- **Comprehensive Testing**: All components tested
- **Error Handling**: Robust error handling throughout
- **Logging**: Detailed logging for debugging
- **Documentation**: Extensive documentation
- **Code Quality**: Clean, well-structured code
- **Security**: Security-first design

This system provides a robust, scalable, and secure implementation of Recursive Language Models for document analysis, ready for production use with large document collections.
