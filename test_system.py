#!/usr/bin/env python3
"""
Test script for the RLM Document Analyzer system.
Tests basic functionality without requiring API calls.
"""

import os
import tempfile
import shutil
from pathlib import Path
from python_repl import SecureREPL, create_repl_environment
from document_analyzer import DocumentLoader, DocumentAnalyzer
from rlm_wrapper import RLMConfig
from utils import Timer, validate_file_path, format_file_size


def test_python_repl():
    """Test the Python REPL environment."""
    print("Testing Python REPL Environment...")
    
    repl, analyzer = create_repl_environment()
    
    # Test basic code execution
    success, output, error = repl.execute_code("x = 42; print(f'x = {x}')")
    assert success, f"Code execution failed: {error}"
    assert "x = 42" in output, f"Expected 'x = 42' in output, got: {output}"
    
    # Test variable storage
    repl.set_variable("test_var", "hello world")
    assert repl.get_variable("test_var") == "hello world"
    
    # Test utility functions
    success, output, error = repl.execute_code("result = grep('hello', 'hello world')")
    assert success, f"Grep function failed: {error}"
    
    print("✓ Python REPL tests passed")


def test_document_loader():
    """Test document loading functionality."""
    print("Testing Document Loader...")
    
    # Create temporary test files
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Create test documents
        test_files = {
            "test1.txt": "This is a test document with some content.",
            "test2.md": "# Test Markdown\n\nThis is a markdown file.",
            "test3.py": "# Test Python file\ndef hello():\n    return 'world'"
        }
        
        for filename, content in test_files.items():
            (temp_path / filename).write_text(content)
        
        # Test loading
        loader = DocumentLoader()
        documents, metadata = loader.load_from_directory(str(temp_path))
        
        assert len(documents) == 3, f"Expected 3 documents, got {len(documents)}"
        assert "test1.txt" in documents, "test1.txt not found in loaded documents"
        assert "test2.md" in documents, "test2.md not found in loaded documents"
        assert "test3.py" in documents, "test3.py not found in loaded documents"
        
        # Test metadata
        assert len(metadata) == 3, f"Expected 3 metadata entries, got {len(metadata)}"
        assert metadata["test1.txt"].file_type == ".txt"
        assert metadata["test2.md"].file_type == ".md"
        assert metadata["test3.py"].file_type == ".py"
    
    print("✓ Document Loader tests passed")


def test_document_analyzer():
    """Test document analysis functionality."""
    print("Testing Document Analyzer...")
    
    # Create test documents
    test_documents = {
        "doc1.txt": "Machine learning is a subset of artificial intelligence.",
        "doc2.txt": "Deep learning uses neural networks for pattern recognition.",
        "doc3.txt": "Natural language processing helps computers understand text."
    }
    
    from document_analyzer import DocumentMetadata
    test_metadata = {
        doc_id: DocumentMetadata(doc_id, doc_id, len(content), 0, 0, "", ".txt")
        for doc_id, content in test_documents.items()
    }
    
    # Test analyzer
    analyzer = DocumentAnalyzer(test_documents, test_metadata)
    
    # Test statistics
    stats = analyzer.get_document_stats()
    assert stats['total_documents'] == 3
    assert stats['total_characters'] > 0
    
    # Test search
    results = analyzer.search_across_documents("machine learning")
    assert len(results) > 0, "Search should find matches"
    
    # Test keyword extraction
    keywords = analyzer.extract_keywords(5)
    assert len(keywords) > 0, "Should extract keywords"
    
    print("✓ Document Analyzer tests passed")


def test_utilities():
    """Test utility functions."""
    print("Testing Utilities...")
    
    # Test file validation
    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        temp_path = temp_file.name
        temp_file.write(b"test content")
    
    try:
        assert validate_file_path(temp_path), "File validation should pass"
        assert not validate_file_path("/nonexistent/file.txt"), "File validation should fail"
    finally:
        os.unlink(temp_path)
    
    # Test formatting functions
    assert format_file_size(1024) == "1.0 KB"
    assert format_file_size(1048576) == "1.0 MB"
    
    # Test timer
    with Timer("test operation"):
        import time
        time.sleep(0.1)
    
    print("✓ Utility tests passed")


def test_configuration():
    """Test configuration management."""
    print("Testing Configuration...")
    
    from config import RLMConfig, ConfigManager
    
    # Test default config
    config = RLMConfig()
    assert config.max_iterations == 10
    assert config.max_recursion_depth == 3
    assert config.model_name == "gemini-2.5-flash"
    
    # Test config manager
    config_manager = ConfigManager()
    config = config_manager.get_config()
    assert isinstance(config, RLMConfig)
    
    # Test validation
    issues = config_manager.validate_config()
    # Should have at least one issue (missing API key)
    assert len(issues) > 0, "Should have validation issues without API key"
    
    print("✓ Configuration tests passed")


def test_integration():
    """Test integration between components."""
    print("Testing Integration...")
    
    # Create test documents
    test_documents = {
        "sample.txt": "This is a sample document for testing the RLM system."
    }
    
    # Test REPL with documents
    repl, analyzer = create_repl_environment()
    analyzer.load_documents(test_documents)
    
    # Test that documents are available in REPL
    assert repl.get_variable("documents") == test_documents
    assert repl.get_variable("num_documents") == 1
    
    # Test code execution with document access
    success, output, error = repl.execute_code("print(f'Loaded {num_documents} documents')")
    assert success, f"Code execution failed: {error}"
    assert "Loaded 1 documents" in output
    
    print("✓ Integration tests passed")


def run_all_tests():
    """Run all tests."""
    print("Running RLM Document Analyzer Tests")
    print("=" * 50)
    
    try:
        test_python_repl()
        test_document_loader()
        test_document_analyzer()
        test_utilities()
        test_configuration()
        test_integration()
        
        print("\n" + "=" * 50)
        print("✓ All tests passed successfully!")
        print("=" * 50)
        
        return True
        
    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)
