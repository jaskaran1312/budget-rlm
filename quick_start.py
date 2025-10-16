#!/usr/bin/env python3
"""
Quick Start Example for RLM Document Analyzer
Demonstrates how to use the system with .env file configuration
"""

from main import RLMDocumentAnalyzer
from pathlib import Path


def quick_example():
    """Run a quick example using the .env file configuration."""
    print("RLM Document Analyzer - Quick Start Example")
    print("=" * 50)
    
    # Initialize the analyzer (will automatically use .env file)
    print("Initializing RLM Document Analyzer...")
    analyzer = RLMDocumentAnalyzer()
    print("✅ Analyzer initialized with API key from .env file")
    
    # Check if there are any documents in the Documents folder
    documents_dir = Path("Documents")
    if documents_dir.exists() and any(documents_dir.iterdir()):
        print(f"\nFound documents in {documents_dir} directory (default)")
        
        # Load documents
        print("Loading documents...")
        analyzer.load_documents_from_directory(str(documents_dir))
        print(f"✅ Loaded {len(analyzer.documents)} documents")
        
        # Get basic statistics
        print("\nDocument Statistics:")
        stats = analyzer.get_document_stats()
        print(f"- Total documents: {stats['total_documents']}")
        print(f"- Total characters: {stats['total_characters']:,}")
        print(f"- Total words: {stats['total_words']:,}")
        print(f"- Average document length: {stats['avg_doc_length']:.0f} characters")
        
        # Run a simple analysis
        print("\nRunning analysis...")
        query = "What are the main topics and key themes in these documents?"
        result = analyzer.analyze_with_rlm(query)
        
        print(f"\nQuery: {query}")
        print("=" * 50)
        print("Analysis Result:")
        print("=" * 50)
        print(result)
        print("=" * 50)
        
    else:
        print(f"\nNo documents found in {documents_dir}")
        print("You can:")
        print("1. Add documents to the Documents/ folder")
        print("2. Use --documents flag to specify a different directory")
        print("3. Use --files flag to analyze specific files")
        
        # Show usage examples
        print("\nUsage Examples:")
        print("python main.py --documents /path/to/documents --query 'Analyze these documents'")
        print("python main.py --interactive --documents /path/to/documents")
        print("python main.py --files file1.txt file2.md --query 'Find key insights'")


if __name__ == "__main__":
    try:
        quick_example()
    except Exception as e:
        print(f"Error: {e}")
        print("\nMake sure you have:")
        print("1. Set up your .env file with GEMINI_API_KEY")
        print("2. Installed dependencies: pip install -r requirements.txt")
        print("3. Run: python setup_env.py to verify your setup")
