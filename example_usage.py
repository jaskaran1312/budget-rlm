#!/usr/bin/env python3
"""
Example usage of the RLM Document Analyzer
Demonstrates various ways to use the system for document analysis.
"""

import os
import json
from pathlib import Path
from main import RLMDocumentAnalyzer
from rlm_wrapper import RLMConfig
from document_analyzer import DocumentLoader


def create_sample_documents():
    """Create sample documents for demonstration."""
    sample_dir = Path("sample_documents")
    sample_dir.mkdir(exist_ok=True)
    
    # Create sample documents
    documents = {
        "research_paper_1.txt": """
Machine Learning in Healthcare: A Comprehensive Review

Abstract:
This paper reviews the applications of machine learning in healthcare, focusing on diagnostic systems, treatment optimization, and patient monitoring. We analyze 150 recent studies and identify key trends in the field.

Introduction:
Machine learning has revolutionized healthcare by enabling automated diagnosis, personalized treatment plans, and predictive analytics. This review examines the current state of the art and future directions.

Key Findings:
1. Deep learning models achieve 95% accuracy in medical image analysis
2. Natural language processing improves clinical documentation
3. Predictive models reduce hospital readmission rates by 30%
4. Real-time monitoring systems enhance patient safety

Conclusion:
Machine learning continues to transform healthcare, with promising applications in precision medicine and automated clinical decision support.
""",
        
        "code_review.py": """
# TODO: Implement error handling for API calls
# FIXME: Memory leak in data processing pipeline

import requests
import json
from typing import Dict, List

class DataProcessor:
    def __init__(self, api_endpoint: str):
        self.api_endpoint = api_endpoint
        self.session = requests.Session()
    
    def fetch_data(self, query: str) -> Dict:
        \"\"\"Fetch data from API endpoint.\"\"\"
        try:
            response = self.session.get(f"{self.api_endpoint}/search", params={"q": query})
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            # TODO: Add proper error logging
            print(f"Error fetching data: {e}")
            return {}
    
    def process_data(self, data: List[Dict]) -> List[Dict]:
        \"\"\"Process and clean the data.\"\"\"
        processed = []
        for item in data:
            # FIXME: Handle missing fields gracefully
            if 'id' in item and 'name' in item:
                processed.append({
                    'id': item['id'],
                    'name': item['name'].strip(),
                    'processed': True
                })
        return processed

# TODO: Add unit tests for DataProcessor class
""",
        
        "project_notes.md": """
# Project Documentation

## Overview
This project implements a document analysis system using recursive language models.

## Features
- Document loading and preprocessing
- Text analysis and keyword extraction
- Machine learning model integration
- API endpoint for document queries

## TODO Items
- [ ] Implement caching for analysis results
- [ ] Add support for more document formats
- [ ] Optimize memory usage for large documents
- [ ] Add comprehensive error handling

## Technical Debt
- Refactor the document parser to use a more efficient algorithm
- Update the API documentation
- Add integration tests for the RLM system

## Performance Notes
- Current processing time: ~2 seconds per 1000 documents
- Memory usage: ~500MB for 10,000 documents
- API rate limit: 100 requests per minute
""",
        
        "error_logs.txt": """
2024-01-15 10:30:15 ERROR: Database connection failed - timeout after 30 seconds
2024-01-15 10:31:22 WARNING: High memory usage detected: 85% of available RAM
2024-01-15 10:32:45 ERROR: API endpoint /search returned 500 status code
2024-01-15 10:33:12 INFO: Successfully processed 150 documents
2024-01-15 10:34:01 ERROR: File not found: /data/missing_file.txt
2024-01-15 10:35:18 WARNING: Slow query detected: 5.2 seconds execution time
2024-01-15 10:36:33 ERROR: Authentication failed for user admin
2024-01-15 10:37:45 INFO: System backup completed successfully
2024-01-15 10:38:12 ERROR: Network timeout when connecting to external API
2024-01-15 10:39:01 WARNING: Disk space low: 15% remaining
""",
        
        "data_analysis.json": """
{
  "dataset_info": {
    "name": "Customer Behavior Analysis",
    "size": 50000,
    "features": ["age", "income", "purchase_history", "location"],
    "target": "churn_prediction"
  },
  "model_performance": {
    "accuracy": 0.87,
    "precision": 0.85,
    "recall": 0.82,
    "f1_score": 0.83
  },
  "key_insights": [
    "Customers aged 25-35 have highest churn rate",
    "Income level is the strongest predictor of retention",
    "Geographic location shows significant correlation with behavior"
  ],
  "recommendations": [
    "Focus retention efforts on younger demographic",
    "Implement income-based pricing strategies",
    "Develop location-specific marketing campaigns"
  ]
}
"""
    }
    
    # Write documents to files
    for filename, content in documents.items():
        file_path = sample_dir / filename
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
    
    print(f"Created {len(documents)} sample documents in {sample_dir}")
    return str(sample_dir)


def example_basic_analysis():
    """Example of basic document analysis."""
    print("\n" + "="*60)
    print("EXAMPLE 1: Basic Document Analysis")
    print("="*60)
    
    # Create sample documents
    sample_dir = create_sample_documents()
    
    # Initialize analyzer
    analyzer = RLMDocumentAnalyzer()
    
    # Load documents
    analyzer.load_documents_from_directory(sample_dir)
    
    # Run analysis
    query = "What are the main topics and key findings in these documents?"
    result = analyzer.analyze_with_rlm(query)
    
    print(f"Query: {query}")
    print(f"Result: {result}")
    
    # Clean up
    import shutil
    shutil.rmtree(sample_dir)


def example_code_analysis():
    """Example of analyzing code files."""
    print("\n" + "="*60)
    print("EXAMPLE 2: Code Analysis")
    print("="*60)
    
    sample_dir = create_sample_documents()
    analyzer = RLMDocumentAnalyzer()
    analyzer.load_documents_from_directory(sample_dir)
    
    # Analyze code-specific queries
    queries = [
        "Find all TODO and FIXME comments in the code files",
        "What are the main functions and classes defined?",
        "Identify any potential issues or technical debt"
    ]
    
    for query in queries:
        print(f"\nQuery: {query}")
        result = analyzer.analyze_with_rlm(query)
        print(f"Result: {result}")
        print("-" * 40)
    
    # Clean up
    import shutil
    shutil.rmtree(sample_dir)


def example_error_analysis():
    """Example of analyzing error logs."""
    print("\n" + "="*60)
    print("EXAMPLE 3: Error Log Analysis")
    print("="*60)
    
    sample_dir = create_sample_documents()
    analyzer = RLMDocumentAnalyzer()
    analyzer.load_documents_from_directory(sample_dir)
    
    # Analyze error patterns
    query = """
    Analyze the error logs and provide:
    1. Most common error types
    2. Error frequency patterns
    3. Recommendations for system improvements
    """
    
    result = analyzer.analyze_with_rlm(query)
    print(f"Query: {query}")
    print(f"Result: {result}")
    
    # Clean up
    import shutil
    shutil.rmtree(sample_dir)


def example_custom_configuration():
    """Example with custom RLM configuration."""
    print("\n" + "="*60)
    print("EXAMPLE 4: Custom Configuration")
    print("="*60)
    
    # Create custom configuration
    config = RLMConfig(
        max_iterations=15,
        max_recursion_depth=2,
        max_code_executions=25
    )
    
    sample_dir = create_sample_documents()
    analyzer = RLMDocumentAnalyzer(config=config)
    analyzer.load_documents_from_directory(sample_dir)
    
    # Complex analysis query
    query = """
    Perform a comprehensive analysis of these documents:
    1. Extract all key metrics and numbers
    2. Identify relationships between different documents
    3. Create a summary report with actionable insights
    """
    
    result = analyzer.analyze_with_rlm(query)
    print(f"Query: {query}")
    print(f"Result: {result}")
    
    # Clean up
    import shutil
    shutil.rmtree(sample_dir)


def example_document_statistics():
    """Example of getting document statistics."""
    print("\n" + "="*60)
    print("EXAMPLE 5: Document Statistics")
    print("="*60)
    
    sample_dir = create_sample_documents()
    analyzer = RLMDocumentAnalyzer()
    analyzer.load_documents_from_directory(sample_dir)
    
    # Get statistics
    stats = analyzer.get_document_stats()
    print("Document Statistics:")
    print(json.dumps(stats, indent=2))
    
    # Generate summary report
    report = analyzer.generate_summary_report()
    print("\nSummary Report:")
    print(report)
    
    # Clean up
    import shutil
    shutil.rmtree(sample_dir)


def example_search_functionality():
    """Example of search functionality."""
    print("\n" + "="*60)
    print("EXAMPLE 6: Search Functionality")
    print("="*60)
    
    sample_dir = create_sample_documents()
    analyzer = RLMDocumentAnalyzer()
    analyzer.load_documents_from_directory(sample_dir)
    
    # Search for specific patterns
    search_patterns = [
        "TODO",
        "ERROR",
        "machine learning",
        "accuracy"
    ]
    
    for pattern in search_patterns:
        results = analyzer.search_documents(pattern)
        print(f"\nSearch results for '{pattern}':")
        for doc_id, matches in results.items():
            print(f"  {doc_id}: {len(matches)} matches")
            for match in matches[:2]:  # Show first 2 matches
                print(f"    {match}")
    
    # Clean up
    import shutil
    shutil.rmtree(sample_dir)


def main():
    """Run all examples."""
    print("RLM Document Analyzer - Example Usage")
    print("=" * 60)
    
    # Check if API key is set
    if not os.environ.get("GEMINI_API_KEY"):
        print("Warning: GEMINI_API_KEY environment variable not set.")
        print("Please set it to run the examples:")
        print("export GEMINI_API_KEY='your_api_key_here'")
        return
    
    try:
        example_basic_analysis()
        example_code_analysis()
        example_error_analysis()
        example_custom_configuration()
        example_document_statistics()
        example_search_functionality()
        
        print("\n" + "="*60)
        print("All examples completed successfully!")
        print("="*60)
        
    except Exception as e:
        print(f"Error running examples: {e}")
        print("Make sure you have set the GEMINI_API_KEY environment variable.")


if __name__ == "__main__":
    main()
