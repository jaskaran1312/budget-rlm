#!/usr/bin/env python3
"""
Main Entry Point for Recursive Language Model (RLM) Document Analysis
Provides a command-line interface for analyzing large collections of documents
using the RLM framework with Python REPL environment.
"""

import os
import sys
import argparse
import json
import logging
from typing import Dict, List, Optional
from pathlib import Path

from rlm_wrapper import RecursiveLanguageModel, RLMConfig, create_rlm
from document_analyzer import DocumentLoader, DocumentAnalyzer
from python_repl import create_repl_environment
from config import get_config, validate_and_setup

# Set up logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class RLMDocumentAnalyzer:
    """
    Main class for RLM-based document analysis.
    Combines the RLM framework with document analysis capabilities.
    """
    
    def __init__(self, api_key: Optional[str] = None, config: Optional[RLMConfig] = None):
        # Load configuration from .env file and system
        if config is None:
            config = get_config()
        
        # Use provided API key or get from config
        if api_key is None:
            api_key = config.api_key
        
        self.config = config
        self.rlm = create_rlm(api_key=api_key, config=self.config)
        self.document_loader = DocumentLoader()
        self.documents = {}
        self.metadata = {}
        self.analyzer = None
        
        logger.info("RLM Document Analyzer initialized")
    
    def load_documents_from_directory(self, directory_path: str, recursive: bool = True):
        """Load documents from a directory."""
        logger.info(f"Loading documents from directory: {directory_path}")
        self.documents, self.metadata = self.document_loader.load_from_directory(
            directory_path, recursive=recursive
        )
        self._update_analyzer()
    
    def load_documents_from_files(self, file_paths: List[str]):
        """Load documents from a list of file paths."""
        logger.info(f"Loading {len(file_paths)} documents from file list")
        self.documents, self.metadata = self.document_loader.load_from_files(file_paths)
        self._update_analyzer()
    
    def load_documents_from_json(self, json_path: str):
        """Load documents from a JSON file."""
        logger.info(f"Loading documents from JSON: {json_path}")
        self.documents, self.metadata = self.document_loader.load_from_json(json_path)
        self._update_analyzer()
    
    def _update_analyzer(self):
        """Update the document analyzer with current documents."""
        if self.documents:
            self.analyzer = DocumentAnalyzer(self.documents, self.metadata)
            # Load documents into RLM REPL
            self.rlm.analyzer.load_documents(self.documents, self.metadata)
            logger.info(f"Updated analyzer with {len(self.documents)} documents")
    
    def analyze_with_rlm(self, query: str) -> str:
        """
        Analyze documents using the RLM framework.
        
        Args:
            query: The analysis query
            
        Returns:
            Analysis result from the RLM
        """
        if not self.documents:
            logger.warning("No documents loaded")
            return "No documents loaded. Please load documents first."
        
        logger.info(f"Starting RLM analysis with query: {query[:100]}...")
        logger.debug(f"Query length: {len(query)} characters")
        logger.debug(f"Documents available: {len(self.documents)}")
        
        messages = [{"role": "user", "content": query}]
        logger.debug(f"Created messages: {len(messages)}")
        
        logger.info("Calling RLM completion...")
        response = self.rlm.completion(messages, context=self.documents)
        
        logger.info(f"RLM completion finished. Success: {response.success}")
        logger.debug(f"Response type: {response.response_type}")
        logger.debug(f"Response content length: {len(response.content)}")
        
        if response.success:
            logger.info("RLM analysis completed successfully")
            return response.content
        else:
            logger.error(f"RLM analysis failed: {response.error}")
            return f"Analysis failed: {response.error}"
    
    def get_document_stats(self) -> Dict:
        """Get document collection statistics."""
        if not self.analyzer:
            return {"error": "No documents loaded"}
        return self.analyzer.get_document_stats()
    
    def search_documents(self, pattern: str, case_sensitive: bool = False) -> Dict:
        """Search for patterns across documents."""
        if not self.analyzer:
            return {"error": "No documents loaded"}
        return self.analyzer.search_across_documents(pattern, case_sensitive)
    
    def generate_summary_report(self) -> str:
        """Generate a summary report of the document collection."""
        if not self.analyzer:
            return "No documents loaded"
        return self.analyzer.generate_summary_report()


def main():
    """Main entry point for the CLI."""
    parser = argparse.ArgumentParser(
        description="RLM Document Analyzer - Analyze large collections of documents using Recursive Language Models"
    )
    
    parser.add_argument(
        "--api-key",
        type=str,
        help="Gemini API key (or set GEMINI_API_KEY environment variable)"
    )
    
    parser.add_argument(
        "--documents",
        type=str,
        help="Path to directory containing documents, or JSON file with documents (default: Documents/)"
    )
    
    parser.add_argument(
        "--files",
        nargs="+",
        help="List of specific files to analyze"
    )
    
    parser.add_argument(
        "--query",
        type=str,
        help="Analysis query to run on the documents"
    )
    
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Run in interactive mode"
    )
    
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=50,
        help="Maximum RLM iterations (default: 50)"
    )
    
    parser.add_argument(
        "--max-recursion-depth",
        type=int,
        default=3,
        help="Maximum RLM recursion depth (default: 3)"
    )
    
    parser.add_argument(
        "--output",
        type=str,
        help="Output file for results"
    )
    
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Validate configuration and setup logging
    if not validate_and_setup():
        logger.error("Configuration validation failed")
        sys.exit(1)
    
    # Get configuration
    config = get_config()
    
    # Override with command line arguments
    if args.max_iterations != 50:
        config.max_iterations = args.max_iterations
    if args.max_recursion_depth != 3:
        config.max_recursion_depth = args.max_recursion_depth
    
    # Get API key (command line takes precedence)
    api_key = args.api_key or config.api_key
    if not api_key:
        logger.error("API key required. Set GEMINI_API_KEY in .env file or use --api-key")
        sys.exit(1)
    
    # Initialize analyzer
    analyzer = RLMDocumentAnalyzer(api_key=api_key, config=config)
    
    # Load documents
    if args.documents:
        if args.documents.endswith('.json'):
            analyzer.load_documents_from_json(args.documents)
        else:
            analyzer.load_documents_from_directory(args.documents)
    elif args.files:
        analyzer.load_documents_from_files(args.files)
    else:
        # Use Documents directory as default
        documents_dir = "Documents"
        if Path(documents_dir).exists() and any(Path(documents_dir).iterdir()):
            logger.info(f"Using default Documents directory: {documents_dir}")
            analyzer.load_documents_from_directory(documents_dir)
        else:
            logger.error(f"No documents found in default directory '{documents_dir}' and no documents specified. Use --documents or --files")
            sys.exit(1)
    
    if not analyzer.documents:
        logger.error("No documents loaded")
        sys.exit(1)
    
    logger.info(f"Loaded {len(analyzer.documents)} documents")
    
    # Run analysis
    if args.query:
        # Single query mode
        result = analyzer.analyze_with_rlm(args.query)
        
        if args.output:
            with open(args.output, 'w', encoding='utf-8') as f:
                f.write(result)
            logger.info(f"Results written to {args.output}")
        else:
            print("\n" + "="*80)
            print("RLM ANALYSIS RESULT")
            print("="*80)
            print(result)
            print("="*80)
    
    elif args.interactive:
        # Interactive mode
        run_interactive_mode(analyzer)
    
    else:
        # Default: show document stats
        stats = analyzer.get_document_stats()
        print("\n" + "="*80)
        print("DOCUMENT COLLECTION STATISTICS")
        print("="*80)
        print(json.dumps(stats, indent=2))
        print("="*80)
        
        # Generate summary report
        report = analyzer.generate_summary_report()
        print("\n" + "="*80)
        print("SUMMARY REPORT")
        print("="*80)
        print(report)
        print("="*80)


def run_interactive_mode(analyzer: RLMDocumentAnalyzer):
    """Run the analyzer in interactive mode."""
    print("\n" + "="*80)
    print("RLM DOCUMENT ANALYZER - INTERACTIVE MODE")
    print("="*80)
    print(f"Loaded {len(analyzer.documents)} documents")
    print("Type 'help' for available commands, 'quit' to exit")
    print("="*80)
    
    while True:
        try:
            user_input = input("\nRLM> ").strip()
            
            if not user_input:
                continue
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("Goodbye!")
                break
            
            if user_input.lower() == 'help':
                print_help()
                continue
            
            if user_input.lower() == 'stats':
                stats = analyzer.get_document_stats()
                print(json.dumps(stats, indent=2))
                continue
            
            if user_input.lower() == 'summary':
                report = analyzer.generate_summary_report()
                print(report)
                continue
            
            if user_input.startswith('search '):
                pattern = user_input[7:].strip()
                if pattern:
                    results = analyzer.search_documents(pattern)
                    print(json.dumps(results, indent=2))
                else:
                    print("Please provide a search pattern")
                continue
            
            # Default: treat as RLM query
            print("\nAnalyzing with RLM...")
            result = analyzer.analyze_with_rlm(user_input)
            print("\n" + "-"*60)
            print("RESULT:")
            print("-"*60)
            print(result)
            print("-"*60)
            
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            logger.error(f"Error in interactive mode: {e}")
            print(f"Error: {e}")


def print_help():
    """Print help information for interactive mode."""
    help_text = """
Available Commands:
- help: Show this help message
- stats: Show document collection statistics
- summary: Generate and show summary report
- search <pattern>: Search for pattern across documents
- <query>: Any other input is treated as an RLM analysis query
- quit/exit/q: Exit the program

Examples:
- "What are the main topics discussed in these documents?"
- "Find all mentions of 'machine learning' in the code files"
- "Summarize the key findings from the research papers"
- "What are the common patterns in the error logs?"
- search "TODO|FIXME"
"""
    print(help_text)


if __name__ == "__main__":
    main()
