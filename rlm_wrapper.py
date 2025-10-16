"""
Recursive Language Model (RLM) Wrapper
Implements the RLM framework where a language model can recursively call itself
or other LMs through a Python REPL environment.
"""

import os
import json
import logging
import time
from typing import Dict, Any, List, Optional, Tuple, Callable
from dataclasses import dataclass
from enum import Enum
import re

from python_repl import SecureREPL, DocumentAnalyzer, create_repl_environment

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


class RLMResponseType(Enum):
    """Types of responses from the RLM."""
    FINAL = "FINAL"
    FINAL_VAR = "FINAL_VAR"
    CODE = "CODE"
    CONTINUE = "CONTINUE"


@dataclass
class RLMResponse:
    """Response from a recursive language model call."""
    response_type: RLMResponseType
    content: str
    success: bool
    error: Optional[str] = None
    execution_output: Optional[str] = None
    variables_used: Optional[List[str]] = None


@dataclass
class RLMConfig:
    """Configuration for the RLM system."""
    max_iterations: int = 50
    max_code_executions: int = 20
    timeout_seconds: int = 300
    enable_recursive_calls: bool = True
    max_recursion_depth: int = 3
    model_name: str = "gemini-2.5-flash"
    system_instruction: str = ""


class LLMClient:
    """
    Abstract base class for LLM clients.
    This allows the RLM to work with different LLM providers.
    """
    
    def generate(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """Generate a response from the LLM."""
        raise NotImplementedError


class GeminiClient(LLMClient):
    """Gemini client implementation using the provided code."""
    
    def __init__(self, api_key: Optional[str] = None, model: str = "gemini-2.5-flash"):
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY")
        self.model = model
        
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY environment variable is required")
        
        # Import here to avoid dependency issues
        try:
            from google import genai
            from google.genai import types
            self.genai = genai
            self.types = types
        except ImportError:
            raise ImportError("Please install google-genai: pip install google-genai")
        
        self.client = self.genai.Client(api_key=self.api_key)
    
    def generate(self, messages: List[Dict[str, str]], system_instruction: str = "", **kwargs) -> str:
        """Generate a response using Gemini."""
        try:
            logger.debug(f"GeminiClient.generate called with {len(messages)} messages")
            logger.debug(f"System instruction length: {len(system_instruction)}")
            
            # Convert messages to Gemini format
            contents = []
            for i, message in enumerate(messages):
                role = "user" if message["role"] == "user" else "model"
                content_preview = message["content"][:200] + "..." if len(message["content"]) > 200 else message["content"]
                logger.debug(f"Message {i}: {role} - {content_preview}")
                
                contents.append(
                    self.types.Content(
                        role=role,
                        parts=[self.types.Part.from_text(text=message["content"])]
                    )
                )
            
            # Configure generation
            config = self.types.GenerateContentConfig(
                thinking_config=self.types.ThinkingConfig(thinking_budget=0),
                system_instruction=[self.types.Part.from_text(text=system_instruction)] if system_instruction else [],
            )
            
            logger.debug(f"Calling Gemini API with model: {self.model}")
            
            # Generate response
            response_text = ""
            chunk_count = 0
            for chunk in self.client.models.generate_content_stream(
                model=self.model,
                contents=contents,
                config=config,
            ):
                chunk_count += 1
                if chunk.text:
                    response_text += chunk.text
                    logger.debug(f"Received chunk {chunk_count}: {len(chunk.text)} characters")
            
            logger.debug(f"Total response length: {len(response_text)} characters from {chunk_count} chunks")
            logger.debug(f"Response preview: {response_text[:300]}...")
            
            return response_text
            
        except Exception as e:
            logger.error(f"Error generating response with Gemini: {e}")
            logger.error(f"Exception type: {type(e).__name__}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            raise


class RecursiveLanguageModel:
    """
    Recursive Language Model implementation.
    Manages recursive LLM calls through a Python REPL environment.
    """
    
    def __init__(self, llm_client: LLMClient, config: RLMConfig = None):
        self.llm_client = llm_client
        self.config = config or RLMConfig()
        self.repl, self.analyzer = create_repl_environment()
        self.call_history = []
        self.current_depth = 0
        
        # Initialize system instruction
        self._setup_system_instruction()
    
    def _setup_system_instruction(self):
        """Set up the system instruction for the RLM."""
        system_instruction = """
        # Recursive Language Model (RLM) System Instructions

        You are an advanced Recursive Language Model with document analysis capabilities through an integrated Python REPL environment. Your purpose is to analyze documents, answer queries, and solve complex problems through code execution and recursive reasoning.

        ---

        ## Core Capabilities

        ### 1. Python REPL Environment
        - **Full Python execution** with persistent state across interactions
        - **Variable persistence** - all variables remain in namespace between executions
        - **Standard libraries** available (re, json, collections, etc.)
        - **Error handling** with detailed traceback and recovery mechanisms

        ### 2. Document Access Layer
        You have immediate access to:
        - `documents`: Dictionary containing all loaded documents (key: doc_id, value: content)
        - `document_metadata`: Metadata for each document (timestamps, sizes, types)
        - `num_documents`: Total count of loaded documents

        ### 3. Recursive Reasoning
        - Make recursive calls to yourself or other language models
        - Decompose complex problems into manageable sub-problems
        - Maintain context across recursive depth levels

        ---

        ## Code Execution Protocol

        ### Syntax
        Wrap all Python code in triple backticks with language specification:
        ```python
        # Your code here
        result = analyze_data()
        ```

        ### Execution Flow
        1. Code is executed in persistent REPL environment
        2. Variables are stored and accessible in subsequent executions
        3. Output, return values, and errors are captured
        4. Execution history is maintained for debugging

        ### Best Practices
        - **Verify before use**: Check variable existence before operations
        - **Incremental development**: Build complex logic step-by-step
        - **Error anticipation**: Use try-except blocks for robust code
        - **Memory efficiency**: Clear large variables when no longer needed
        - **Debugging first**: Check execution history if errors occur

        ---

        ## Result Communication

        ### Method 1: FINAL (Recommended for most cases)
        Use when you have a definitive answer:

        ```python
        # For literal responses
        FINAL("The document contains 42 occurrences of the pattern.")

        # For variable-based responses
        answer = compute_analysis()
        FINAL(answer)  # Variable will be automatically evaluated
        ```

        ### Method 2: FINAL_VAR (Explicit variable return)
        Use when explicitly returning a pre-existing variable:

        ```python
        # Create variable first
        result_data = {"count": 42, "summary": "Analysis complete"}

        # Then return it
        FINAL_VAR(result_data)
        ```

        **Critical Requirements for FINAL_VAR:**
        - Variable MUST exist in REPL namespace before calling
        - If variable doesn't exist, you'll receive an error with available variables
        - System will list all available variables to help you identify the issue
        - You can then create the variable and retry

        ### Method 3: Mixed Approach
        Combine code execution and result return in a single response:

        ```python
        # Create and process data
        analysis_result = perform_complex_analysis(documents)

        # Immediately return it
        FINAL_VAR(analysis_result)
        ```

        ---

        ## Built-in Utility Functions

        ### Basic Text Processing
        ```python
        # Pattern searching with regex support
        matches = grep(pattern, text)

        # Count pattern occurrences
        count = count_occurrences(pattern, text)

        # Extract content between delimiters
        sections = extract_sections(text, start_pattern, end_pattern)

        # Generate concise summaries
        summary = summarize_text(text, max_length=500)
        ```

        ### Debugging & Execution History
        ```python
        # View recent code executions with results
        history = get_recent_executions(n=5)

        # Examine recent errors for debugging
        errors = get_execution_errors(n=5)
        ```

        **When to use execution history:**
        - After encountering errors
        - Before retrying failed operations
        - To understand variable state
        - To learn from previous mistakes

        ### Advanced Document Search
        ```python
        # Search with contextual lines around matches
        results = advanced_search(
            pattern="error|exception",
            context_lines=3,
            case_sensitive=False,
            doc_ids=None  # None = all documents
        )

        # Find multiple patterns simultaneously
        pattern_results = find_patterns(
            patterns=["TODO", "FIXME", "WARNING"],
            case_sensitive=False,
            doc_ids=[0, 1, 2]
        )
        ```

        ### Section Extraction
        ```python
        # Extract sections from documents
        sections = extract_sections_docs(
            start_pattern=r"^## Introduction",
            end_pattern=r"^## Conclusion",
            include_delimiters=True,
            doc_ids=None
        )

        # Extract from single text string
        single_sections = extract_sections_single(
            text=my_text,
            start_pattern="START",
            end_pattern="END",
            include_delimiters=False
        )
        ```

        ### Document Slicing
        ```python
        # Flexible slicing with multiple methods
        sliced = slice_docs(
            start_pattern=r"^Chapter \d+",  # Regex pattern
            end_pattern=r"^Appendix",
            start_line=10,                   # Line number
            end_line=100,
            start_char=0,                    # Character position
            end_char=5000,
            doc_ids=[0, 1]
        )

        # Slice single text
        text_slice = slice_single_text(
            text=document_text,
            start_line=50,
            end_line=150
        )
        ```

        ### Structural & Semantic Analysis
        ```python
        # Analyze document structure
        structure = analyze_structure(doc_ids=None)
        # Returns: line counts, paragraph stats, heading hierarchy, etc.

        # Perform semantic analysis
        semantics = semantic_analysis(doc_ids=None)
        # Returns: word frequency, key phrases, readability metrics

        # Analyze single text string
        text_analysis = analyze_single_text(
            text=my_document,
            analysis_type='structure'  # or 'semantic'
        )
        ```

        ### Single Text Operations
        For operations on text strings (not document IDs):
        ```python
        # Search single text with context
        search_single_text(text, pattern, context_lines=3, case_sensitive=False)

        # Extract sections from single text
        extract_sections_single(text, start_pattern, end_pattern, include_delimiters=False)

        # Slice single text
        slice_single_text(text, start_pattern, end_pattern, start_line, end_line)

        # Analyze single text
        analyze_single_text(text, analysis_type='structure')
        ```

        ---

        ## Advanced Features

        ### Dynamic Tool Creation
        Create custom text processing functions on-the-fly:

        ```python
        # Define custom tool
        create_custom_tool(
            name="extract_emails",
            code=\"\"\"
        def extract_emails(text):
            import re
            pattern = r'\\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\\.[A-Z|a-z]{2,}\\b'
            return re.findall(pattern, text)
        \"\"\""
            description="Extract all email addresses from text"
        )

        # Use the custom tool
        emails = execute_tool("extract_emails", documents[0])
        ```

        **Guidelines for custom tools:**
        - Include necessary imports within the function
        - Return structured data when possible
        - Add clear docstrings/descriptions
        - Handle edge cases and errors
        - Test with sample data first

        ### Text Processing Pipelines
        Chain multiple operations for complex workflows:

        ```python
        # Create a pipeline
        create_text_pipeline(
            name="extract_code_blocks",
            steps=[
                {
                    "type": "search",
                    "params": {"pattern": "```python", "context_lines": 0}
                },
                {
                    "type": "extract",
                    "params": {
                        "start_pattern": "```python",
                        "end_pattern": "```"
                    }
                },
                {
                    "type": "filter",
                    "params": {
                        "filter_type": "length",
                        "min_length": 50
                    }
                }
            ],
            description="Extract and filter Python code blocks"
        )

        # Execute the pipeline
        code_blocks = execute_pipeline("extract_code_blocks", doc_ids=[0, 1, 2])
        ```

        **Pipeline step types:**
        - `search`: Find patterns with context
        - `extract`: Extract content between delimiters
        - `slice`: Slice by line/character/pattern
        - `filter`: Filter results by criteria
        - `transform`: Apply transformations
        - `custom`: Use custom tools

        ### Tool Discovery
        ```python
        # List all available tools
        tools = get_available_tools()

        # Returns dictionary with:
        # - built_in: standard functions
        # - custom: user-defined tools
        # - pipelines: defined pipelines
        ```

        ### Recursive Calls
        Decompose complex problems into sub-problems:

        ```python
        # Make recursive call
        sub_result = rlm_call(
            query="Analyze the sentiment of section 3",
            depth=1  # Current recursion depth
        )

        # Use result in parent analysis
        final_analysis = combine_results(sub_result, other_data)
        ```

        **Recursion best practices:**
        - Use for genuinely decomposable problems
        - Pass minimal necessary context
        - Track recursion depth
        - Set clear termination conditions
        - Aggregate results at parent level

        ---

        ## Error Handling & Recovery

        ### Error Detection
        When errors occur, the system provides:
        - Detailed error messages with tracebacks
        - List of available variables in namespace
        - Execution history context
        - Suggestions for recovery

        ### Recovery Workflow
        ```python
        # 1. Check recent errors
        errors = get_execution_errors(5)

        # 2. Review execution history
        history = get_recent_executions(5)

        # 3. Identify the issue
        # - Missing variable?
        # - Wrong variable name?
        # - Logic error?

        # 4. Fix and retry
        # Create missing variable or correct logic
        corrected_result = fixed_computation()

        # 5. Verify and return
        FINAL_VAR(corrected_result)
        ```

        ### Common Error Patterns
        1. **NameError**: Variable doesn't exist
        - Check spelling and creation order
        - Verify variable was actually created in previous execution
        
        2. **KeyError**: Dictionary key missing
        - Verify document IDs exist
        - Check metadata keys
        
        3. **IndexError**: List index out of range
        - Check list lengths before indexing
        - Use conditional access

        4. **TypeError**: Type mismatch
        - Verify data types before operations
        - Add type conversion if needed

        ---

        ## Optimization Strategies

        ### For Large Documents
        ```python
        # Use grep/regex to narrow down first
        relevant_sections = grep(r"specific_pattern", documents[0])

        # Then perform detailed analysis
        detailed_results = analyze_relevant_sections(relevant_sections)
        ```

        ### For Multiple Documents
        ```python
        # Process in batches
        batch_results = []
        for doc_id in range(0, num_documents, 10):
            batch = process_batch(doc_id, doc_id + 10)
            batch_results.append(batch)

        # Aggregate results
        final_result = aggregate(batch_results)
        ```

        ### Memory Management
        ```python
        # Clear large intermediate variables
        large_data = process_huge_dataset()
        result = extract_summary(large_data)
        del large_data  # Free memory

        # Use generators for large iterations
        def process_documents():
            for doc_id in range(num_documents):
                yield analyze_document(documents[doc_id])
        ```

        ---

        ## Workflow Templates

        ### Simple Query Pattern
        ```python
        # 1. Search for relevant content
        matches = advanced_search("keyword", context_lines=2)

        # 2. Process results
        processed = analyze_matches(matches)

        # 3. Return answer
        FINAL(f"Found {len(matches)} occurrences with key insights: {processed}")
        ```

        ### Complex Analysis Pattern
        ```python
        # 1. Understand the problem
        structure = analyze_structure()

        # 2. Break down into steps
        step1_result = perform_initial_analysis()
        step2_result = perform_detailed_analysis(step1_result)
        step3_result = synthesize_findings(step1_result, step2_result)

        # 3. Create comprehensive result
        final_report = {
            "summary": step3_result,
            "details": step2_result,
            "metadata": structure
        }

        # 4. Return structured result
        FINAL_VAR(final_report)
        ```

        ### Error-Resilient Pattern
        ```python
        # Check execution history first
        recent = get_recent_executions(3)

        try:
            # Attempt primary approach
            result = primary_analysis()
        except Exception as e:
            # Fall back to alternative approach
            result = fallback_analysis()
            
        # Verify result before returning
        if validate_result(result):
            FINAL_VAR(result)
        else:
            FINAL("Analysis incomplete - requires manual review")
        ```

        ---

        ## System Configuration

        **Current Settings:**
        - Maximum iterations: {self.config.max_iterations}
        - Maximum code executions: {self.config.max_code_executions}
        - Recursive calls enabled: {self.config.enable_recursive_calls}
        - Maximum recursion depth: {self.config.max_recursion_depth}

        ---

        ## Response Guidelines

        ### Structure Your Responses
        1. **Acknowledge the query**: Show understanding of the request
        2. **Plan your approach**: Briefly outline your strategy
        3. **Execute incrementally**: Build analysis step-by-step
        4. **Verify results**: Check outputs before finalizing
        5. **Deliver clearly**: Use FINAL with clear, actionable answer

        ### Communication Style
        - Be precise and technical when analyzing data
        - Explain your reasoning for complex operations
        - Point out limitations or assumptions
        - Suggest alternative approaches when relevant
        - Provide context for numerical results

        ### Quality Checks
        Before using FINAL or FINAL_VAR:
        - ✓ Have I fully answered the query?
        - ✓ Are my results validated and accurate?
        - ✓ Have I handled edge cases?
        - ✓ Is my response clear and actionable?
        - ✓ Do variables exist if using FINAL_VAR?

        ---

        ## Quick Reference

        ### Most Common Operations
        ```python
        # Search documents
        advanced_search(pattern, context_lines=3)

        # Extract sections
        extract_sections_docs(start_pattern, end_pattern)

        # Analyze structure
        analyze_structure()

        # Get execution history
        get_recent_executions(5)

        # Return results
        FINAL("answer") or FINAL_VAR(variable)
        ```

        ### Decision Tree: Which Function to Use?
        - **Need to search?** → `advanced_search()` or `grep()`
        - **Extract sections?** → `extract_sections_docs()` or `extract_sections_single()`
        - **Analyze structure?** → `analyze_structure()` or `semantic_analysis()`
        - **Process single text?** → Use `_single()` variants
        - **Complex workflow?** → Create pipeline
        - **Custom logic?** → Create custom tool
        - **Debug issues?** → `get_execution_errors()` and `get_recent_executions()`

        ---

        ## Remember

        - **Always verify variable existence** before using FINAL_VAR
        - **Check execution history** when encountering errors
        - **Build incrementally** rather than one large code block
        - **Use appropriate tools** for document vs. text string operations
        - **Optimize for large datasets** with grep/regex filtering first
        - **Learn from errors** using execution history functions
        - **Test assumptions** before finalizing answers

        **Your goal**: Provide accurate, efficient, and well-reasoned analysis of documents through systematic code execution and recursive reasoning.
"""
        
        self.config.system_instruction = system_instruction
    
    def completion(self, messages: List[Dict[str, str]], context: Optional[Dict[str, str]] = None) -> RLMResponse:
        """
        Main entry point for RLM completion.
        
        Args:
            messages: List of message dictionaries with 'role' and 'content'
            context: Optional context to load into the REPL
            
        Returns:
            RLMResponse with the final answer
        """
        logger.info(f"RLM.completion called with {len(messages)} messages")
        logger.debug(f"Context provided: {bool(context)}")
        
        # Load context if provided
        if context:
            logger.info(f"Loading {len(context)} documents into REPL")
            self.analyzer.load_documents(context)
        
        # Extract the query from the last user message
        query = messages[-1]["content"] if messages else ""
        logger.info(f"User query: {query[:100]}...")
        
        # Initialize the REPL with the query
        self.repl.set_variable("user_query", query)
        logger.debug("Set user_query variable in REPL")
        
        # Start the recursive process
        logger.info("Starting recursive processing")
        return self._process_recursively(messages, depth=0)
    
    def _process_recursively(self, messages: List[Dict[str, str]], depth: int = 0) -> RLMResponse:
        """
        Process the query recursively through the REPL environment.
        
        Args:
            messages: Current conversation messages
            depth: Current recursion depth
            
        Returns:
            RLMResponse with the final answer
        """
        logger.info(f"Processing recursively at depth {depth}")
        
        if depth > self.config.max_recursion_depth:
            logger.warning(f"Maximum recursion depth {self.config.max_recursion_depth} exceeded")
            return RLMResponse(
                response_type=RLMResponseType.FINAL,
                content="Maximum recursion depth reached",
                success=False,
                error="Recursion depth limit exceeded"
            )
        
        self.current_depth = depth
        iteration = 0
        code_executions = 0
        
        logger.info(f"Starting iteration loop: max_iterations={self.config.max_iterations}, max_code_executions={self.config.max_code_executions}")
        
        while iteration < self.config.max_iterations and code_executions < self.config.max_code_executions:
            try:
                logger.info(f"Iteration {iteration + 1}/{self.config.max_iterations}, Code executions: {code_executions}/{self.config.max_code_executions}")
                
                # Prepare the prompt for the LLM
                prompt = self._prepare_prompt(messages, iteration, code_executions)
                logger.debug(f"Prepared prompt length: {len(prompt)}")
                logger.debug(f"Prompt preview: {prompt[:500]}...")
                
                # Get response from LLM
                logger.info("Calling LLM client...")
                response_text = self.llm_client.generate(
                    messages=[{"role": "user", "content": prompt}],
                    system_instruction=self.config.system_instruction
                )
                
                logger.info(f"Received LLM response: {len(response_text)} characters")
                logger.debug(f"LLM response preview: {response_text[:500]}...")
                
                # Record the call
                self.call_history.append({
                    "depth": depth,
                    "iteration": iteration,
                    "response": response_text,
                    "timestamp": time.time()
                })
                
                # Parse the response (now returns a list of responses)
                logger.debug("Parsing LLM response...")
                parsed_responses = self._parse_response(response_text)
                logger.info(f"Parsed {len(parsed_responses)} response(s)")
                
                # Process responses in order, but execute CODE first, then FINAL/FINAL_VAR
                final_result = None
                
                # First pass: Execute all CODE responses
                for i, parsed_response in enumerate(parsed_responses):
                    if parsed_response.response_type == RLMResponseType.CODE:
                        logger.info(f"Processing CODE response {i+1}/{len(parsed_responses)}")
                        logger.debug(f"Code to execute: {parsed_response.content[:300]}...")
                        
                        success, output, error = self.repl.execute_code(parsed_response.content)
                        code_executions += 1
                        
                        logger.info(f"Code execution result: success={success}")
                        if success:
                            logger.debug(f"Code output: {output[:300]}...")
                        else:
                            logger.warning(f"Code execution failed: {error}")
                        
                        if not success:
                            # Continue with the error information
                            error_msg = f"Code execution failed: {error}\nPlease fix the code and try again."
                            messages.append({
                                "role": "assistant",
                                "content": error_msg
                            })
                            logger.debug(f"Added error message to conversation: {error_msg[:100]}...")
                        else:
                            # Add execution output to context
                            success_msg = f"Code executed successfully. Output:\n{output}"
                            messages.append({
                                "role": "assistant",
                                "content": success_msg
                            })
                            logger.debug(f"Added success message to conversation: {success_msg[:100]}...")
                
                # Second pass: Process FINAL and FINAL_VAR responses
                for i, parsed_response in enumerate(parsed_responses):
                    logger.info(f"Processing response {i+1}/{len(parsed_responses)}: {parsed_response.response_type}")
                    logger.debug(f"Response content: {parsed_response.content[:200]}...")
                    
                    if parsed_response.response_type == RLMResponseType.FINAL:
                        logger.info("Received FINAL response, checking if content is a variable...")
                        content = parsed_response.content
                        
                        # Check if the content is a variable name that should be evaluated
                        if content and not content.startswith('"') and not content.startswith("'") and len(content.split()) == 1:
                            # It looks like a variable name, try to evaluate it
                            logger.debug(f"FINAL content '{content}' looks like a variable name, checking REPL...")
                            var_value = self.repl.get_variable(content)
                            logger.debug(f"Variable '{content}' value: {var_value}")
                            if var_value is not None:
                                logger.debug(f"FINAL content '{content}' is a variable, evaluating to: {str(var_value)[:100]}...")
                                return RLMResponse(
                                    response_type=RLMResponseType.FINAL,
                                    content=str(var_value),
                                    success=True
                                )
                            else:
                                logger.debug(f"Variable '{content}' not found, treating as literal content")
                        
                        # Treat as literal content or return as-is
                        logger.info("Returning FINAL response as literal content")
                        return parsed_response
                    elif parsed_response.response_type == RLMResponseType.FINAL_VAR:
                        # Get variable from REPL
                        var_name = parsed_response.content.strip()
                        logger.info(f"Getting variable '{var_name}' from REPL")
                        var_value = self.repl.get_variable(var_name)
                        if var_value is not None:
                            logger.info(f"Variable found, returning value")
                            return RLMResponse(
                                response_type=RLMResponseType.FINAL,
                                content=str(var_value),
                                success=True
                            )
                        else:
                            logger.warning(f"Variable '{var_name}' not found in REPL")
                            # Get list of available variables for debugging
                            available_vars = self.repl.list_variables()
                            var_list = list(available_vars.keys())
                            logger.debug(f"Available variables: {var_list}")
                            
                            # Create a helpful error message and continue instead of failing
                            error_msg = f"Variable '{var_name}' not found in REPL. Available variables: {', '.join(var_list[:10])}"
                            if len(var_list) > 10:
                                error_msg += f" (and {len(var_list) - 10} more)"
                            
                            # Add error to conversation and continue
                            messages.append({
                                "role": "assistant",
                                "content": f"ERROR: {error_msg}\nPlease create the variable first or use a different approach."
                            })
                            logger.info("Added variable error to conversation, continuing analysis")
                            continue  # Continue processing other responses
                            
                    elif parsed_response.response_type == RLMResponseType.CONTINUE:
                        # Add continuation message to conversation
                        messages.append({
                            "role": "assistant",
                            "content": parsed_response.content
                        })
                        logger.debug("Added continuation message to conversation")
                
                iteration += 1
                logger.debug(f"Completed iteration {iteration}")
                
            except Exception as e:
                logger.error(f"Error in recursive processing at iteration {iteration}: {e}")
                logger.error(f"Exception type: {type(e).__name__}")
                import traceback
                logger.error(f"Traceback: {traceback.format_exc()}")
                return RLMResponse(
                    response_type=RLMResponseType.FINAL,
                    content=f"Error during processing: {str(e)}",
                    success=False,
                    error=str(e)
                )
        
        # If we've exhausted iterations, return what we have
        logger.warning(f"Exhausted iterations: {iteration} iterations, {code_executions} code executions")
        return RLMResponse(
            response_type=RLMResponseType.FINAL,
            content="Maximum iterations reached without final answer",
            success=False,
            error="Iteration limit exceeded"
        )
    
    def _prepare_prompt(self, messages: List[Dict[str, str]], iteration: int, code_executions: int) -> str:
        """Prepare the prompt for the LLM."""
        prompt = f"""
Iteration {iteration + 1} of {self.config.max_iterations}
Code executions used: {code_executions} of {self.config.max_code_executions}
Recursion depth: {self.current_depth}

Current conversation:
"""
        
        for msg in messages[-3:]:  # Include last 3 messages for context
            prompt += f"{msg['role'].upper()}: {msg['content']}\n\n"
        
        prompt += """
Available variables in REPL:
"""
        
        variables = self.repl.list_variables()
        for name, value in variables.items():
            prompt += f"- {name}: {value}\n"
        
        prompt += """
Instructions:
1. Analyze the query and available data
2. Use Python code (```python ... ```) to process data if needed
3. Use FINAL(answer) when you have the final answer
4. Use FINAL_VAR(variable_name) to return a variable from the REPL
5. You can make recursive calls with rlm_call(query, depth=1) if needed

What would you like to do next?
"""
        
        return prompt
    
    def _parse_response(self, response_text: str) -> List[RLMResponse]:
        """Parse the LLM response to determine the types and content. Returns a list of responses."""
        import re  # Import at the top of the method
        
        logger.debug(f"Parsing response: {len(response_text)} characters")
        response_text = response_text.strip()
        responses = []
        
        # Check for multiple response types in the same response
        # Look for FINAL responses first
        final_matches = re.findall(r'FINAL\((.*?)\)', response_text, re.DOTALL)
        for match in final_matches:
            content = match.strip()
            logger.debug(f"Found FINAL response: {content[:100]}...")
            
            # Store the content for later evaluation (after code execution)
            responses.append(RLMResponse(
                response_type=RLMResponseType.FINAL,
                content=content,
                success=True
            ))
        
        # Look for FINAL_VAR responses
        final_var_matches = re.findall(r'FINAL_VAR\((.*?)\)', response_text, re.DOTALL)
        for match in final_var_matches:
            content = match.strip()
            logger.debug(f"Found FINAL_VAR response: {content}")
            responses.append(RLMResponse(
                response_type=RLMResponseType.FINAL_VAR,
                content=content,
                success=True
            ))
        
        # Look for code blocks
        code_matches = re.findall(r'```python\s*(.*?)\s*```', response_text, re.DOTALL)
        for match in code_matches:
            content = match.strip()
            logger.debug(f"Found CODE response: {len(content)} characters")
            responses.append(RLMResponse(
                response_type=RLMResponseType.CODE,
                content=content,
                success=True
            ))
        
        # Check for f-string literals that need evaluation
        # Support multiple f-string formats: f"...", f'...', f"""...""", f'''...'''
        f_string_patterns = [
            (r'^f"([^"]*(?:"[^"]*)*)"$', 'f"'),
            (r"^f'([^']*(?:'[^']*)*)'$", "f'"),
            (r'^f"""([^"]*(?:"""[^"]*)*)"""$', 'f"""'),
            (r"^f'''([^']*(?:'''[^']*)*)'''$", "f'''")
        ]
        
        is_f_string = False
        for pattern, prefix in f_string_patterns:
            if re.match(pattern, response_text):
                is_f_string = True
                logger.debug(f"Found f-string literal with pattern {prefix}, attempting to evaluate")
                break
        
        if is_f_string:
            try:
                # First try to evaluate as-is
                success, result, error = self.repl.execute_code(f'result = {response_text}')
                
                if not success:
                    # Try different approaches to fix common f-string issues
                    logger.debug(f"Initial evaluation failed: {error}")
                    
                    # Approach 1: Handle quote conflicts by escaping
                    if "f-string: expecting '}'" in str(error) or "unterminated string literal" in str(error):
                        logger.debug("Quote conflict detected, attempting to fix f-string")
                        # Try to fix by converting problematic quotes
                        fixed_f_string = self._fix_f_string_quotes(response_text)
                        if fixed_f_string != response_text:
                            logger.debug(f"Fixed f-string: {fixed_f_string}")
                            success, result, error = self.repl.execute_code(f'result = {fixed_f_string}')
                    
                    # Approach 2: Try evaluating as a regular string if f-string fails
                    if not success:
                        logger.debug("Trying to evaluate as regular string")
                        # Remove the 'f' prefix and try as regular string
                        regular_string = response_text[1:]  # Remove 'f' prefix
                        success, result, error = self.repl.execute_code(f'result = {regular_string}')
                
                if success:
                    # Get the evaluated result
                    evaluated_result = self.repl.get_variable("result")
                    if evaluated_result is not None:
                        logger.debug(f"Successfully evaluated f-string: {str(evaluated_result)[:100]}...")
                        responses.append(RLMResponse(
                            response_type=RLMResponseType.FINAL,
                            content=str(evaluated_result),
                            success=True
                        ))
                    else:
                        logger.warning("F-string evaluation succeeded but result is None")
                else:
                    logger.warning(f"Failed to evaluate f-string: {error}")
                    # As a fallback, try to extract any meaningful content from the f-string
                    fallback_content = self._extract_f_string_content(response_text)
                    if fallback_content:
                        logger.debug(f"Using fallback content: {fallback_content[:100]}...")
                        responses.append(RLMResponse(
                            response_type=RLMResponseType.FINAL,
                            content=fallback_content,
                            success=True
                        ))
            except Exception as e:
                logger.warning(f"Exception while evaluating f-string: {e}")
                # Try fallback extraction
                fallback_content = self._extract_f_string_content(response_text)
                if fallback_content:
                    responses.append(RLMResponse(
                        response_type=RLMResponseType.FINAL,
                        content=fallback_content,
                        success=True
                    ))
        
        # Check for other formatted string patterns that might need evaluation
        # Look for patterns like: "An example of language used for a Table of Contents (from document '{doc_id_with_toc}'"
        if not responses and ('{' in response_text and '}' in response_text):
            # Check if this looks like a formatted string with variables
            variable_pattern = r'\{[^}]+\}'
            variables = re.findall(variable_pattern, response_text)
            
            if variables:
                logger.debug(f"Found potential formatted string with variables: {variables}")
                try:
                    # Try to evaluate as an f-string
                    f_string_version = f"f{response_text}" if not response_text.startswith('f') else response_text
                    success, result, error = self.repl.execute_code(f'result = {f_string_version}')
                    
                    if success:
                        evaluated_result = self.repl.get_variable("result")
                        if evaluated_result is not None:
                            logger.debug(f"Successfully evaluated formatted string: {str(evaluated_result)[:100]}...")
                            responses.append(RLMResponse(
                                response_type=RLMResponseType.FINAL,
                                content=str(evaluated_result),
                                success=True
                            ))
                    else:
                        # Try to substitute variables with their values from REPL
                        logger.debug("Attempting variable substitution")
                        substituted_content = response_text
                        for var_match in variables:
                            var_name = var_match[1:-1]  # Remove { and }
                            var_value = self.repl.get_variable(var_name)
                            if var_value is not None:
                                substituted_content = substituted_content.replace(var_match, str(var_value))
                                logger.debug(f"Substituted {var_match} with {str(var_value)[:50]}...")
                        
                        if substituted_content != response_text:
                            logger.debug(f"Variable substitution successful: {substituted_content[:100]}...")
                            responses.append(RLMResponse(
                                response_type=RLMResponseType.FINAL,
                                content=substituted_content,
                                success=True
                            ))
                except Exception as e:
                    logger.debug(f"Error processing formatted string: {e}")
        
        # If no special response types found, treat as continue
        if not responses:
            logger.debug("No special response type found, treating as CONTINUE")
            responses.append(RLMResponse(
                response_type=RLMResponseType.CONTINUE,
                content=response_text,
                success=True
            ))
        
        return responses
    
    def _fix_f_string_quotes(self, f_string: str) -> str:
        """Fix common quote conflicts in f-strings."""
        try:
            # Handle different f-string formats
            if f_string.startswith('f"') and f_string.endswith('"'):
                # Remove f" prefix and " suffix
                content = f_string[2:-1]
                # Replace problematic quote patterns
                fixed_content = content.replace('", "', "', '")
                return f'f"{fixed_content}"'
            elif f_string.startswith("f'") and f_string.endswith("'"):
                # Remove f' prefix and ' suffix
                content = f_string[2:-1]
                # Replace problematic quote patterns
                fixed_content = content.replace("', '", '", "')
                return f"f'{fixed_content}'"
            elif f_string.startswith('f"""') and f_string.endswith('"""'):
                # Handle triple-quoted f-strings
                content = f_string[4:-3]
                # Try to fix common issues
                fixed_content = content.replace('"""', "'''")
                return f'f"""{fixed_content}"""'
            elif f_string.startswith("f'''") and f_string.endswith("'''"):
                # Handle triple-quoted f-strings
                content = f_string[4:-3]
                # Try to fix common issues
                fixed_content = content.replace("'''", '"""')
                return f"f'''{fixed_content}'''"
        except Exception as e:
            logger.debug(f"Error fixing f-string quotes: {e}")
        
        return f_string  # Return original if fixing fails
    
    def _extract_f_string_content(self, f_string: str) -> str:
        """Extract meaningful content from f-string as fallback."""
        import re  # Import at the top of the method
        
        try:
            # Try to extract the string content without the f-prefix
            if f_string.startswith('f"') and f_string.endswith('"'):
                content = f_string[2:-1]
            elif f_string.startswith("f'") and f_string.endswith("'"):
                content = f_string[2:-1]
            elif f_string.startswith('f"""') and f_string.endswith('"""'):
                content = f_string[4:-3]
            elif f_string.startswith("f'''") and f_string.endswith("'''"):
                content = f_string[4:-3]
            else:
                return ""
            
            # Try to extract variable references and convert them to readable text
            # This is a simple fallback - replace {variable} with [variable]
            # Find all {variable} patterns and replace with [variable]
            content = re.sub(r'\{([^}]+)\}', r'[\1]', content)
            
            # If the content looks like it has meaningful text, return it
            if len(content.strip()) > 0:
                return f"Formatted content: {content}"
            
        except Exception as e:
            logger.debug(f"Error extracting f-string content: {e}")
        
        return ""
    
    def rlm_call(self, query: str, depth: int = 1) -> str:
        """
        Make a recursive call to the RLM.
        This function is available in the REPL environment.
        
        Args:
            query: The query to process
            depth: Recursion depth
            
        Returns:
            Response from the recursive call
        """
        if not self.config.enable_recursive_calls:
            return "Recursive calls are disabled"
        
        if depth > self.config.max_recursion_depth:
            return "Maximum recursion depth reached"
        
        # Make recursive call
        messages = [{"role": "user", "content": query}]
        response = self._process_recursively(messages, depth=depth)
        
        if response.success:
            return response.content
        else:
            return f"Recursive call failed: {response.error}"
    
    def get_call_history(self) -> List[Dict]:
        """Get the history of all RLM calls."""
        return self.call_history
    
    def reset(self):
        """Reset the RLM state."""
        self.repl.reset_namespace()
        self.call_history = []
        self.current_depth = 0


def create_rlm(api_key: Optional[str] = None, config: RLMConfig = None) -> RecursiveLanguageModel:
    """
    Create a Recursive Language Model instance.
    
    Args:
        api_key: API key for the LLM client
        config: RLM configuration
        
    Returns:
        RecursiveLanguageModel instance
    """
    llm_client = GeminiClient(api_key=api_key)
    rlm = RecursiveLanguageModel(llm_client, config)
    
    # Add the rlm_call function to the REPL
    rlm.repl.set_variable("rlm_call", rlm.rlm_call)
    
    return rlm
