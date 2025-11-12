"""
RAG (Retrieval-Augmented Generation) service for Service AI.
Provides methods to index documents and perform queries with retrieval augmentation.
"""
import logging
import traceback
import os
import time
from typing import Dict, Any, Optional, List
from jinja2 import FileSystemLoader
from jinja2.sandbox import ImmutableSandboxedEnvironment
from jinja2.ext import loopcontrols

from openai import OpenAI

from cpe_ai.config.settings import (
    VECTOR_STORE_PATH,
    CONTEXT_NAMES,
    DOCS_PER_CONTEXT,
    OPENAI_API_KEY,
    OPENAI_MODEL,
    PROMPT_TEMPLATE_PATH,
)

from cpe_ai.rag.vector_store_base import BaseVectorStoreRetriever
from cpe_ai.rag.hf_vector_store import SentenceMxbaiRetriever

logger = logging.getLogger(__name__)

def log_timing(operation_name: str, duration: float):
    """Log timing information in blue color"""
    # ANSI escape code for blue text
    blue_text = f"\033[94m[TIMING] {operation_name}: {duration:.3f}s\033[0m"
    logger.info(blue_text)

class RAGService:

    # Load from CWD and disable hot-reload for better performance. Otherwise, loosely mimics Transformers' environment
    jinja_environment = ImmutableSandboxedEnvironment(
        loader=FileSystemLoader("."), 
        auto_reload=False, 
        trim_blocks=True, 
        lstrip_blocks=True, 
        extensions=[loopcontrols]
    )
    
    """
    Service for handling RAG (Retrieval-Augmented Generation) operations.
    """
    def __init__(
        self, 
        retriever: Optional[BaseVectorStoreRetriever] = None,
        openai_api_key: Optional[str] = None,
        openai_model: Optional[str] = None,
        prompt_template_path: Optional[str] = None
    ):
        """
        Initialize the RAG service with necessary components.
        
        Args:
            retriever: Pre-loaded retriever instance or None
            openai_api_key: OpenAI API key (defaults to OPENAI_API_KEY from settings)
            openai_model: OpenAI model to use (defaults to OPENAI_MODEL from settings)
            prompt_template_path: Path to Jinja template for prompts (defaults to PROMPT_TEMPLATE_PATH from settings)
        """

        # Init retriever
        self.retriever = self._initialize_retriever(retriever)
        
        # Initialize OpenAI client
        api_key = openai_api_key or OPENAI_API_KEY
        if not api_key:
            raise ValueError("OpenAI API key must be provided or set in OPENAI_API_KEY environment variable")
        
        self.openai_client = OpenAI(api_key=api_key)
        self.openai_model = openai_model or OPENAI_MODEL
        
        # Load prompt template
        template_path = prompt_template_path or PROMPT_TEMPLATE_PATH
        if os.path.isfile(template_path):
            self.prompt_template = RAGService.jinja_environment.get_template(template_path)
        else:
            raise ValueError(f"Prompt template not found at {template_path}.")
        
        # Ensure paths exist
        os.makedirs(VECTOR_STORE_PATH, exist_ok=True)
        
        logger.info(f"RAG service initialized successfully with OpenAI model: {self.openai_model}")

    def _initialize_retriever(self, retriever: Optional[BaseVectorStoreRetriever]) -> BaseVectorStoreRetriever:
        """
        Initialize the retriever pair.
        
        Args:
            retriever: Pre-loaded retriever instance or None
            
        Returns:
            Retriever instance
        """
        if retriever:
            logger.info("Using pre-loaded retriever from dependency injection")
            return retriever
        else:
            logger.info("Creating new retriever instance (fallback)")
            return SentenceMxbaiRetriever()

    def _format_inputs(self, query: str, context: List[Dict[str, str]]) -> str:
        """
        Format inputs using the Jinja template.
        
        Args:
            query: The user's query
            context: List of context documents with 'source' and 'content' keys
            
        Returns:
            Formatted prompt string
        """
        return self.prompt_template.render(query=query, context=context)

    def _make_safe_collection_name(self, collection_name: str) -> str:
        """
        Create a safe collection name by removing special characters.
        
        Args:
            collection_name: Original collection name
            
        Returns:
            Safe collection name
        """
        return ''.join(c if c.isalnum() else '_' for c in collection_name)
            
    # TODO: Refactoring this to use context area instead of collection name
    def query(
        self,
        query: str,
        collection_names: Optional[List[str]] = None,
        k: int = 1000,
        n: int = 500,
    ) -> Dict[str, Any]:
        """
        Query documents using RAG with OpenAI API.
        
        Args:
            query: User query string
            collection_names: List of collection names to search (defaults to CONTEXT_NAMES from settings)
            k: Number of initial documents to retrieve before reranking
            n: Number of top documents to retrieve after reranking
            
        Returns:
            Dictionary containing the generated response and usage statistics
        """
        try:
            # Use default collection names if not provided
            if collection_names is None:
                collection_names = CONTEXT_NAMES

            context = []

            # Retrieve documents from all collections
            for collection_name in collection_names:
                safe_collection_name = self._make_safe_collection_name(str(collection_name))

                start_time = time.time()
                retrieval_result = self.retriever.retrieve(
                    query, 
                    collection_name=safe_collection_name, 
                    k=k, 
                    n=n
                )
                retrieval_duration = time.time() - start_time
                log_timing(f"Vector store retrieval for '{safe_collection_name}'", retrieval_duration)
                
                docs = retrieval_result["documents"]

                for j, doc in enumerate(docs):
                    if j >= DOCS_PER_CONTEXT:
                        break

                    # Prepare the context from retrieved documents
                    context.append({
                        "source": doc.metadata.get("source", "unknown"),
                        "content": doc.page_content,
                    })

            # Format the prompt using Jinja template
            prompt = self._format_inputs(query=query, context=context)

            # Log the formatted prompt
            logger.info(f"Formatted Prompt: {prompt}")

            # Call OpenAI API
            try:
                start_time = time.time()
                response = self.openai_client.chat.completions.create(
                    model=self.openai_model,
                    messages=[
                        {"role": "user", "content": prompt}
                    ],
                    reasoning_effort="medium"
                )
                generation_duration = time.time() - start_time
                log_timing("OpenAI API call", generation_duration)
                
                # Extract response text
                response_text = response.choices[0].message.content
                
                # Check if we got a valid response
                if not response_text or response_text.isspace():
                    logger.warning("No text was generated by the model, setting default message")
                    response_text = "I was unable to generate a response based on the provided context. Please try rephrasing your question."
                
                # Get actual token usage from OpenAI response
                usage = response.usage
                prompt_tokens = usage.prompt_tokens
                completion_tokens = usage.completion_tokens
                total_tokens = usage.total_tokens
                
                # Format the response
                result = {
                    "response": response_text,
                    "usage": {
                        "prompt_tokens": prompt_tokens,
                        "completion_tokens": completion_tokens,
                        "total_tokens": total_tokens
                    },
                    "context_sources": [doc["source"] for doc in context]
                }

                # Debug log the result
                logger.info(f"RAG query result: {result}")

                return result
                
            except Exception as e:
                # If API call fails, return error message
                logger.error(f"Error calling OpenAI API: {str(e)}")
                return {
                    "response": f"I found relevant information, but couldn't generate a response. Error: {str(e)}",
                    "usage": {
                        "prompt_tokens": 0,
                        "completion_tokens": 0,
                        "total_tokens": 0
                    },
                    "context_sources": [doc["source"] for doc in context],
                    "error": str(e)
                }
            
        except Exception as e:
            tb = traceback.format_exc()
            # Log error with traceback
            logger.error(f"Error performing RAG query: {e}\n{tb}")
            raise e