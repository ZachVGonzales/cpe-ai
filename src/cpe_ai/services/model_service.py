"""
Model Service for centralized model loading and management.

This service loads all models once at startup and provides them through dependency injection.
This approach improves performance by avoiding repeated model loading and ensures efficient 
memory usage.
"""
import logging
from typing import Optional, Any, Dict
import torch
import gc

from cpe_ai.config.settings import (
    EMBEDDINGS_MODEL,
    RERANKER_MODEL,
    HUGGINGFACE_CACHE_DIR,
)

from cpe_ai.rag.vector_store_base import BaseVectorStoreRetriever
from cpe_ai.rag.hf_vector_store import SentenceMxbaiRetriever

# Import transformer models and dependencies
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class ModelService:
    """
    Centralized service for loading and managing all AI models.
    
    This service loads models once at startup and provides them to other services
    through dependency injection. This approach:
    - Reduces startup time for individual requests
    - Ensures efficient memory usage
    - Provides centralized model management
    - Enables better error handling and fallbacks
    """
    
    def __init__(self):
        """Initialize the model service."""
        self.models_loaded = False
        self.load_errors = {}
        
        # Vector store models
        self.embedding_model: Optional[SentenceTransformer] = None
        self.reranker_model: Optional[Any] = None
        
        # Retriever instance
        self.retriever: Optional[BaseVectorStoreRetriever] = None

    def _get_device_with_fallback(self) -> str:
        """Get the best available device, with fallback to CPU if GPU memory is insufficient."""
        if torch.cuda.is_available():
            try:
                # Test if we can allocate memory on CUDA
                test_tensor = torch.zeros(1).cuda()
                del test_tensor
                torch.cuda.empty_cache()
                return "cuda"
            except Exception as e:
                logger.warning(f"CUDA available but memory allocation failed: {str(e)}")
                return "cpu"
        return "cpu"
    
    def load_all_models(self) -> Dict[str, Any]:
        """
        Load all models required by the application.
        
        Returns:
            Dictionary with loading status and any errors
        """
        logger.info("Starting model loading process...")
        results = {
            "qa_model": {"status": "pending", "error": None},
            "summarizer": {"status": "pending", "error": None},
            "embeddings": {"status": "pending", "error": None},
            "reranker": {"status": "pending", "error": None},
            "retriever": {"status": "pending", "error": None}
        }
        
        # Load embedding models
        try:
            logger.info("Loading embedding model...")
            self.load_embedding_model()
            results["embeddings"]["status"] = "success"
            logger.info("Embedding model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load embedding model: {str(e)}")
            results["embeddings"]["status"] = "error"
            results["embeddings"]["error"] = str(e)
            self.load_errors["embeddings"] = str(e)
        
        # Load reranker models
        try:
            logger.info("Loading reranker model...")
            self.load_reranker_model()
            results["reranker"]["status"] = "success"
            logger.info("Reranker model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load reranker model: {str(e)}")
            results["reranker"]["status"] = "error"
            results["reranker"]["error"] = str(e)
            self.load_errors["reranker"] = str(e)
        
        # Create retriever instance
        try:
            logger.info("Creating retriever instance...")
            self.create_retriever()
            results["retriever"]["status"] = "success"
            logger.info("Retriever created successfully")
        except Exception as e:
            logger.error(f"Failed to create retriever: {str(e)}")
            results["retriever"]["status"] = "error"
            results["retriever"]["error"] = str(e)
            self.load_errors["retriever"] = str(e)
        
        self.models_loaded = True
        success_count = sum(1 for r in results.values() if r["status"] == "success")
        total_count = len(results)
        logger.info(f"Model loading complete: {success_count}/{total_count} models loaded successfully")
        
        return results

    def load_embedding_model(self):
        """Public method to load embedding model."""
        logger.info(f"Loading embedding model: {EMBEDDINGS_MODEL}")
        
        device = self._get_device_with_fallback()
        self.embedding_model = SentenceTransformer(
            EMBEDDINGS_MODEL,
            cache_folder=HUGGINGFACE_CACHE_DIR,
            device=device
        ).eval()
        
        # Set padding token if needed
        if hasattr(self.embedding_model, "tokenizer") and self.embedding_model.tokenizer.pad_token is None:
            self.embedding_model.tokenizer.pad_token = (
                getattr(self.embedding_model.tokenizer, "eos_token", None)
                or getattr(self.embedding_model.tokenizer, "cls_token", None)
                or "[PAD]"
            )
            logger.info(f"Set padding token for embeddings tokenizer: {self.embedding_model.tokenizer.pad_token}")
        
        logger.info(f"Successfully loaded embedding model on device: {device}")

    def load_reranker_model(self):
        """Public method to load reranker model (if configured)."""
        if RERANKER_MODEL:
            logger.info(f"Loading MXBAI reranker: {RERANKER_MODEL}")
            try:
                from mxbai_rerank import MxbaiRerankV2
                self.reranker_model = MxbaiRerankV2(RERANKER_MODEL, cache_dir=HUGGINGFACE_CACHE_DIR)
                
                if hasattr(self.reranker_model, "tokenizer") and self.reranker_model.tokenizer.pad_token is None:
                    self.reranker_model.tokenizer.pad_token = (
                        getattr(self.reranker_model.tokenizer, "eos_token", None)
                        or getattr(self.reranker_model.tokenizer, "cls_token", None)
                        or "[PAD]"
                    )
                    logger.info(f"Set padding token for reranker tokenizer: {self.reranker_model.tokenizer.pad_token}")
                
                logger.info("Successfully loaded MXBAI reranker")
            except ImportError:
                logger.warning("MXBAI rerank library not found. Reranker will be disabled.")
                self.reranker_model = None
        else:
            logger.info("No reranker model specified, skipping reranker loading")
            self.reranker_model = None

    def create_retriever(self):
        """Public method to create retriever from preloaded components."""
        logger.info("Creating retriever with pre-loaded models...")
        
        device = self._get_device_with_fallback()
        self.retriever = SentenceMxbaiRetriever(
            embeddings_model=EMBEDDINGS_MODEL,
            reranker_model=RERANKER_MODEL,
            device=device,
            preloaded_embeddings=self.embedding_model,
            preloaded_reranker=self.reranker_model
        )
        
        logger.info("Successfully created retriever with pre-loaded models")
    
    def cleanup_models(self):
        """Clean up all loaded models to free memory."""
        logger.info("Cleaning up models...")
        
        models_to_cleanup = [
            ("qa_model", self.qa_model_obj),
            ("summarizer_model", self.summarizer_model),
            ("embedding_model", self.embedding_model),
            ("reranker_model", self.reranker_model)
        ]
        
        for name, model in models_to_cleanup:
            if model is not None:
                try:
                    del model
                    logger.info(f"Cleaned up {name}")
                except Exception as e:
                    logger.warning(f"Error cleaning up {name}: {str(e)}")
        
        # Clear pipelines
        pipelines_to_cleanup = [
            ("qa_pipeline", self.qa_pipeline),
            ("summarizer_pipeline", self.summarizer_pipeline)
        ]
        
        for name, pipeline_obj in pipelines_to_cleanup:
            if pipeline_obj is not None:
                try:
                    del pipeline_obj
                    logger.info(f"Cleaned up {name}")
                except Exception as e:
                    logger.warning(f"Error cleaning up {name}: {str(e)}")
        
        # Reset all attributes
        self.qa_model = None
        self.qa_tokenizer = None
        self.qa_model_obj = None
        self.qa_pipeline = None
        self.summarizer_model = None
        self.summarizer_tokenizer = None
        self.summarizer_pipeline = None
        self.embedding_model = None
        self.reranker_model = None
        self.retriever = None
        
        # Force garbage collection and clear CUDA cache
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        self.models_loaded = False
        logger.info("Model cleanup complete")
    
    def get_model_status(self) -> Dict[str, Any]:
        """Get the current status of all models."""
        return {
            "models_loaded": self.models_loaded,
            "load_errors": self.load_errors,
            "qa_model_loaded": self.qa_model is not None,
            "summarizer_loaded": self.summarizer_model is not None,
            "embedding_model_loaded": self.embedding_model is not None,
            "reranker_model_loaded": self.reranker_model is not None,
            "retriever_loaded": self.retriever is not None
        }


# Create global instance
model_service = ModelService()