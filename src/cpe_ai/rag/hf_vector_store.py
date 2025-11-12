from typing import Any, Dict, Optional
from .vector_store_base import BaseVectorStoreRetriever
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
import os
from cpe_ai.config.settings import HUGGINGFACE_CACHE_DIR, VECTOR_STORE_PATH, EMBEDDINGS_MODEL, RERANKER_MODEL, RERANK_BATCH_SIZE
import torch
import logging
from sentence_transformers import SentenceTransformer, CrossEncoder # for reranking
from pathlib import Path
from typing import List, Optional, Any
import gc

logger = logging.getLogger(__name__)

class HuggingFaceVectorStoreRetriever(BaseVectorStoreRetriever):
    def __init__(self, embeddings_model: Optional[str] = None):
        self.embeddings_model = embeddings_model or EMBEDDINGS_MODEL
        self.embeddings = None

        
        self.reranker = CrossEncoder(RERANKER_MODEL)
        self._initialize_embeddings()

    def _initialize_embeddings(self):
        if self.embeddings is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            self.embeddings = HuggingFaceEmbeddings(
                model_name=self.embeddings_model,
                model_kwargs={'device': device},
                encode_kwargs={'normalize_embeddings': True}
            )
            logger.info(f"Initialized HuggingFace embeddings on {device}")    
    
    def retrieve(self, query: str, collection_name: str, k: int = 5, **kwargs) -> Dict[str, Any]:
        # Use Path to create a normalized path with forward slashes and ensure it starts with ./
        base_path = Path(VECTOR_STORE_PATH)
        
        # collection_path = f"{base_path}\\{collection_name}"
        collection_path = os.path.join(base_path, collection_name)
        
        
        # Convert to Path object for existence check (handles platform differences)
        path_obj = Path(collection_path)
        if not path_obj.exists():
            raise ValueError(f"Collection '{collection_name}' does not exist at {collection_path}")
            
        vector_store = Chroma(
            collection_name=collection_name,
            embedding_function=self.embeddings,
            persist_directory=collection_path
        )
        docs = vector_store.similarity_search(query, k=k)

        #if docs is empty
        if not docs:
            logger.error(f"No documents found for query: {query} in collection: {collection_name}")
            return {"documents": []}
        
        # Prepare (query, doc) pairs for re-ranking
        pairs = [(query, doc.page_content) for doc in docs]
        scores = self.reranker.predict(pairs)
        reranked = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
        docs = [doc for doc, score in reranked]

        
        return {
            "documents": docs
        }


class SentenceMxbaiRetriever(BaseVectorStoreRetriever):
    """
    Retriever implementation using SentenceTransformers for embedding 
    and MXBAI for reranking.
    
    This implementation follows a two-stage approach:
    1. Initial retrieval using semantic similarity with embeddings
    2. Reranking of top candidates using MXBAI reranker
    """
    
    def __init__(self, 
                 embeddings_model: Optional[str] = None,
                 reranker_model: Optional[str] = None,
                 device: Optional[str] = None,
                 preloaded_embeddings: Optional[Any] = None,
                 preloaded_reranker: Optional[Any] = None):
        """
        Initialize the Sentence-MXBAI retriever.
        
        Args:
            embedding_model_name: Name/path of the SentenceTransformer model
            reranker_model_name: Name/path of the MXBAI reranker model
            device: Device to use for models
            preloaded_embeddings: Pre-loaded embedding model to avoid duplicate loading
            preloaded_reranker: Pre-loaded reranker model to avoid duplicate loading
        """
        self.embeddings_model = embeddings_model or EMBEDDINGS_MODEL
        self.reranker_model = reranker_model or RERANKER_MODEL
        self.device = None
        self.device = device or self._get_device_with_fallback()
        self.embeddings = preloaded_embeddings
        self.reranker = preloaded_reranker

        # Only initialize models if they weren't pre-loaded
        if self.embeddings is None:
            self._initialize_embeddings()
        else:
            logger.info("Using pre-loaded embedding model")
            
        if self.reranker is None:
            self._initialize_reranker()
        else:
            logger.info("Using pre-loaded reranker model")
        
    def _initialize_embeddings(self):
        # Initialize embedding model
        logger.info(f"Loading embedding model: {self.embeddings_model}")
        try:
            self.embeddings = SentenceTransformer(
                self.embeddings_model, 
                cache_folder=HUGGINGFACE_CACHE_DIR,  # Use cache directory for embeddings
                device=self.device
            ).eval()
            if hasattr(self.embeddings, "tokenizer") and self.embeddings.tokenizer.pad_token is None:
                # Try to set to eos_token, cls_token, or fallback to [PAD]
                self.embeddings.tokenizer.pad_token = (
                    getattr(self.embeddings.tokenizer, "eos_token", None)
                    or getattr(self.embeddings.tokenizer, "cls_token", None)
                    or "[PAD]"
                )
                logger.info(f"Set padding token for embeddings tokenizer: {self.embeddings.tokenizer.pad_token}")
        except Exception as e:
            logger.error(f"Failed to load embedding model {self.embeddings_model} on {self.device}. Error: {str(e)}")

            if "out of memory" in str(e).lower() and self.device == "cuda":
                logger.warning(f"GPU memory insufficient for {self.embeddings_model}. Falling back to CPU.")
                if self.embeddings is not None:
                    del self.embeddings
                self._clear_torch_cache()
                self.device = "cpu"

                self.embeddings = SentenceTransformer(
                    self.embeddings_model, 
                    cache_folder=HUGGINGFACE_CACHE_DIR,  # Use cache directory for embeddings
                    device=self.device
                ).eval()
            else:
                raise e

    def _initialize_reranker(self):
        # Initialize reranker model
        if self.reranker_model:
            logger.info(f"Loading MXBAI reranker: {self.reranker_model}")
            try:
                from mxbai_rerank import MxbaiRerankV2
                self.reranker = MxbaiRerankV2(self.reranker_model, cache_dir=HUGGINGFACE_CACHE_DIR)
                if hasattr(self.reranker, "tokenizer") and self.reranker.tokenizer.pad_token is None:
                    # Try to set to eos_token, cls_token, or fallback to [PAD]
                    self.reranker.tokenizer.pad_token = (
                        getattr(self.reranker.tokenizer, "eos_token", None)
                        or getattr(self.reranker.tokenizer, "cls_token", None)
                        or "[PAD]"
                    )
                    logger.info(f"Set padding token for reranker tokenizer: {self.reranker.tokenizer.pad_token}")
            except ImportError:
                logger.error("mxbai_rerank package not found. Please install it to use MXBAI reranking.")
                raise ImportError("mxbai_rerank package is required for this retriever")
            except Exception as e:
                logger.error(f"Failed to load MXBAI reranker {self.reranker_model}: {str(e)}")
                raise e
        else:
            self.reranker = None

    def _get_device_with_fallback(self) -> str:
        """Get the best available device, with fallback to CPU if GPU memory is insufficient."""
        if self.device:
            return self.device
            
        if torch.cuda.is_available():
            try:
                # Test CUDA availability with a small tensor
                test_tensor = torch.randn(100, 100).cuda()
                del test_tensor
                torch.cuda.empty_cache()
                return "cuda"
            except RuntimeError as e:
                logger.warning(f"CUDA available but failed memory test: {e}. Falling back to CPU.")
                return "cpu"
        return "cpu"
    
    def _clear_torch_cache(self):
        """Clear PyTorch CUDA cache and run garbage collection."""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
    
    def _cleanup_model(self, model):
        """Safely cleanup model and free GPU memory."""
        if model is not None:
            if hasattr(model, 'to'):
                model.to('cpu')
            del model
        self._clear_torch_cache()


    def embed_documents(self, documents: List[str]) -> List[List[float]]:
        """
        Embed documents into embeddings using the SentenceTransformer model.

        Args:
            documents: List of documents to embed

        Returns:
            List of document embeddings
        """
        if not self.embeddings:
            raise RuntimeError("Embedding model not initialized. Call initialize_models() first.")
        
        logger.info(f"Embedding {len(documents)} documents...")
        embeddings = self.embeddings.encode(documents).tolist()
        logger.info(f"Successfully embedded documents")
        return embeddings

    def embed_query(self, query: str) -> List[float]:
        """
        Embed a query string into an embedding vector.

        Args:
            query: The query string to embed

        Returns:
            Embedding vector for the query
        """
        if not self.embeddings:
            raise RuntimeError("Embedding model not initialized. Call initialize_models() first.")
        
        logger.info(f"Embedding query: {query}")
        embedding = self.embeddings.encode(query).tolist()
        logger.info(f"Successfully embedded query")
        return embedding

    def retrieve(self, query: str, collection_name: str, k: int = 50, n: int = 5, **kwargs) -> dict[str, list[Document]]:
        """
        Retrieve the most relevant documents for a query using two-stage retrieval.
        
        Args:
            query: The search query
            collection_name: Name of the collection to search in
            k: Number of initial prospect documents to retrieve before reranking
            n: Number of top documents to retrieve after reranking

        Returns:
            List of retrieved documents, ordered by relevance
        """
        base_path = Path(VECTOR_STORE_PATH)
        collection_path = os.path.join(base_path, collection_name)

        path_obj = Path(collection_path)
        if not path_obj.exists():
            raise ValueError(f"Collection '{collection_name}' does not exist at {collection_path}")

        # Stage 1: Initial retrieval using embeddings
        vector_store = Chroma(
            collection_name=collection_name,
            embedding_function=self,
            persist_directory=collection_path
        )
        docs = vector_store.similarity_search(query, k=k)

        #if docs is empty
        if not docs:
            logger.error(f"No documents found for query: {query} in collection: {collection_name}")
            return {"documents": []}
        
        # Stage 2: Reranking (if reranker is available)
        if self.reranker and len(docs) > 1:
            try:
                # Use MXBAI reranker
                logger.info(f"Reranking {len(docs)} initial documents...")
                # Extract page_content from Document objects for reranking
                doc_contents = [doc.page_content for doc in docs]
                results = []
                for i in range(0, len(doc_contents), RERANK_BATCH_SIZE):
                    batch = doc_contents[i:i+RERANK_BATCH_SIZE]
                    batch_results = self.reranker.rank(query, batch, return_documents=True, top_k=min(n, len(batch)))
                    results.extend(batch_results)
                # Sort by score and extract documents
                results.sort(key=lambda x: x.score, reverse=True)
                
                # Map reranked strings back to original Document objects
                # TODO: this is a bit inefficient; may need to change later
                content_to_doc = {doc.page_content: doc for doc in docs}
                reranked_docs = []
                for result in results[:n]:
                    original_doc = content_to_doc.get(result.document)
                    if original_doc:
                        reranked_docs.append(original_doc)
                
                logger.info(f"Reranking completed. Returning top {n} documents.")
                logger.info(f"Reranked documents: {[doc.page_content[:500] + '...' if len(doc.page_content) > 500 else doc.page_content for doc in reranked_docs]}")
                return {"documents": reranked_docs}
            except Exception as e:
                logger.warning(f"Reranking failed: {str(e)}. Falling back to embedding-only retrieval.")
        
        # log the retrieved documents for debugging
        # logger.info(f"Retrieved documents: {[doc.page_content for doc in docs[:n]]}")

        # Fallback: Return top-n from embedding similarity only
        return {"documents": docs[:n]}

    def cleanup(self) -> None:
        """
        Clean up resources and free memory.
        """
        logger.info("Cleaning up retriever resources...")
        
        # Clean up embedding model
        if self.embedding_model:
            self._cleanup_model(self.embedding_model)
            self.embedding_model = None
        
        # Clean up reranker model
        if self.reranker_model:
            del self.reranker_model
            self.reranker_model = None
        
        # Clean up document embeddings
        if self._document_embeddings is not None:
            del self._document_embeddings
            self._document_embeddings = None
        
        self._clear_torch_cache()
        logger.info("Cleanup completed.")
    
    def update_documents(self, documents: List[str]) -> None:
        """
        Update the document collection and re-encode if models are initialized.
        
        Args:
            documents: New list of documents to search through
        """
        super().update_documents(documents)
        
        # Re-encode documents if embedding model is available
        if self.embedding_model and documents:
            self._document_embeddings = self.encode_documents(documents)
        else:
            self._document_embeddings = None
    
    def get_model_info(self) -> dict:
        """
        Get information about the loaded models.
        
        Returns:
            Dictionary with model information
        """
        return {
            "embedding_model": self.embedding_model_name,
            "reranker_model": self.reranker_model_name,
            "device": self.device,
            "num_documents": len(self.documents),
            "models_initialized": self.embedding_model is not None,
            "embeddings_computed": self._document_embeddings is not None
        }
