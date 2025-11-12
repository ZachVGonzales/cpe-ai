from abc import ABC, abstractmethod
from typing import Any, Dict

class BaseVectorStoreRetriever(ABC):
    @abstractmethod
    def __init__(self, *args, **kwargs):
        pass

    @abstractmethod
    def retrieve(self, query: str, k: int = 5, **kwargs) -> Dict[str, Any]:
        """
        Retrieve relevant documents or information from the vector store.
        Args:
            query: The query string to search for.
            k: Number of top results to return.
        Returns:
            A dictionary with retrieval results.
        """
        pass
