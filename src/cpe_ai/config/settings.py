import os
from pathlib import Path
from dotenv import load_dotenv

env_path = Path(__file__).parent.parent.parent.parent / '.env'
load_dotenv(env_path)

# Vector store and embeddings settings
HUGGINGFACE_CACHE_DIR = Path(os.getenv("HUGGINGFACE_CACHE_DIR", "./cache/huggingface"))
VECTOR_STORE_PATH = Path(os.getenv("VECTOR_STORE_PATH", "./data/vector-store/databases/"))
EMBEDDINGS_MODEL = os.getenv("EMBEDDINGS_MODEL", "mixedbread-ai/mxbai-embed-large-v1")
RERANKER_MODEL = os.getenv("RERANKER_MODEL", "mixedbread-ai/mxbai-rerank-large-v2")
RERANK_BATCH_SIZE = int(os.getenv("RERANK_BATCH_SIZE", 16))
DEFAULT_QA_MODEL = os.getenv("DEFAULT_QA_MODEL", "ibm-granite/granite-3.3-8b-base")

# OpenAI settings
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-5")
PROMPT_TEMPLATE_PATH = os.getenv("PROMPT_TEMPLATE_PATH", "data/system-prompts/lean.jinja")

# Logging
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

# Context settings
CONTEXT_NAMES = os.getenv("CONTEXT_NAMES", "lean-api,lean-info").split(",")
DOCS_PER_CONTEXT = int(os.getenv("DOCS_PER_CONTEXT", 50))

# Debug mode
DEBUG = os.getenv("DEBUG", "False").lower() in ("true", "1", "yes")