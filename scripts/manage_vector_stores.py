#!/usr/bin/env python3
"""
Utility script to manage OpenAI vector stores for the iterative pipeline.

Usage:
    python scripts/manage_vector_stores.py list         # List all vector stores
    python scripts/manage_vector_stores.py clear        # Clear cached vector store IDs (forces re-upload)
    python scripts/manage_vector_stores.py delete <id>  # Delete a specific vector store
    python scripts/manage_vector_stores.py cleanup      # Delete old vector stores and clear cache
"""

import json
import sys
from pathlib import Path

from openai import OpenAI

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from itter_pipeline.config import get_api_key

VECTOR_STORE_CACHE_FILE = Path(__file__).parent.parent / ".vector_store_cache.json"


def list_vector_stores(client):
    """List all vector stores."""
    print("\n" + "="*80)
    print("Vector Stores")
    print("="*80)
    
    vector_stores = client.vector_stores.list()
    
    if not vector_stores.data:
        print("No vector stores found.")
        return
    
    for vs in vector_stores.data:
        print(f"\nID: {vs.id}")
        print(f"Name: {vs.name}")
        print(f"Files: {vs.file_counts.total}")
        print(f"Created: {vs.created_at}")
    
    print("\n" + "="*80)


def clear_cache():
    """Clear the cached vector store IDs."""
    if VECTOR_STORE_CACHE_FILE.exists():
        VECTOR_STORE_CACHE_FILE.unlink()
        print(f"✓ Cleared cache: {VECTOR_STORE_CACHE_FILE}")
        print("  Next run will create new vector stores and upload all files.")
    else:
        print(f"No cache file found at {VECTOR_STORE_CACHE_FILE}")


def delete_vector_store(client, vs_id):
    """Delete a specific vector store."""
    try:
        client.vector_stores.delete(vs_id)
        print(f"✓ Deleted vector store: {vs_id}")
    except Exception as e:
        print(f"✗ Failed to delete {vs_id}: {e}")


def cleanup(client):
    """Delete all vector stores and clear cache."""
    print("\n" + "="*80)
    print("Cleanup: Deleting all vector stores and clearing cache")
    print("="*80 + "\n")
    
    vector_stores = client.vector_stores.list()
    
    if not vector_stores.data:
        print("No vector stores to delete.")
    else:
        for vs in vector_stores.data:
            print(f"Deleting: {vs.name} ({vs.id})...")
            delete_vector_store(client, vs.id)
    
    print()
    clear_cache()
    print("\n✓ Cleanup complete!")


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    
    command = sys.argv[1]
    
    # Get API key
    api_key = get_api_key()
    client = OpenAI(api_key=api_key)
    
    if command == "list":
        list_vector_stores(client)
    
    elif command == "clear":
        clear_cache()
    
    elif command == "delete":
        if len(sys.argv) < 3:
            print("Error: delete command requires a vector store ID")
            print("Usage: python scripts/manage_vector_stores.py delete <id>")
            sys.exit(1)
        vs_id = sys.argv[2]
        delete_vector_store(client, vs_id)
    
    elif command == "cleanup":
        confirm = input("This will delete ALL vector stores and clear the cache. Continue? (yes/no): ")
        if confirm.lower() == "yes":
            cleanup(client)
        else:
            print("Cancelled.")
    
    else:
        print(f"Unknown command: {command}")
        print(__doc__)
        sys.exit(1)


if __name__ == "__main__":
    main()
