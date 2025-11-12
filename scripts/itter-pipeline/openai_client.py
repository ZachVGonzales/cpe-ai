"""
OpenAI API client and system prompt management.
"""

import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime

from openai import OpenAI

from .config import DEFAULT_SYSTEM_PROMPT, MAX_RETRIES, RETRY_DELAY


class OpenAIClient:
    """Client for interacting with OpenAI API."""

    def __init__(self, api_key: str, model: str, reasoning_effort: str = "high"):
        """
        Initialize OpenAI client.

        Args:
            api_key: OpenAI API key
            model: Model ID to use
            reasoning_effort: Reasoning effort level (low, medium, high)
        """
        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.reasoning_effort = reasoning_effort

    def call_api(self, system_prompt: str, user_prompt: str) -> Optional[str]:
        """
        Call OpenAI API with retry logic.

        Args:
            system_prompt: System prompt for the API
            user_prompt: User prompt with the problem

        Returns:
            Response content or None if failed
        """
        for attempt in range(MAX_RETRIES):
            try:
                # Build API parameters
                api_params = {
                    "model": self.model,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    "reasoning_effort": self.reasoning_effort,
                }

                response = self.client.chat.completions.create(**api_params)

                return response.choices[0].message.content

            except Exception as e:
                print(f"API call attempt {attempt + 1} failed: {e}")
                if attempt < MAX_RETRIES - 1:
                    time.sleep(RETRY_DELAY * (attempt + 1))
                else:
                    print("Max retries reached")
                    return None

        return None

    def create_batch_request(
        self,
        requests: List[Dict[str, str]],
        batch_dir: Path,
    ) -> Tuple[Optional[str], str]:
        """
        Create a batch API request file.

        Args:
            requests: List of request dicts with 'custom_id', 'system_prompt', 'user_prompt'
                     (optionally 'original_problem_id' for tracking duplicates)
            batch_dir: Directory to store batch files

        Returns:
            Tuple of (batch_id, input_file_path) or (None, error_message)
        """
        batch_dir.mkdir(parents=True, exist_ok=True)

        # Create JSONL file for batch API
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        input_file = batch_dir / f"batch_input_{timestamp}.jsonl"

        try:
            with open(input_file, "w") as f:
                for req in requests:
                    batch_request = {
                        "custom_id": req["custom_id"],
                        "method": "POST",
                        "url": "/v1/chat/completions",
                        "body": {
                            "model": self.model,
                            "messages": [
                                {"role": "system", "content": req["system_prompt"]},
                                {"role": "user", "content": req["user_prompt"]},
                            ],
                            "reasoning_effort": self.reasoning_effort,
                        },
                    }
                    f.write(json.dumps(batch_request) + "\n")

            print(f"📝 Created batch input file: {input_file}")
            print(f"   Total requests: {len(requests)}")

            # Upload the batch file
            with open(input_file, "rb") as f:
                batch_input_file = self.client.files.create(file=f, purpose="batch")

            print(f"📤 Uploaded batch file: {batch_input_file.id}")

            # Create the batch
            batch = self.client.batches.create(
                input_file_id=batch_input_file.id,
                endpoint="/v1/chat/completions",
                completion_window="24h",
            )

            print(f"🚀 Batch created: {batch.id}")
            print(f"   Status: {batch.status}")

            return batch.id, str(input_file)

        except Exception as e:
            print(f"Error creating batch: {e}")
            return None, str(e)

    def check_batch_status(self, batch_id: str) -> Tuple[str, Dict]:
        """
        Check the status of a batch request.

        Args:
            batch_id: The batch ID to check

        Returns:
            Tuple of (status, batch_info_dict)
        """
        try:
            batch = self.client.batches.retrieve(batch_id)

            info = {
                "id": batch.id,
                "status": batch.status,
                "created_at": batch.created_at,
                "completed_at": getattr(batch, "completed_at", None),
                "failed_at": getattr(batch, "failed_at", None),
                "request_counts": {
                    "total": batch.request_counts.total,
                    "completed": batch.request_counts.completed,
                    "failed": batch.request_counts.failed,
                },
            }

            if batch.status == "completed":
                info["output_file_id"] = batch.output_file_id
            if batch.status == "failed":
                if hasattr(batch, "error_file_id"):
                    info["error_file_id"] = batch.error_file_id
                if hasattr(batch, "errors") and batch.errors:
                    info["errors"] = batch.errors
                    # Print detailed error information
                    print(f"\nBatch failed with errors:")
                    if hasattr(batch.errors, "data"):
                        for error in batch.errors.data:
                            print(f"   - {error}")
                    else:
                        print(f"   - {batch.errors}")

            return batch.status, info

        except Exception as e:
            print(f"Error checking batch status: {e}")
            return "error", {"error": str(e)}

    def retrieve_batch_results(
        self, batch_id: str, output_file: Path
    ) -> Optional[Dict[str, str]]:
        """
        Retrieve results from a completed batch.

        Args:
            batch_id: The batch ID
            output_file: Path to save the raw output

        Returns:
            Dictionary mapping custom_id to response content, or None if failed
        """
        try:
            batch = self.client.batches.retrieve(batch_id)

            if batch.status != "completed":
                print(f"⚠️  Batch not completed yet. Status: {batch.status}")
                return None

            # Download the output file
            file_response = self.client.files.content(batch.output_file_id)

            # Save raw output
            output_file.parent.mkdir(parents=True, exist_ok=True)
            output_file.write_bytes(file_response.content)

            print(f"📥 Downloaded batch results to: {output_file}")

            # Parse results
            results = {}
            for line in file_response.content.decode("utf-8").strip().split("\n"):
                result = json.loads(line)
                custom_id = result["custom_id"]

                if result["response"]["status_code"] == 200:
                    content = result["response"]["body"]["choices"][0]["message"][
                        "content"
                    ]
                    results[custom_id] = content
                else:
                    print(
                        f"⚠️  Request {custom_id} failed: {result['response']['status_code']}"
                    )
                    results[custom_id] = None

            return results

        except Exception as e:
            print(f"Error retrieving batch results: {e}")
            return None

    def wait_for_batch(
        self, batch_id: str, check_interval: int = 60, max_wait: int = 86400
    ) -> str:
        """
        Wait for a batch to complete.

        Args:
            batch_id: The batch ID to wait for
            check_interval: Seconds between status checks (default: 60)
            max_wait: Maximum seconds to wait (default: 86400 = 24h)

        Returns:
            Final batch status
        """
        print(f"Waiting for batch {batch_id} to complete...")

        elapsed = 0
        while elapsed < max_wait:
            status, info = self.check_batch_status(batch_id)

            if status in ["completed", "failed", "cancelled", "expired"]:
                print(f"\nBatch {status}!")
                if status == "completed":
                    counts = info["request_counts"]
                    print(f"   Completed: {counts['completed']}/{counts['total']}")
                    if counts["failed"] > 0:
                        print(f"   Failed: {counts['failed']}")
                return status

            # Print progress
            counts = info["request_counts"]
            progress = (
                counts["completed"] / counts["total"] * 100
                if counts["total"] > 0
                else 0
            )
            print(
                f"   Progress: {counts['completed']}/{counts['total']} ({progress:.1f}%) - "
                f"Status: {status}",
                end="\r",
            )

            time.sleep(check_interval)
            elapsed += check_interval

        print(f"\n⚠️  Max wait time ({max_wait}s) exceeded")
        return "timeout"


def load_system_prompt(system_prompt_file: Optional[str] = None) -> str:
    """
    Load system prompt from file or use default.

    Args:
        system_prompt_file: Path to system prompt file (optional)

    Returns:
        System prompt string
    """
    if system_prompt_file:
        with open(system_prompt_file, "r") as f:
            return f.read()
    return DEFAULT_SYSTEM_PROMPT
