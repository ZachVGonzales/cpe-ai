"""
OpenAI API client and system prompt management.
"""

import time
from typing import Optional

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
