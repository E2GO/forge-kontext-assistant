"""
Token counting utilities for FLUX.1 Kontext prompts.
FLUX.1 Kontext has a maximum prompt length of 512 tokens.
"""

import logging
import re
from typing import Tuple

logger = logging.getLogger(__name__)

# FLUX.1 Kontext uses T5 tokenizer, approximate token counting
# Average: 1 token ≈ 3-4 characters for English text
# This is a simplified approximation without external dependencies
APPROXIMATE_CHARS_PER_TOKEN = 3.5
MAX_TOKENS_KONTEXT = 512
WARNING_THRESHOLD = 450  # Warn when approaching limit


class TokenCounter:
    """Simple token counter for FLUX.1 Kontext prompts."""
    
    @staticmethod
    def estimate_tokens(text: str) -> int:
        """
        Estimate token count for a given text.
        
        This is a simplified estimation based on character count and punctuation.
        For more accurate counting, use the actual T5 tokenizer.
        
        Args:
            text: The prompt text to count tokens for
            
        Returns:
            Estimated token count
        """
        if not text:
            return 0
            
        # Clean up the text
        text = text.strip()
        
        # Count words and punctuation separately as they often tokenize differently
        words = len(re.findall(r'\b\w+\b', text))
        punctuation = len(re.findall(r'[.,!?;:()"\'-]', text))
        
        # Estimate based on character count with adjustments
        char_based_estimate = len(text) / APPROXIMATE_CHARS_PER_TOKEN
        
        # Adjust for punctuation (often separate tokens)
        punctuation_adjustment = punctuation * 0.5
        
        # Use the higher estimate to be conservative
        estimated_tokens = max(char_based_estimate, words + punctuation_adjustment)
        
        return int(estimated_tokens)
    
    @staticmethod
    def validate_prompt_length(prompt: str) -> Tuple[bool, str, int]:
        """
        Validate if prompt is within FLUX.1 Kontext's 512 token limit.
        
        Args:
            prompt: The prompt to validate
            
        Returns:
            Tuple of (is_valid, message, token_count)
        """
        token_count = TokenCounter.estimate_tokens(prompt)
        
        if token_count > MAX_TOKENS_KONTEXT:
            return (False, 
                   f"Prompt exceeds maximum length ({token_count} tokens > {MAX_TOKENS_KONTEXT} tokens)", 
                   token_count)
        elif token_count > WARNING_THRESHOLD:
            return (True, 
                   f"Prompt is approaching maximum length ({token_count}/{MAX_TOKENS_KONTEXT} tokens)", 
                   token_count)
        else:
            return (True, 
                   f"Prompt length OK ({token_count}/{MAX_TOKENS_KONTEXT} tokens)", 
                   token_count)
    
    @staticmethod
    def truncate_to_token_limit(prompt: str, max_tokens: int = MAX_TOKENS_KONTEXT) -> str:
        """
        Truncate prompt to fit within token limit.
        
        Args:
            prompt: The prompt to truncate
            max_tokens: Maximum allowed tokens
            
        Returns:
            Truncated prompt
        """
        if TokenCounter.estimate_tokens(prompt) <= max_tokens:
            return prompt
        
        # Binary search for the right truncation point
        left, right = 0, len(prompt)
        best_length = 0
        
        while left <= right:
            mid = (left + right) // 2
            truncated = prompt[:mid]
            
            if TokenCounter.estimate_tokens(truncated) <= max_tokens:
                best_length = mid
                left = mid + 1
            else:
                right = mid - 1
        
        # Try to truncate at a word boundary
        truncated = prompt[:best_length]
        last_space = truncated.rfind(' ')
        if last_space > best_length * 0.9:  # If we're not losing too much
            truncated = truncated[:last_space]
        
        return truncated.strip()
    
    @staticmethod
    def get_token_info_display(prompt: str) -> str:
        """
        Get a formatted display of token information.
        
        Args:
            prompt: The prompt to analyze
            
        Returns:
            Formatted string with token info
        """
        token_count = TokenCounter.estimate_tokens(prompt)
        percentage = (token_count / MAX_TOKENS_KONTEXT) * 100
        
        # Create visual indicator
        if percentage >= 100:
            status = "❌ EXCEEDS LIMIT"
            color = "🔴"
        elif percentage >= 88:  # Warning at ~450 tokens
            status = "⚠️ APPROACHING LIMIT"
            color = "🟡"
        else:
            status = "✅ OK"
            color = "🟢"
        
        return f"{color} Tokens: {token_count}/{MAX_TOKENS_KONTEXT} ({percentage:.0f}%) {status}"


# Utility functions for easy import
def count_tokens(text: str) -> int:
    """Count tokens in text."""
    return TokenCounter.estimate_tokens(text)


def validate_prompt(prompt: str) -> Tuple[bool, str, int]:
    """Validate prompt length."""
    return TokenCounter.validate_prompt_length(prompt)


def truncate_prompt(prompt: str, max_tokens: int = MAX_TOKENS_KONTEXT) -> str:
    """Truncate prompt to token limit."""
    return TokenCounter.truncate_to_token_limit(prompt, max_tokens)


def get_token_display(prompt: str) -> str:
    """Get formatted token display."""
    return TokenCounter.get_token_info_display(prompt)