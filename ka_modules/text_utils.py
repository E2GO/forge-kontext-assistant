"""
Text processing utilities for Kontext Assistant
"""

import re
from typing import List, Optional

class DescriptionCleaner:
    """Efficient description cleaning using compiled regex patterns"""
    
    # Compile regex patterns once for efficiency
    PREFIX_PATTERN = re.compile(
        r'^(The image shows?|The photo shows?|This image shows?|'
        r'The picture shows?|It shows?|The scene shows?|The image is)\s*',
        re.IGNORECASE
    )
    
    PHRASE_PATTERN = re.compile(
        r',?\s*(giving it a dynamic feel|'
        r'creating an animated atmosphere|'
        r'giving the image an animated feel|'
        r'from the game\s*[^,]*|'
        r'as indicated by the title of the image[^,]*|'
        r'as if (?:he|she|they) (?:is|are) ready to take on the world|'
        r'and has a determined expression on (?:his|her|their) face|'
        r'animated,\s*)',
        re.IGNORECASE
    )
    
    # Additional cleanup patterns
    GAME_REFERENCE_PATTERN = re.compile(
        r'(?:from the game|as indicated by)[^,.]*[,.]?\s*',
        re.IGNORECASE
    )
    
    # Remove this pattern as it can cut off important parts of descriptions
    # REDUNDANT_PHRASES = re.compile(
    #     r'\b(appears to be|seems to be|looks like)\s+',
    #     re.IGNORECASE
    # )
    
    @staticmethod
    def clean(description: str) -> str:
        """
        Clean description by removing common prefixes and problematic phrases
        
        Args:
            description: Raw description text
            
        Returns:
            Cleaned description
        """
        if not description:
            return ""
        
        # Remove prefixes
        desc = DescriptionCleaner.PREFIX_PATTERN.sub('', description)
        
        # Remove problematic phrases
        desc = DescriptionCleaner.PHRASE_PATTERN.sub('', desc)
        
        # Remove game references
        desc = DescriptionCleaner.GAME_REFERENCE_PATTERN.sub('', desc)
        
        # Don't remove "appears to be" etc as they can be part of valid descriptions
        # desc = DescriptionCleaner.REDUNDANT_PHRASES.sub('', desc)
        
        # Clean up whitespace and punctuation
        desc = re.sub(r'\s+', ' ', desc)  # Multiple spaces to single
        desc = re.sub(r'\s*,\s*,\s*', ', ', desc)  # Double commas
        desc = re.sub(r'\s*\.\s*\.\s*', '. ', desc)  # Double periods
        desc = desc.strip(' ,.')
        
        # Capitalize first letter
        if desc and desc[0].islower():
            desc = desc[0].upper() + desc[1:]
        
        return desc

    @staticmethod
    def clean_tag_list(tags: List[str]) -> List[str]:
        """
        Clean a list of tags by removing articles and connectors
        
        Args:
            tags: List of tag strings
            
        Returns:
            Cleaned tag list
        """
        cleaned = []
        
        # Patterns to remove from tags
        tag_cleaners = [
            (r'^(a|an|the|with|and|there are|there is)\s+', ''),
            (r'\s+(in|at|on)$', ''),
            (r'^(in|at|on|with)\s+', '')
        ]
        
        for tag in tags:
            clean_tag = tag.strip()
            for pattern, replacement in tag_cleaners:
                clean_tag = re.sub(pattern, replacement, clean_tag, flags=re.IGNORECASE)
            
            clean_tag = clean_tag.strip()
            if clean_tag and len(clean_tag) > 1:
                cleaned.append(clean_tag)
        
        return cleaned


class TextExtractor:
    """Extract specific information from text using patterns"""
    
    @staticmethod
    def extract_colors(text: str) -> List[str]:
        """Extract color mentions from text"""
        color_pattern = re.compile(
            r'\b(red|blue|green|yellow|orange|purple|pink|brown|black|white|'
            r'gray|grey|gold|silver|bronze|cyan|magenta|turquoise|violet|'
            r'crimson|scarlet|emerald|sapphire|ruby|amber|jade)\b',
            re.IGNORECASE
        )
        return list(set(color_pattern.findall(text.lower())))
    
    @staticmethod
    def extract_numbers(text: str) -> List[str]:
        """Extract number mentions from text"""
        number_pattern = re.compile(r'\b(one|two|three|four|five|six|seven|eight|nine|ten|\d+)\b', re.IGNORECASE)
        return number_pattern.findall(text)
    
    @staticmethod
    def extract_actions(text: str) -> List[str]:
        """Extract action words from text"""
        action_pattern = re.compile(
            r'\b(walking|running|standing|sitting|flying|jumping|fighting|'
            r'holding|carrying|wearing|looking|smiling|crying|laughing|'
            r'reading|writing|dancing|sleeping|eating|drinking)\b',
            re.IGNORECASE
        )
        return list(set(action_pattern.findall(text.lower())))