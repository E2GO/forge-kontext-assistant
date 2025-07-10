"""
Validation script for PromptGen v2.0 output
Ensures output matches expected format from documentation
"""

import logging
import re
from typing import Dict, Any, List, Tuple
from PIL import Image
import json
from pathlib import Path

# Add parent directory to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from ka_modules.image_analyzer import ImageAnalyzer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PromptGenValidator:
    """Validates PromptGen v2.0 output against expected formats"""
    
    # Expected output patterns based on documentation
    EXPECTED_PATTERNS = {
        "<GENERATE_TAGS>": {
            "type": "tags",
            "format": "comma_separated",
            "min_tags": 5,
            "max_tags": 50,
            "examples": [
                "1girl, solo, long hair, blue eyes, dress, standing, outdoors",
                "scenery, landscape, mountain, sky, clouds, sunset, nature",
                "abstract, colorful, digital art, pattern, geometric"
            ]
        },
        "<CAPTION>": {
            "type": "single_line",
            "min_length": 20,
            "max_length": 200,
            "examples": [
                "A young woman with long hair standing in a field",
                "Abstract colorful pattern with geometric shapes"
            ]
        },
        "<DETAILED_CAPTION>": {
            "type": "structured",
            "min_length": 50,
            "max_length": 500,
            "contains": ["subject", "position", "appearance"]
        },
        "<MORE_DETAILED_CAPTION>": {
            "type": "paragraph",
            "min_length": 100,
            "max_length": 1000,
            "quality_markers": ["detailed", "comprehensive", "descriptive"]
        },
        "<ANALYZE>": {
            "type": "analysis",
            "min_length": 100,
            "contains": ["composition", "lighting", "mood", "style"]
        },
        "<MIXED_CAPTION>": {
            "type": "mixed",
            "min_length": 150,
            "contains_tags": True,
            "contains_description": True
        },
        "<MIXED_CAPTION_PLUS>": {
            "type": "mixed_plus",
            "min_length": 200,
            "contains_tags": True,
            "contains_description": True,
            "contains_analysis": True
        }
    }
    
    def __init__(self):
        self.analyzer = None
        self.validation_results = {}
    
    def setup_analyzer(self):
        """Initialize PromptGen analyzer"""
        self.analyzer = ImageAnalyzer(
            device='cuda' if torch.cuda.is_available() else 'cpu',
            model_type='promptgen_v2'
        )
        self.analyzer._ensure_initialized()
    
    def validate_tags_format(self, text: str, task: str) -> Tuple[bool, List[str]]:
        """Validate tags format"""
        issues = []
        
        if not text:
            issues.append("Empty tags output")
            return False, issues
        
        # Check if it's comma-separated
        tags = [t.strip() for t in text.split(',')]
        
        # Check tag count
        expected = self.EXPECTED_PATTERNS[task]
        if len(tags) < expected.get("min_tags", 1):
            issues.append(f"Too few tags: {len(tags)} < {expected['min_tags']}")
        if len(tags) > expected.get("max_tags", 100):
            issues.append(f"Too many tags: {len(tags)} > {expected['max_tags']}")
        
        # Check tag format (should be lowercase with underscores)
        for tag in tags[:10]:  # Check first 10 tags
            if not tag:
                continue
            # Danbooru tags typically use underscores, not spaces
            if ' ' in tag and '_' not in tag:
                issues.append(f"Tag '{tag}' should use underscores")
            # Check for valid characters
            if not re.match(r'^[a-zA-Z0-9_\-\(\):\s]+$', tag):
                issues.append(f"Tag '{tag}' contains invalid characters")
        
        return len(issues) == 0, issues
    
    def validate_caption_format(self, text: str, task: str) -> Tuple[bool, List[str]]:
        """Validate caption format"""
        issues = []
        
        if not text:
            issues.append("Empty caption output")
            return False, issues
        
        expected = self.EXPECTED_PATTERNS[task]
        
        # Check length
        if len(text) < expected.get("min_length", 0):
            issues.append(f"Caption too short: {len(text)} < {expected['min_length']}")
        if len(text) > expected.get("max_length", 10000):
            issues.append(f"Caption too long: {len(text)} > {expected['max_length']}")
        
        # Check content requirements
        if "contains" in expected:
            for required in expected["contains"]:
                if required.lower() not in text.lower():
                    issues.append(f"Missing expected content: '{required}'")
        
        # Check for complete sentences
        if expected["type"] in ["paragraph", "structured"]:
            if not text.endswith(('.', '!', '?')):
                issues.append("Caption should end with punctuation")
        
        return len(issues) == 0, issues
    
    def validate_mixed_format(self, text: str, task: str) -> Tuple[bool, List[str]]:
        """Validate mixed caption format"""
        issues = []
        
        if not text:
            issues.append("Empty mixed caption output")
            return False, issues
        
        expected = self.EXPECTED_PATTERNS[task]
        
        # Check if it contains both tags and description
        # Tags are usually at the beginning or end, separated by commas
        has_tags = ',' in text and any(
            re.search(r'\b(1girl|1boy|solo|scenery|portrait|landscape)\b', text.lower())
        )
        
        # Description should have complete sentences
        has_description = any(text.count(sentence_end) > 0 for sentence_end in ['. ', '! ', '? '])
        
        if expected.get("contains_tags") and not has_tags:
            issues.append("Mixed caption missing tags section")
        
        if expected.get("contains_description") and not has_description:
            issues.append("Mixed caption missing description section")
        
        # Check length
        if len(text) < expected.get("min_length", 0):
            issues.append(f"Mixed caption too short: {len(text)} < {expected['min_length']}")
        
        return len(issues) == 0, issues
    
    def validate_analysis_format(self, text: str, task: str) -> Tuple[bool, List[str]]:
        """Validate analysis format"""
        issues = []
        
        if not text:
            issues.append("Empty analysis output")
            return False, issues
        
        expected = self.EXPECTED_PATTERNS[task]
        
        # Check for analysis keywords
        analysis_keywords = ["composition", "lighting", "color", "mood", "style", "subject"]
        found_keywords = sum(1 for kw in analysis_keywords if kw.lower() in text.lower())
        
        if found_keywords < 2:
            issues.append(f"Analysis lacks depth - only {found_keywords} analysis aspects found")
        
        # Check length
        if len(text) < expected.get("min_length", 0):
            issues.append(f"Analysis too short: {len(text)} < {expected['min_length']}")
        
        return len(issues) == 0, issues
    
    def validate_output(self, output: Any, task: str) -> Tuple[bool, List[str]]:
        """Validate output based on task type"""
        if task not in self.EXPECTED_PATTERNS:
            return False, [f"Unknown task: {task}"]
        
        # Extract text from output
        if isinstance(output, dict):
            # Try to get the result from various possible keys
            text = (output.get(task) or 
                   output.get(task.strip('<>')) or
                   list(output.values())[0] if len(output) == 1 else '')
        else:
            text = str(output) if output else ''
        
        if not text:
            return False, ["No text content found in output"]
        
        # Validate based on expected type
        expected_type = self.EXPECTED_PATTERNS[task]["type"]
        
        if expected_type == "tags":
            return self.validate_tags_format(text, task)
        elif expected_type in ["single_line", "paragraph", "structured"]:
            return self.validate_caption_format(text, task)
        elif expected_type in ["mixed", "mixed_plus"]:
            return self.validate_mixed_format(text, task)
        elif expected_type == "analysis":
            return self.validate_analysis_format(text, task)
        else:
            return False, [f"Unknown validation type: {expected_type}"]
    
    def run_validation_suite(self, test_image: Image.Image):
        """Run complete validation suite"""
        logger.info("=" * 60)
        logger.info("PROMPTGEN V2.0 OUTPUT VALIDATION")
        logger.info("=" * 60)
        
        if not self.analyzer:
            self.setup_analyzer()
        
        results = {}
        
        # Test each task
        for task in self.EXPECTED_PATTERNS.keys():
            logger.info(f"\nValidating task: {task}")
            
            try:
                # Run the task
                output = self.analyzer._run_florence_task(test_image, task)
                
                # Validate output
                is_valid, issues = self.validate_output(output, task)
                
                results[task] = {
                    "valid": is_valid,
                    "issues": issues,
                    "output_preview": str(output)[:200] if output else "None"
                }
                
                # Log results
                if is_valid:
                    logger.info(f"✓ {task} output is valid")
                else:
                    logger.warning(f"✗ {task} validation failed:")
                    for issue in issues:
                        logger.warning(f"  - {issue}")
                
                # Log preview
                if output:
                    if isinstance(output, dict):
                        text = list(output.values())[0] if output else ''
                    else:
                        text = str(output)
                    logger.info(f"  Preview: {text[:100]}...")
                
            except Exception as e:
                logger.error(f"Error running task {task}: {e}")
                results[task] = {
                    "valid": False,
                    "issues": [f"Runtime error: {str(e)}"],
                    "output_preview": "Error"
                }
        
        # Summary
        logger.info("\n" + "=" * 60)
        logger.info("VALIDATION SUMMARY")
        logger.info("=" * 60)
        
        valid_count = sum(1 for r in results.values() if r["valid"])
        total_count = len(results)
        
        logger.info(f"Valid outputs: {valid_count}/{total_count}")
        
        for task, result in results.items():
            status = "✓" if result["valid"] else "✗"
            logger.info(f"{status} {task}: {'Valid' if result['valid'] else 'Invalid'}")
            if not result["valid"] and result["issues"]:
                logger.info(f"    Issues: {result['issues'][0]}")
        
        # Save detailed report
        report_path = Path(__file__).parent / "promptgen_validation_report.json"
        with open(report_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"\nDetailed report saved to: {report_path}")
        
        return results
    
    def validate_against_reference(self, test_image: Image.Image, reference_outputs: Dict[str, str]):
        """Validate outputs against reference implementation"""
        logger.info("\n" + "=" * 60)
        logger.info("REFERENCE VALIDATION")
        logger.info("=" * 60)
        
        if not self.analyzer:
            self.setup_analyzer()
        
        for task, reference in reference_outputs.items():
            logger.info(f"\nValidating {task} against reference")
            
            # Get our output
            our_output = self.analyzer._run_florence_task(test_image, task)
            
            if isinstance(our_output, dict):
                our_text = list(our_output.values())[0] if our_output else ''
            else:
                our_text = str(our_output) if our_output else ''
            
            # Compare
            logger.info(f"Reference length: {len(reference)}")
            logger.info(f"Our length: {len(our_text)}")
            
            # Check if outputs are similar in structure
            if task == "<GENERATE_TAGS>":
                ref_tags = set(t.strip() for t in reference.split(','))
                our_tags = set(t.strip() for t in our_text.split(','))
                overlap = ref_tags.intersection(our_tags)
                logger.info(f"Tag overlap: {len(overlap)}/{len(ref_tags)} tags")
            else:
                # For captions, check key concepts
                ref_words = set(reference.lower().split())
                our_words = set(our_text.lower().split())
                overlap = ref_words.intersection(our_words)
                logger.info(f"Word overlap: {len(overlap)}/{len(ref_words)} words")


def create_test_image() -> Image.Image:
    """Create a complex test image"""
    img = Image.new('RGB', (768, 768), color='white')
    from PIL import ImageDraw
    draw = ImageDraw.Draw(img)
    
    # Draw a scene with multiple elements
    # Sky
    draw.rectangle([0, 0, 768, 400], fill=(135, 206, 235))
    
    # Sun
    draw.ellipse([600, 50, 700, 150], fill=(255, 255, 0))
    
    # Mountains
    draw.polygon([(0, 400), (200, 200), (400, 400)], fill=(105, 105, 105))
    draw.polygon([(300, 400), (500, 250), (700, 400)], fill=(128, 128, 128))
    
    # Ground
    draw.rectangle([0, 400, 768, 768], fill=(34, 139, 34))
    
    # Tree
    draw.rectangle([100, 350, 150, 450], fill=(101, 67, 33))
    draw.ellipse([50, 300, 200, 400], fill=(0, 128, 0))
    
    # House
    draw.rectangle([400, 350, 550, 450], fill=(178, 34, 34))
    draw.polygon([(380, 350), (475, 280), (570, 350)], fill=(139, 69, 19))
    
    # Add text element
    draw.text((50, 50), "Test Scene", fill=(255, 255, 255))
    
    return img


def main():
    """Run validation tests"""
    validator = PromptGenValidator()
    
    # Create test image
    test_image = create_test_image()
    
    # Run validation suite
    results = validator.run_validation_suite(test_image)
    
    # Optional: Validate against reference outputs
    # Uncomment and provide reference outputs from another implementation
    # reference_outputs = {
    #     "<GENERATE_TAGS>": "scenery, landscape, mountain, house, tree, sun, sky, outdoors, nature",
    #     "<MORE_DETAILED_CAPTION>": "A serene landscape scene featuring..."
    # }
    # validator.validate_against_reference(test_image, reference_outputs)
    
    # Clean up
    if validator.analyzer:
        validator.analyzer.unload_model()


if __name__ == "__main__":
    main()