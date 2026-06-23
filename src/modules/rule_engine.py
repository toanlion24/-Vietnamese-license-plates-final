"""
Module 7: Rule Engine & Regex
Validates and normalizes Vietnamese license plate text
"""

import re
from typing import List, Dict, Optional, Tuple, Callable
from dataclasses import dataclass, field
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class PlateType(Enum):
    """Vietnamese license plate types"""
    PRIVATE_CAR = "private_car"
    COMMERCIAL = "commercial"
    MOTORCYCLE = "motorcycle"
    POLICE = "police"
    ARMY = "army"
    FOREIGN = "foreign"
    TEMPORARY = "temporary"
    UNKNOWN = "unknown"


@dataclass
class ValidationResult:
    """Result of plate validation"""
    is_valid: bool
    plate_type: PlateType
    normalized_text: str
    raw_text: str
    confidence: float = 1.0
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    format_pattern: str = ""


@dataclass
class CorrectionResult:
    """Result of OCR correction"""
    original: str
    corrected: str
    changes: List[Tuple[int, str, str]] = field(default_factory=list)  # (pos, old, new)
    confidence: float = 1.0


class VietnamesePlatePatterns:
    """
    Regex patterns for Vietnamese license plates.
    
    Vietnamese plate formats:
    - Private car: 30A-1234.56 (2 digits + letter + hyphen + 4 digits + dot + 2 digits)
    - Motorcycle: 43-12345 (2 digits + hyphen + 5 digits)
    - Police: 60-1234-56 (2 digits + hyphen + 4 digits + hyphen + 2 digits)
    - Army: 123456-78 (6 digits + hyphen + 2 digits)
    """
    
    # Private car: XX-YYYY.NN
    PRIVATE_CAR = re.compile(r'^(\d{2})([A-Z])-(\d{4})\.(\d{2})$')
    
    # Motorcycle: 
    # - Old format: YY-NNNNN (2 digits - 5 digits) e.g. 43-12345
    # - New format: YY-XXX-YY (2 digits - 2-3 letters - 2 digits) e.g. 80-NG-63, 15-PHP-01
    MOTORCYCLE_OLD = re.compile(r'^(\d{2})-(\d{5})(?:\.(\d{2}))?$')
    MOTORCYCLE_NEW = re.compile(r'^(\d{2})-([A-Z]{2,3})-(\d{2})$')
    
    # Combined motorcycle pattern (matches both formats)
    MOTORCYCLE = re.compile(r'^(\d{2})-(\d{5}|[A-Z]{2,3})-?(\d{2})?$')
    
    # Police: XX-YYYY-NN
    POLICE = re.compile(r'^(\d{2})-(\d{4})-(\d{2})$')
    
    # Army: YYYYYY-NN (6 digits)
    ARMY = re.compile(r'^(\d{6})-(\d{2})$')
    
    # Commercial: Similar to private but different prefix ranges
    COMMERCIAL = re.compile(r'^(\d{2})([A-Z])-(\d{4})\.(\d{2})$')
    
    # All patterns
    ALL_PATTERNS = {
        PlateType.PRIVATE_CAR: PRIVATE_CAR,
        PlateType.COMMERCIAL: COMMERCIAL,
        PlateType.MOTORCYCLE: MOTORCYCLE,
        PlateType.POLICE: POLICE,
        PlateType.ARMY: ARMY,
    }
    
    # Province codes (first 2 digits)
    PROVINCE_CODES = {
        '11': 'Hà Nội', '12': 'Hà Nội', '13': 'Hà Nội',
        '14': 'Hà Nội', '15': 'Hà Nội', '16': 'Hà Nội',
        '17': 'Hà Nội', '18': 'Hà Nội',
        '29': 'Hà Nội', '30': 'TP.HCM', '31': 'TP.HCM',
        '32': 'TP.HCM', '33': 'TP.HCM', '40': 'Hải Phòng',
        '43': 'Đà Nẵng', '47': 'Đắk Lắk', '48': 'Khánh Hòa',
        '50': 'Bình Dương', '51': 'Bình Dương', '52': 'Bình Dương',
        '53': 'Bình Dương', '54': 'Bình Dương', '55': 'Bình Dương',
        '56': 'Bình Dương', '57': 'Bình Dương', '58': 'Bình Dương',
        '59': 'Bình Dương', '60': 'Đồng Nai', '61': 'Đồng Nai',
        '62': 'Đồng Nai', '63': 'Đồng Nai', '64': 'Đồng Nai',
        '65': 'Đồng Nai', '66': 'Đồng Nai', '67': 'Đồng Nai',
        '68': 'Đồng Nai', '69': 'Đồng Nai',
    }


class CharacterCorrector:
    """
    OCR error correction for license plates.
    
    Known OCR confusions:
    - 0 vs O vs D
    - 1 vs I vs l vs |
    - 5 vs S
    - 8 vs B
    - 4 vs A (in some fonts)
    """
    
    # Common OCR confusions
    CONFUSION_PAIRS = {
        'O': ['0', 'D'],
        '0': ['O', 'D'],
        'D': ['0', 'O'],
        'I': ['1', 'l', '|'],
        '1': ['I', 'l', '|'],
        'l': ['1', 'I', '|'],
        'S': ['5'],
        '5': ['S'],
        'B': ['8'],
        '8': ['B'],
        'A': ['4'],
        '4': ['A'],
    }
    
    # Position-specific corrections
    # First character after province code is usually letter for car
    LETTER_POSITIONS = {
        PlateType.PRIVATE_CAR: [2],  # Position 2 is letter
        PlateType.COMMERCIAL: [2],
    }
    
    # Digit positions
    DIGIT_POSITIONS = {
        PlateType.MOTORCYCLE: [0, 1, 3, 4, 5, 6, 7],  # All except dash
        PlateType.ARMY: [0, 1, 2, 3, 4, 5, 7, 8],  # All except dash
    }
    
    def correct(self, text: str, plate_type: PlateType = None) -> CorrectionResult:
        """
        Apply character corrections to OCR text.
        
        Args:
            text: Raw OCR text
            plate_type: Known plate type
            
        Returns:
            CorrectionResult with corrections
        """
        corrected = list(text.upper())
        changes = []
        
        for i, char in enumerate(text.upper()):
            if char in self.CONFUSION_PAIRS:
                alternatives = self.CONFUSION_PAIRS[char]
                
                # Check if this position should be a digit
                if plate_type and plate_type in self.DIGIT_POSITIONS:
                    if i in self.DIGIT_POSITIONS[plate_type]:
                        if '0' in alternatives:
                            corrected[i] = '0'
                            if corrected[i] != char:
                                changes.append((i, char, '0'))
                        continue
                
                # Check if this position should be a letter
                if plate_type and plate_type in self.LETTER_POSITIONS:
                    if i in self.LETTER_POSITIONS[plate_type]:
                        if char not in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ':
                            # Try to find best letter alternative
                            pass  # Keep as is
                        continue
        
        return CorrectionResult(
            original=text,
            corrected="".join(corrected),
            changes=changes,
            confidence=1.0 - len(changes) * 0.1
        )


class PlateValidator:
    """
    Validates and normalizes Vietnamese license plate text.
    """
    
    def __init__(self):
        self.patterns = VietnamesePlatePatterns()
        self.corrector = CharacterCorrector()
        self.province_codes = self.patterns.PROVINCE_CODES
    
    def validate(self, text: str) -> ValidationResult:
        """
        Validate and normalize plate text.
        
        Args:
            text: Raw text from OCR
            
        Returns:
            ValidationResult
        """
        text = text.strip()
        raw_text = text
        
        if not text:
            return ValidationResult(
                is_valid=False,
                plate_type=PlateType.UNKNOWN,
                normalized_text="",
                raw_text="",
                errors=["Empty text"]
            )
        
        # Preprocessing
        text = self._preprocess(text)
        
        # Try each pattern
        for plate_type, pattern in self.patterns.ALL_PATTERNS.items():
            match = pattern.match(text)
            if match:
                normalized = self._normalize_match(text, plate_type, match)
                return ValidationResult(
                    is_valid=True,
                    plate_type=plate_type,
                    normalized_text=normalized,
                    raw_text=raw_text,
                    format_pattern=pattern.pattern,
                    confidence=1.0
                )
        
        # Try correction
        corrected = self.corrector.correct(text)
        if corrected.changes:
            # Try patterns again with corrected text
            for plate_type, pattern in self.patterns.ALL_PATTERNS.items():
                match = pattern.match(corrected.corrected)
                if match:
                    normalized = self._normalize_match(corrected.corrected, plate_type, match)
                    return ValidationResult(
                        is_valid=True,
                        plate_type=plate_type,
                        normalized_text=normalized,
                        raw_text=raw_text,
                        format_pattern=pattern.pattern,
                        confidence=corrected.confidence,
                        warnings=[f"Applied {len(corrected.changes)} corrections"]
                    )
        
        return ValidationResult(
            is_valid=False,
            plate_type=PlateType.UNKNOWN,
            normalized_text=text,
            raw_text=raw_text,
            errors=["No matching pattern found"]
        )
    
    def _preprocess(self, text: str) -> str:
        """Preprocess text before validation"""
        # Remove spaces
        text = text.replace(" ", "")
        
        # Replace similar characters
        replacements = {
            '[': '', ']': '', '{': '', '}': '',
            '|': '1', '!': '1', '|': '1',
            'O': '0',  # O -> 0 in plates
            'I': '1',  # I -> 1 in plates
            'l': '1',  # l -> 1 in plates
            'S': '5',  # S -> 5 (common OCR error)
            'B': '8',  # B -> 8 (common OCR error)
        }
        
        for old, new in replacements.items():
            text = text.replace(old, new)
        
        return text.upper()
    
    def _normalize_match(self, text: str, plate_type: PlateType, match) -> str:
        """Normalize matched plate text"""
        if plate_type == PlateType.PRIVATE_CAR:
            return f"{match.group(1)}{match.group(2)}-{match.group(3)}.{match.group(4)}"
        elif plate_type == PlateType.MOTORCYCLE:
            # Check if it's the new format (YY-XXX-YY) or old format (YY-NNNNN)
            if match.group(3):
                # New format: YY-XXX-YY
                return f"{match.group(1)}-{match.group(2)}-{match.group(3)}"
            else:
                # Old format: YY-NNNNN
                return f"{match.group(1)}-{match.group(2)}"
        elif plate_type == PlateType.POLICE:
            return f"{match.group(1)}-{match.group(2)}-{match.group(3)}"
        elif plate_type == PlateType.ARMY:
            return f"{match.group(1)}-{match.group(2)}"
        return text
    
    def get_province(self, plate: str) -> Optional[str]:
        """Get province/city from plate number"""
        if len(plate) < 2:
            return None
        
        code = plate[:2]
        return self.province_codes.get(code)
    
    def validate_batch(self, texts: List[str]) -> List[ValidationResult]:
        """Validate multiple plates"""
        return [self.validate(text) for text in texts]


class PlateRuleEngine:
    """
    Rule engine for Vietnamese license plate processing.
    """
    
    def __init__(self):
        self.validator = PlateValidator()
        self._rules: List[Callable] = []
        self._init_default_rules()
    
    def _init_default_rules(self):
        """Initialize default rules"""
        # Rule: Check plate length
        def check_length(text: str) -> Tuple[bool, str]:
            if len(text) < 5 or len(text) > 12:
                return False, f"Invalid length: {len(text)}"
            return True, ""
        
        # Rule: Check for valid characters
        def check_chars(text: str) -> Tuple[bool, str]:
            valid_chars = set("0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ-.")
            invalid = set(text.upper()) - valid_chars
            if invalid:
                return False, f"Invalid characters: {invalid}"
            return True, ""
        
        # Rule: Check format
        def check_format(text: str) -> Tuple[bool, str]:
            result = self.validator.validate(text)
            if not result.is_valid:
                return False, f"Invalid format: {result.errors}"
            return True, ""
        
        self.add_rule("length", check_length)
        self.add_rule("characters", check_chars)
        self.add_rule("format", check_format)
    
    def add_rule(self, name: str, rule_func: Callable):
        """Add a processing rule"""
        self._rules.append((name, rule_func))
    
    def process(self, text: str) -> ValidationResult:
        """Process text through all rules"""
        result = self.validator.validate(text)
        
        for rule_name, rule_func in self._rules:
            valid, msg = rule_func(text)
            if not valid:
                result.warnings.append(f"[{rule_name}] {msg}")
        
        return result


def normalize_vietnamese_plate(text: str) -> str:
    """
    Normalize Vietnamese license plate text.
    
    Args:
        text: Raw plate text
        
    Returns:
        Normalized plate string
    """
    validator = PlateValidator()
    result = validator.validate(text)
    return result.normalized_text


def validate_vietnamese_plate(text: str) -> Tuple[bool, PlateType, str]:
    """
    Quick validation of plate text.
    
    Args:
        text: Plate text
        
    Returns:
        Tuple of (is_valid, plate_type, normalized_text)
    """
    validator = PlateValidator()
    result = validator.validate(text)
    return result.is_valid, result.plate_type, result.normalized_text


def get_province_from_plate(plate: str) -> Optional[str]:
    """
    Get province/city from plate number.
    
    Args:
        plate: License plate number
        
    Returns:
        Province name or None
    """
    validator = PlateValidator()
    return validator.get_province(plate)


def visualize_validation(
    raw_text: str,
    result: ValidationResult,
) -> str:
    """
    Generate visual representation of validation result.
    
    Args:
        raw_text: Raw OCR text
        result: Validation result
        
    Returns:
        Markdown representation
    """
    status = "✓" if result.is_valid else "✗"
    plate_type = result.plate_type.value
    normalized = result.normalized_text
    
    output = f"""
## Validation Result {status}

| Field | Value |
|-------|-------|
| Status | {"Valid" if result.is_valid else "Invalid"} |
| Plate Type | {plate_type} |
| Raw Text | `{raw_text}` |
| Normalized | `{normalized}` |
| Confidence | {result.confidence:.2f} |
"""
    
    if result.errors:
        output += f"\n### Errors\n"
        for error in result.errors:
            output += f"- {error}\n"
    
    if result.warnings:
        output += f"\n### Warnings\n"
        for warning in result.warnings:
            output += f"- {warning}\n"
    
    province = get_province_from_plate(normalized)
    if province:
        output += f"\n### Province: {province}\n"
    
    return output
