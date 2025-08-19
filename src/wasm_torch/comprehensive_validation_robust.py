"""
Comprehensive Validation - Generation 2: Make It Robust
Advanced input/output validation, schema checking, and data integrity systems.
"""

import asyncio
import logging
import time
import hashlib
import json
import re
from typing import Dict, Any, List, Optional, Union, Callable, Type, Tuple
from dataclasses import dataclass, field
from enum import Enum, auto
from functools import wraps
import threading
from pathlib import Path

logger = logging.getLogger(__name__)


class ValidationLevel(Enum):
    """Validation strictness levels."""
    
    BASIC = auto()      # Basic type and null checks
    STANDARD = auto()   # Standard validation with common rules
    STRICT = auto()     # Strict validation with comprehensive checks
    PARANOID = auto()   # Paranoid validation with security checks


class ValidationCategory(Enum):
    """Categories of validation checks."""
    
    TYPE = auto()       # Type validation
    RANGE = auto()      # Range and boundary validation  
    FORMAT = auto()     # Format and pattern validation
    SECURITY = auto()   # Security-related validation
    BUSINESS = auto()   # Business logic validation
    INTEGRITY = auto()  # Data integrity validation


@dataclass
class ValidationRule:
    """Individual validation rule definition."""
    
    name: str
    category: ValidationCategory
    validator: Callable[[Any], bool]
    error_message: str
    level: ValidationLevel = ValidationLevel.STANDARD
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass 
class ValidationResult:
    """Result of validation operation."""
    
    valid: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    validation_time: float = 0.0
    
    def add_error(self, message: str) -> None:
        """Add validation error."""
        self.valid = False
        self.errors.append(message)
    
    def add_warning(self, message: str) -> None:
        """Add validation warning."""
        self.warnings.append(message)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'valid': self.valid,
            'errors': self.errors,
            'warnings': self.warnings,
            'metadata': self.metadata,
            'validation_time': self.validation_time
        }


class SchemaValidator:
    """
    JSON Schema-like validator with custom rules.
    """
    
    def __init__(self):
        self._type_validators = {
            'string': self._validate_string,
            'integer': self._validate_integer,
            'number': self._validate_number,
            'boolean': self._validate_boolean,
            'array': self._validate_array,
            'object': self._validate_object,
            'null': self._validate_null
        }
    
    def validate(self, data: Any, schema: Dict[str, Any]) -> ValidationResult:
        """Validate data against schema."""
        result = ValidationResult(valid=True)
        start_time = time.time()
        
        try:
            self._validate_recursive(data, schema, result, "root")
            result.validation_time = time.time() - start_time
            return result
            
        except Exception as e:
            result.add_error(f"Schema validation failed: {e}")
            result.validation_time = time.time() - start_time
            return result
    
    def _validate_recursive(
        self, 
        data: Any, 
        schema: Dict[str, Any], 
        result: ValidationResult, 
        path: str
    ) -> None:
        """Recursively validate data against schema."""
        
        # Check required fields
        if schema.get('required', False) and data is None:
            result.add_error(f"{path}: Required field is missing")
            return
        
        # Check type
        expected_type = schema.get('type')
        if expected_type and not self._check_type(data, expected_type):
            result.add_error(f"{path}: Expected {expected_type}, got {type(data).__name__}")
            return
        
        # Type-specific validation
        if expected_type in self._type_validators:
            self._type_validators[expected_type](data, schema, result, path)
        
        # Custom validation rules
        rules = schema.get('rules', [])
        for rule in rules:
            if not self._apply_rule(data, rule, result, path):
                break  # Stop on first failure if specified
    
    def _check_type(self, data: Any, expected_type: str) -> bool:
        """Check if data matches expected type."""
        if data is None:
            return expected_type == 'null'
        
        type_mapping = {
            'string': str,
            'integer': int,
            'number': (int, float),
            'boolean': bool,
            'array': (list, tuple),
            'object': dict,
            'null': type(None)
        }
        
        expected = type_mapping.get(expected_type)
        if expected is None:
            return False
        
        return isinstance(data, expected)
    
    def _validate_string(
        self, 
        data: str, 
        schema: Dict[str, Any], 
        result: ValidationResult, 
        path: str
    ) -> None:
        """Validate string data."""
        if not isinstance(data, str):
            return
        
        # Length validation
        min_length = schema.get('minLength')
        if min_length is not None and len(data) < min_length:
            result.add_error(f"{path}: String too short (min: {min_length})")
        
        max_length = schema.get('maxLength')
        if max_length is not None and len(data) > max_length:
            result.add_error(f"{path}: String too long (max: {max_length})")
        
        # Pattern validation
        pattern = schema.get('pattern')
        if pattern and not re.match(pattern, data):
            result.add_error(f"{path}: String doesn't match pattern")
        
        # Enum validation
        enum_values = schema.get('enum')
        if enum_values and data not in enum_values:
            result.add_error(f"{path}: Value not in allowed enum values")
    
    def _validate_integer(
        self, 
        data: int, 
        schema: Dict[str, Any], 
        result: ValidationResult, 
        path: str
    ) -> None:
        """Validate integer data."""
        if not isinstance(data, int):
            return
        
        # Range validation
        minimum = schema.get('minimum')
        if minimum is not None and data < minimum:
            result.add_error(f"{path}: Value too small (min: {minimum})")
        
        maximum = schema.get('maximum')
        if maximum is not None and data > maximum:
            result.add_error(f"{path}: Value too large (max: {maximum})")
    
    def _validate_number(
        self, 
        data: Union[int, float], 
        schema: Dict[str, Any], 
        result: ValidationResult, 
        path: str
    ) -> None:
        """Validate numeric data."""
        if not isinstance(data, (int, float)):
            return
        
        # Range validation
        minimum = schema.get('minimum')
        if minimum is not None and data < minimum:
            result.add_error(f"{path}: Value too small (min: {minimum})")
        
        maximum = schema.get('maximum')
        if maximum is not None and data > maximum:
            result.add_error(f"{path}: Value too large (max: {maximum})")
        
        # Multiple validation
        multiple_of = schema.get('multipleOf')
        if multiple_of and data % multiple_of != 0:
            result.add_error(f"{path}: Value is not a multiple of {multiple_of}")
    
    def _validate_boolean(
        self, 
        data: bool, 
        schema: Dict[str, Any], 
        result: ValidationResult, 
        path: str
    ) -> None:
        """Validate boolean data."""
        # Boolean validation is just type checking
        pass
    
    def _validate_array(
        self, 
        data: List[Any], 
        schema: Dict[str, Any], 
        result: ValidationResult, 
        path: str
    ) -> None:
        """Validate array data."""
        if not isinstance(data, (list, tuple)):
            return
        
        # Length validation
        min_items = schema.get('minItems')
        if min_items is not None and len(data) < min_items:
            result.add_error(f"{path}: Array too short (min items: {min_items})")
        
        max_items = schema.get('maxItems')
        if max_items is not None and len(data) > max_items:
            result.add_error(f"{path}: Array too long (max items: {max_items})")
        
        # Items validation
        items_schema = schema.get('items')
        if items_schema:
            for i, item in enumerate(data):
                self._validate_recursive(item, items_schema, result, f"{path}[{i}]")
        
        # Unique items
        if schema.get('uniqueItems', False):
            if len(data) != len(set(json.dumps(item, sort_keys=True) for item in data)):
                result.add_error(f"{path}: Array items must be unique")
    
    def _validate_object(
        self, 
        data: Dict[str, Any], 
        schema: Dict[str, Any], 
        result: ValidationResult, 
        path: str
    ) -> None:
        """Validate object data."""
        if not isinstance(data, dict):
            return
        
        # Properties validation
        properties = schema.get('properties', {})
        for prop_name, prop_schema in properties.items():
            prop_path = f"{path}.{prop_name}"
            if prop_name in data:
                self._validate_recursive(data[prop_name], prop_schema, result, prop_path)
            elif prop_schema.get('required', False):
                result.add_error(f"{prop_path}: Required property missing")
        
        # Additional properties
        if not schema.get('additionalProperties', True):
            extra_props = set(data.keys()) - set(properties.keys())
            if extra_props:
                result.add_error(f"{path}: Additional properties not allowed: {extra_props}")
    
    def _validate_null(
        self, 
        data: None, 
        schema: Dict[str, Any], 
        result: ValidationResult, 
        path: str
    ) -> None:
        """Validate null data."""
        # Null validation is just type checking
        pass
    
    def _apply_rule(
        self, 
        data: Any, 
        rule: Dict[str, Any], 
        result: ValidationResult, 
        path: str
    ) -> bool:
        """Apply custom validation rule."""
        try:
            rule_func = rule.get('validator')
            if callable(rule_func):
                if not rule_func(data):
                    message = rule.get('message', f"Custom validation failed for {path}")
                    result.add_error(message)
                    return False
            return True
            
        except Exception as e:
            result.add_error(f"{path}: Rule validation error: {e}")
            return False


class ComprehensiveValidatorRobust:
    """
    Comprehensive validation system with multiple validation strategies.
    Generation 2: Make It Robust - Enhanced reliability and error handling.
    """
    
    def __init__(self, level: ValidationLevel = ValidationLevel.STANDARD):
        self.level = level
        self.schema_validator = SchemaValidator()
        self._validation_rules: Dict[str, List[ValidationRule]] = {}
        self._security_patterns = self._load_security_patterns()
        self._statistics = {
            'total_validations': 0,
            'failed_validations': 0,
            'validation_time_total': 0.0,
            'rules_applied': 0
        }
        self._lock = threading.RLock()
    
    def validate_model_input(
        self, 
        input_data: Any, 
        model_id: str,
        expected_shape: Optional[Tuple[int, ...]] = None
    ) -> ValidationResult:
        """Validate model input data with comprehensive checks."""
        result = ValidationResult(valid=True)
        start_time = time.time()
        
        try:
            # Basic null check
            if input_data is None:
                result.add_error("Input data cannot be None")
                return result
            
            # Type validation
            if not isinstance(input_data, (list, tuple, dict, str, int, float)):
                result.add_warning(f"Unusual input type: {type(input_data).__name__}")
            
            # Size validation for sequences
            if hasattr(input_data, '__len__'):
                length = len(input_data)
                if length == 0:
                    result.add_warning("Input data is empty")
                elif length > 1000000:  # 1M elements
                    result.add_error("Input data is too large (>1M elements)")
            
            # Value validation for numeric types
            if isinstance(input_data, (int, float)):
                if not (-1e10 <= input_data <= 1e10):
                    result.add_warning("Numeric input has extreme value")
            
            # String validation
            if isinstance(input_data, str):
                if len(input_data) > 100000:  # 100KB
                    result.add_warning("String input is very long")
                
                # Check for suspicious patterns
                if self.level in [ValidationLevel.STRICT, ValidationLevel.PARANOID]:
                    self._validate_string_security(input_data, result)
            
            # Dictionary validation
            if isinstance(input_data, dict):
                if len(input_data) > 10000:
                    result.add_warning("Dictionary has too many keys")
                
                # Validate nested structure
                if self._calculate_dict_depth(input_data) > 20:
                    result.add_error("Dictionary nesting too deep (>20 levels)")
            
            # List/array validation  
            if isinstance(input_data, (list, tuple)):
                if len(input_data) > 0:
                    # Check type consistency
                    first_type = type(input_data[0])
                    mixed_types = not all(isinstance(item, first_type) for item in input_data[:100])
                    if mixed_types:
                        result.add_warning("Array contains mixed types")
            
            result.validation_time = time.time() - start_time
            return result
            
        except Exception as e:
            result.add_error(f"Validation error: {e}")
            result.validation_time = time.time() - start_time
            return result
    
    def validate_model_output(
        self, 
        output_data: Any, 
        model_id: str,
        expected_format: Optional[str] = None
    ) -> ValidationResult:
        """Validate model output data."""
        result = ValidationResult(valid=True)
        start_time = time.time()
        
        try:
            # Basic null check
            if output_data is None:
                result.add_error("Output data cannot be None")
                return result
            
            # Format-specific validation
            if expected_format == 'classification':
                if isinstance(output_data, dict):
                    if 'prediction' not in output_data:
                        result.add_error("Classification output missing 'prediction' field")
                    
                    confidence = output_data.get('confidence')
                    if confidence is not None:
                        if not isinstance(confidence, (int, float)):
                            result.add_error("Confidence must be numeric")
                        elif not 0.0 <= confidence <= 1.0:
                            result.add_error("Confidence must be between 0 and 1")
                else:
                    result.add_warning("Classification output should be a dictionary")
            
            elif expected_format == 'regression':
                if not isinstance(output_data, (int, float, list, tuple)):
                    result.add_error("Regression output should be numeric or array")
                
                if isinstance(output_data, (list, tuple)):
                    if len(output_data) == 0:
                        result.add_error("Regression output array is empty")
                    elif not all(isinstance(x, (int, float)) for x in output_data):
                        result.add_error("Regression output array contains non-numeric values")
            
            # General output validation
            if isinstance(output_data, dict):
                # Check for reasonable dictionary size
                if len(output_data) > 1000:
                    result.add_warning("Output dictionary is very large")
                
                # Validate common output fields
                for field in ['error', 'success', 'status']:
                    if field in output_data:
                        field_value = output_data[field]
                        if field_value is None:
                            result.add_warning(f"Output field '{field}' is None")
            
            result.validation_time = time.time() - start_time
            return result
            
        except Exception as e:
            result.add_error(f"Output validation error: {e}")
            result.validation_time = time.time() - start_time
            return result
    
    def validate_configuration(
        self, 
        config: Dict[str, Any],
        required_fields: Optional[List[str]] = None
    ) -> ValidationResult:
        """Validate system configuration."""
        result = ValidationResult(valid=True)
        start_time = time.time()
        
        try:
            if not isinstance(config, dict):
                result.add_error("Configuration must be a dictionary")
                return result
            
            # Check required fields
            if required_fields:
                for field in required_fields:
                    if field not in config:
                        result.add_error(f"Required configuration field missing: {field}")
                    elif config[field] is None:
                        result.add_error(f"Configuration field cannot be None: {field}")
            
            # Validate common configuration patterns
            if 'timeout' in config:
                timeout = config['timeout']
                if not isinstance(timeout, (int, float)) or timeout <= 0:
                    result.add_error("Timeout must be a positive number")
            
            if 'max_workers' in config:
                max_workers = config['max_workers']
                if not isinstance(max_workers, int) or max_workers < 1:
                    result.add_error("max_workers must be a positive integer")
                elif max_workers > 100:
                    result.add_warning("max_workers is very high (>100)")
            
            if 'batch_size' in config:
                batch_size = config['batch_size']
                if not isinstance(batch_size, int) or batch_size < 1:
                    result.add_error("batch_size must be a positive integer")
                elif batch_size > 10000:
                    result.add_warning("batch_size is very large (>10000)")
            
            # Security validation for paths
            for key, value in config.items():
                if 'path' in key.lower() and isinstance(value, str):
                    if '../' in value or '..\\' in value:
                        result.add_error(f"Potential path traversal in {key}: {value}")
                    if not Path(value).is_absolute() and self.level == ValidationLevel.PARANOID:
                        result.add_warning(f"Relative path in {key}: {value}")
            
            result.validation_time = time.time() - start_time
            return result
            
        except Exception as e:
            result.add_error(f"Configuration validation error: {e}")
            result.validation_time = time.time() - start_time
            return result
    
    def _validate_string_security(self, data: str, result: ValidationResult) -> None:
        """Validate string for security issues."""
        try:
            # Check for potential injection patterns
            for pattern_name, pattern in self._security_patterns.items():
                if re.search(pattern, data, re.IGNORECASE):
                    if self.level == ValidationLevel.PARANOID:
                        result.add_error(f"Security risk detected: {pattern_name}")
                    else:
                        result.add_warning(f"Potential security issue: {pattern_name}")
            
            # Check for suspicious characters
            suspicious_chars = ['<', '>', '&', '"', "'", '\\', '\x00']
            found_suspicious = [char for char in suspicious_chars if char in data]
            if found_suspicious:
                if self.level == ValidationLevel.PARANOID:
                    result.add_error(f"Dangerous characters found: {found_suspicious}")
                else:
                    result.add_warning(f"Special characters found: {found_suspicious}")
            
            # Check encoding issues
            try:
                data.encode('utf-8')
            except UnicodeEncodeError:
                result.add_error("String contains invalid Unicode characters")
                
        except Exception as e:
            result.add_warning(f"Security validation error: {e}")
    
    def _calculate_dict_depth(self, data: Dict[str, Any], current_depth: int = 0) -> int:
        """Calculate maximum nesting depth of dictionary."""
        if not isinstance(data, dict) or current_depth > 50:  # Prevent infinite recursion
            return current_depth
        
        max_depth = current_depth
        for value in data.values():
            if isinstance(value, dict):
                depth = self._calculate_dict_depth(value, current_depth + 1)
                max_depth = max(max_depth, depth)
        
        return max_depth
    
    def _load_security_patterns(self) -> Dict[str, str]:
        """Load security validation patterns."""
        return {
            'sql_injection': r'(union|select|insert|update|delete|drop|create|alter)\s+',
            'xss_script': r'<script[^>]*>.*?</script>',
            'xss_on': r'on\w+\s*=',
            'path_traversal': r'\.\./|\.\.\\\\',
            'command_injection': r'[;&|`$()]',
            'null_byte': r'\x00',
            'eval_injection': r'eval\s*\(',
            'import_injection': r'import\s+\w+'
        }
    
    def get_validation_statistics(self) -> Dict[str, Any]:
        """Get comprehensive validation statistics."""
        with self._lock:
            stats = self._statistics.copy()
            
            if stats['total_validations'] > 0:
                stats['failure_rate'] = stats['failed_validations'] / stats['total_validations']
                stats['average_validation_time'] = (
                    stats['validation_time_total'] / stats['total_validations']
                )
            else:
                stats['failure_rate'] = 0.0
                stats['average_validation_time'] = 0.0
            
            stats['validation_level'] = self.level.name
            stats['registered_rules'] = len(self._validation_rules)
            
            return stats


# Global validator instance
_global_validator_robust: Optional[ComprehensiveValidatorRobust] = None


def get_global_validator_robust() -> ComprehensiveValidatorRobust:
    """Get the global robust validator instance."""
    global _global_validator_robust
    if _global_validator_robust is None:
        _global_validator_robust = ComprehensiveValidatorRobust()
    return _global_validator_robust


# Enhanced validation decorators
def validate_robust_input(
    model_id: Optional[str] = None,
    expected_shape: Optional[Tuple[int, ...]] = None,
    level: ValidationLevel = ValidationLevel.STANDARD
):
    """Decorator for robust input validation."""
    def decorator(func):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            validator = get_global_validator_robust()
            validator.level = level
            
            # Validate first argument as model input
            if args:
                result = validator.validate_model_input(
                    args[0], 
                    model_id or "unknown",
                    expected_shape
                )
                if not result.valid:
                    raise ValueError(f"Input validation failed: {'; '.join(result.errors)}")
                if result.warnings:
                    logger.warning(f"Input validation warnings: {'; '.join(result.warnings)}")
            
            return await func(*args, **kwargs)
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            return asyncio.run(async_wrapper(*args, **kwargs))
        
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator


# Example usage and testing
async def demo_robust_validation():
    """Demonstration of robust validation system."""
    validator = ComprehensiveValidatorRobust(ValidationLevel.STRICT)
    
    print("Testing robust validation system...")
    print("=" * 50)
    
    # Test model input validation
    test_inputs = [
        ([1, 2, 3, 4, 5], "numeric_array"),
        ("Hello world", "text_input"),
        ({'data': [1, 2, 3], 'meta': {'version': 1}}, "structured_input"),
        (None, "null_input"),
        ("" * 100001, "very_long_string"),  # Should trigger warning
    ]
    
    print("Input validation tests:")
    for i, (test_input, description) in enumerate(test_inputs):
        print(f"\nTest {i+1} ({description}):")
        result = validator.validate_model_input(test_input, f"test_model_{i}")
        print(f"  Valid: {result.valid}")
        if result.errors:
            print(f"  Errors: {result.errors}")
        if result.warnings:
            print(f"  Warnings: {result.warnings}")
        print(f"  Time: {result.validation_time:.4f}s")
    
    # Test output validation
    test_outputs = [
        ({'prediction': 'positive', 'confidence': 0.85}, 'classification'),
        ([0.1, 0.2, 0.3], 'regression'),
        ({'prediction': 'negative'}, 'classification'),  # Missing confidence
        (42, 'regression'),
        (None, 'classification'),  # Should fail
    ]
    
    print("\n" + "="*50)
    print("Output validation tests:")
    for i, (test_output, expected_format) in enumerate(test_outputs):
        print(f"\nTest {i+1} ({expected_format}):")
        result = validator.validate_model_output(test_output, f"test_model_{i}", expected_format)
        print(f"  Valid: {result.valid}")
        if result.errors:
            print(f"  Errors: {result.errors}")
        if result.warnings:
            print(f"  Warnings: {result.warnings}")
    
    # Test configuration validation
    test_configs = [
        ({'timeout': 30, 'max_workers': 4, 'batch_size': 32}, ['timeout']),
        ({'timeout': -5, 'max_workers': 0}, ['timeout', 'max_workers']),  # Invalid values
        ({'model_path': '../../../etc/passwd'}, []),  # Security issue
        ({}, ['required_field']),  # Missing required field
    ]
    
    print("\n" + "="*50)
    print("Configuration validation tests:")
    for i, (config, required_fields) in enumerate(test_configs):
        print(f"\nTest {i+1}:")
        result = validator.validate_configuration(config, required_fields)
        print(f"  Valid: {result.valid}")
        if result.errors:
            print(f"  Errors: {result.errors}")
        if result.warnings:
            print(f"  Warnings: {result.warnings}")
    
    # Show statistics
    print("\n" + "="*50)
    print("Validation statistics:")
    stats = validator.get_validation_statistics()
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    asyncio.run(demo_robust_validation())