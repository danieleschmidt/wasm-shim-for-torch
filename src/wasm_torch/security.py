"""Security utilities for WASM Torch."""

import os
import hashlib
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Set
import logging

logger = logging.getLogger(__name__)


class SecurityConfig:
    """Security configuration and policies."""
    
    def __init__(
        self,
        allowed_model_extensions: Optional[Set[str]] = None,
        max_model_size_mb: float = 1000.0,
        allowed_directories: Optional[List[str]] = None,
        enable_model_verification: bool = True
    ):
        self.allowed_model_extensions = allowed_model_extensions or {'.wasm', '.pth', '.pt', '.onnx'}
        self.max_model_size_mb = max_model_size_mb
        self.allowed_directories = allowed_directories or []
        self.enable_model_verification = enable_model_verification


def validate_model_path(
    model_path: Path,
    config: Optional[SecurityConfig] = None
) -> Path:
    """Validate model file path for security.
    
    Args:
        model_path: Path to model file
        config: Security configuration
        
    Returns:
        Validated path
        
    Raises:
        ValueError: If path is invalid or unsafe
        FileNotFoundError: If file doesn't exist
    """
    if config is None:
        config = SecurityConfig()
    
    # Resolve path to prevent directory traversal
    try:
        resolved_path = model_path.resolve()
    except (OSError, ValueError) as e:
        raise ValueError(f"Invalid path: {model_path}") from e
    
    # Check if file exists
    if not resolved_path.exists():
        raise FileNotFoundError(f"Model file not found: {resolved_path}")
    
    # Check file extension
    if resolved_path.suffix.lower() not in config.allowed_model_extensions:
        raise ValueError(
            f"File extension {resolved_path.suffix} not allowed. "
            f"Allowed extensions: {config.allowed_model_extensions}"
        )
    
    # Check file size
    file_size_mb = resolved_path.stat().st_size / (1024 * 1024)
    if file_size_mb > config.max_model_size_mb:
        raise ValueError(
            f"Model file too large ({file_size_mb:.1f}MB). "
            f"Maximum allowed: {config.max_model_size_mb}MB"
        )
    
    # Check directory restrictions
    if config.allowed_directories:
        allowed = False
        for allowed_dir in config.allowed_directories:
            try:
                allowed_path = Path(allowed_dir).resolve()
                if resolved_path.is_relative_to(allowed_path):
                    allowed = True
                    break
            except (ValueError, OSError):
                continue
        
        if not allowed:
            raise ValueError(
                f"Model file not in allowed directory. "
                f"Allowed directories: {config.allowed_directories}"
            )
    
    # Check for suspicious paths
    suspicious_patterns = [
        '/etc/', '/usr/bin/', '/usr/sbin/', '/bin/', '/sbin/',
        '/proc/', '/sys/', '/dev/', '/var/run/'
    ]
    
    path_str = str(resolved_path)
    for pattern in suspicious_patterns:
        if pattern in path_str:
            raise ValueError(f"Access to system directory not allowed: {resolved_path}")
    
    return resolved_path


def compute_file_hash(file_path: Path, algorithm: str = 'sha256') -> str:
    """Compute hash of file for verification.
    
    Args:
        file_path: Path to file
        algorithm: Hash algorithm to use
        
    Returns:
        Hex digest of file hash
    """
    hash_obj = hashlib.new(algorithm)
    
    try:
        with open(file_path, 'rb') as f:
            # Read in chunks to handle large files
            for chunk in iter(lambda: f.read(8192), b''):
                hash_obj.update(chunk)
    except (OSError, IOError) as e:
        raise RuntimeError(f"Failed to read file for hashing: {e}") from e
    
    return hash_obj.hexdigest()


def verify_model_integrity(
    model_path: Path,
    expected_hash: Optional[str] = None,
    hash_algorithm: str = 'sha256'
) -> bool:
    """Verify model file integrity.
    
    Args:
        model_path: Path to model file
        expected_hash: Expected hash value (if known)
        hash_algorithm: Hash algorithm to use
        
    Returns:
        True if verification passes
        
    Raises:
        RuntimeError: If verification fails
    """
    try:
        file_hash = compute_file_hash(model_path, hash_algorithm)
        logger.debug(f"Model hash ({hash_algorithm}): {file_hash}")
        
        if expected_hash and file_hash != expected_hash:
            raise RuntimeError(
                f"Model integrity verification failed. "
                f"Expected: {expected_hash}, Got: {file_hash}"
            )
        
        return True
        
    except Exception as e:
        logger.error(f"Model integrity verification failed: {e}")
        raise


def create_secure_temp_dir(prefix: str = "wasm_torch_") -> Path:
    """Create a secure temporary directory.
    
    Args:
        prefix: Directory name prefix
        
    Returns:
        Path to secure temporary directory
    """
    try:
        temp_dir = tempfile.mkdtemp(prefix=prefix)
        temp_path = Path(temp_dir)
        
        # Set restrictive permissions (owner read/write/execute only)
        temp_path.chmod(0o700)
        
        logger.debug(f"Created secure temp directory: {temp_path}")
        return temp_path
        
    except (OSError, ValueError) as e:
        raise RuntimeError(f"Failed to create secure temp directory: {e}") from e


def sanitize_environment() -> Dict[str, str]:
    """Get sanitized environment variables for subprocess execution.
    
    Returns:
        Dictionary of sanitized environment variables
    """
    # Define allowed environment variables
    allowed_vars = {
        'PATH', 'HOME', 'USER', 'USERNAME', 'TMPDIR', 'TEMP', 'TMP',
        'EMSCRIPTEN', 'EMSCRIPTEN_ROOT', 'EMSDK', 'EMSDK_NODE',
        'CMAKE_PREFIX_PATH', 'PKG_CONFIG_PATH'
    }
    
    # Filter environment to only include allowed variables
    sanitized_env = {}
    
    for var in allowed_vars:
        if var in os.environ:
            sanitized_env[var] = os.environ[var]
    
    # Add essential minimal environment if missing
    if 'PATH' not in sanitized_env:
        sanitized_env['PATH'] = '/usr/local/bin:/usr/bin:/bin'
    
    if 'HOME' not in sanitized_env:
        sanitized_env['HOME'] = os.path.expanduser('~')
    
    return sanitized_env


def check_resource_limits() -> Dict[str, bool]:
    """Check system resource limits and availability.
    
    Returns:
        Dictionary of resource check results
    """
    limits = {}
    
    # Check available disk space
    try:
        import shutil
        free_space_gb = shutil.disk_usage('.').free / (1024**3)
        limits['sufficient_disk_space'] = free_space_gb >= 1.0
        limits['disk_space_gb'] = free_space_gb
    except Exception:
        limits['sufficient_disk_space'] = True  # Assume sufficient if can't check
        limits['disk_space_gb'] = float('inf')
    
    # Check available memory
    try:
        import psutil
        available_memory_gb = psutil.virtual_memory().available / (1024**3)
        limits['sufficient_memory'] = available_memory_gb >= 2.0
        limits['available_memory_gb'] = available_memory_gb
    except ImportError:
        limits['sufficient_memory'] = True  # Assume sufficient if psutil not available
        limits['available_memory_gb'] = float('inf')
    
    # Check process limits
    try:
        import resource
        max_processes = resource.getrlimit(resource.RLIMIT_NPROC)[0]
        limits['process_limit_ok'] = max_processes > 100
        limits['max_processes'] = max_processes
    except Exception:
        limits['process_limit_ok'] = True
        limits['max_processes'] = float('inf')
    
    return limits


def audit_log_event(event_type: str, details: Dict[str, str]) -> None:
    """Log security-relevant events for auditing.
    
    Args:
        event_type: Type of security event
        details: Event details to log
    """
    audit_logger = logging.getLogger(f"{__name__}.audit")
    
    audit_entry = f"SECURITY_EVENT: {event_type} | " + " | ".join(
        f"{k}={v}" for k, v in details.items()
    )
    
    audit_logger.info(audit_entry)