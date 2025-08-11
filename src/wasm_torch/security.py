"""Security utilities for WASM Torch."""

import os
import hashlib
import tempfile
import time
from pathlib import Path
from typing import Dict, List, Optional, Set, Union, Any
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


def validate_path(path: Union[str, Path], allow_write: bool = False) -> Path:
    """Enhanced path validation with comprehensive security checks.
    
    Args:
        path: Path to validate
        allow_write: Whether write access is allowed
        
    Returns:
        Validated Path object
        
    Raises:
        ValueError: If path is unsafe
        PermissionError: If access is denied
    """
    from pathlib import Path
    
    try:
        path_obj = Path(path).resolve()
    except (OSError, ValueError) as e:
        raise ValueError(f"Invalid path: {path}") from e
    
    # Check for directory traversal
    if ".." in str(path_obj) or str(path_obj).startswith("/.."):
        raise ValueError(f"Directory traversal detected: {path}")
    
    # Check for access to restricted system directories
    restricted_dirs = [
        "/etc", "/usr/bin", "/usr/sbin", "/bin", "/sbin", "/boot",
        "/proc", "/sys", "/dev", "/var/run", "/var/lib", "/opt/system"
    ]
    
    for restricted in restricted_dirs:
        if str(path_obj).startswith(restricted):
            raise ValueError(f"Access to restricted directory {restricted} denied")
    
    # For write operations, ensure we have permission
    if allow_write:
        parent_dir = path_obj.parent
        if not parent_dir.exists():
            try:
                parent_dir.mkdir(parents=True, exist_ok=True)
            except PermissionError as e:
                raise PermissionError(f"Cannot create directory {parent_dir}") from e
        
        if path_obj.exists() and not os.access(path_obj, os.W_OK):
            raise PermissionError(f"No write permission for {path_obj}")
            
        if not path_obj.exists() and not os.access(parent_dir, os.W_OK):
            raise PermissionError(f"No write permission for directory {parent_dir}")
    
    return path_obj


def log_security_event(event_type: str, details: Dict[str, Any]) -> None:
    """Log security events with structured format.
    
    Args:
        event_type: Type of security event
        details: Event details
    """
    import time
    import json
    
    event_data = {
        "timestamp": time.time(),
        "event_type": event_type,
        "details": details,
        "process_id": os.getpid()
    }
    
    security_logger = logging.getLogger(f"{__name__}.security")
    security_logger.info(json.dumps(event_data))


def check_file_permissions(file_path: Path) -> Dict[str, bool]:
    """Check file permissions and ownership.
    
    Args:
        file_path: Path to check
        
    Returns:
        Dictionary with permission information
    """
    import stat
    
    try:
        file_stat = file_path.stat()
        
        permissions = {
            "readable": os.access(file_path, os.R_OK),
            "writable": os.access(file_path, os.W_OK),
            "executable": os.access(file_path, os.X_OK),
            "owner_readable": bool(file_stat.st_mode & stat.S_IRUSR),
            "owner_writable": bool(file_stat.st_mode & stat.S_IWUSR),
            "owner_executable": bool(file_stat.st_mode & stat.S_IXUSR),
            "group_readable": bool(file_stat.st_mode & stat.S_IRGRP),
            "group_writable": bool(file_stat.st_mode & stat.S_IWGRP),
            "other_readable": bool(file_stat.st_mode & stat.S_IROTH),
            "other_writable": bool(file_stat.st_mode & stat.S_IWOTH),
            "is_secure": not (file_stat.st_mode & (stat.S_IWGRP | stat.S_IWOTH))  # Not writable by group/other
        }
        
        return permissions
        
    except (OSError, PermissionError) as e:
        logger.error(f"Cannot check permissions for {file_path}: {e}")
        return {"error": str(e)}


def secure_subprocess_env() -> Dict[str, str]:
    """Create secure environment for subprocess execution.
    
    Returns:
        Sanitized environment variables
    """
    # Base secure environment
    secure_env = {
        "PATH": "/usr/local/bin:/usr/bin:/bin",
        "LANG": "C",
        "LC_ALL": "C"
    }
    
    # Add essential development tools if available
    essential_vars = ["EMSCRIPTEN", "EMSDK", "EMSDK_NODE", "CMAKE_PREFIX_PATH"]
    
    for var in essential_vars:
        if var in os.environ:
            secure_env[var] = os.environ[var]
    
    # Add current working directory to PATH if safe
    cwd = os.getcwd()
    if not any(restricted in cwd for restricted in ["/etc", "/usr", "/bin", "/sbin"]):
        secure_env["PATH"] = f"{cwd}:{secure_env['PATH']}"
    
    return secure_env


def validate_model_metadata(metadata: Dict[str, Any]) -> bool:
    """Validate model metadata for security issues.
    
    Args:
        metadata: Model metadata dictionary
        
    Returns:
        True if metadata is safe
        
    Raises:
        ValueError: If metadata contains unsafe elements
    """
    # Check for suspicious keys
    suspicious_keys = [
        "__builtins__", "__import__", "exec", "eval", "compile",
        "open", "file", "input", "raw_input", "__file__", "__name__"
    ]
    
    def check_dict_recursive(d: Dict[str, Any], path: str = "") -> None:
        for key, value in d.items():
            full_path = f"{path}.{key}" if path else key
            
            # Check for suspicious keys
            if key in suspicious_keys:
                raise ValueError(f"Suspicious key '{key}' found in metadata at {full_path}")
            
            # Check string values for suspicious content
            if isinstance(value, str):
                if any(sus in value.lower() for sus in ["__import__", "exec(", "eval(", "os.system"]):
                    raise ValueError(f"Suspicious string content in metadata at {full_path}")
            
            # Recursively check nested dictionaries
            elif isinstance(value, dict):
                check_dict_recursive(value, full_path)
            
            # Check list items
            elif isinstance(value, list):
                for i, item in enumerate(value):
                    if isinstance(item, dict):
                        check_dict_recursive(item, f"{full_path}[{i}]")
                    elif isinstance(item, str) and any(sus in item.lower() for sus in ["__import__", "exec(", "eval("]):
                        raise ValueError(f"Suspicious list item in metadata at {full_path}[{i}]")
    
    try:
        check_dict_recursive(metadata)
        return True
    except Exception as e:
        logger.error(f"Metadata validation failed: {e}")
        raise


class SecurityManager:
    """Centralized security management for WASM Torch operations."""
    
    def __init__(self, config: Optional[SecurityConfig] = None):
        self.config = config or SecurityConfig()
        self.audit_logger = logging.getLogger(f"{__name__}.audit")
        self._security_events = []
        
    def validate_operation(self, operation_type: str, **kwargs) -> bool:
        """Validate security for a specific operation.
        
        Args:
            operation_type: Type of operation to validate
            **kwargs: Operation-specific parameters
            
        Returns:
            True if operation is allowed
        """
        try:
            if operation_type == "model_load":
                model_path = kwargs.get("model_path")
                if model_path:
                    validate_path(model_path, allow_write=False)
                    
            elif operation_type == "model_export":
                output_path = kwargs.get("output_path")
                if output_path:
                    validate_path(output_path, allow_write=True)
                    
            elif operation_type == "compilation":
                # Check compilation environment is secure
                env_status = validate_compilation_environment()
                if not env_status.get("emscripten_available", False):
                    logger.warning("Emscripten not available for secure compilation")
                    
            self._log_security_event("operation_validated", {
                "operation_type": operation_type,
                "status": "allowed"
            })
            
            return True
            
        except Exception as e:
            self._log_security_event("operation_blocked", {
                "operation_type": operation_type,
                "reason": str(e)
            })
            raise
            
    def _log_security_event(self, event_type: str, details: Dict[str, Any]) -> None:
        """Internal method to log security events."""
        event = {
            "timestamp": time.time(),
            "event_type": event_type,
            "details": details
        }
        
        self._security_events.append(event)
        self.audit_logger.info(f"SECURITY: {event_type} - {details}")
        
    def get_security_report(self) -> Dict[str, Any]:
        """Generate security report with recent events.
        
        Returns:
            Security report dictionary
        """
        import time
        
        recent_events = [
            event for event in self._security_events
            if time.time() - event["timestamp"] < 3600  # Last hour
        ]
        
        return {
            "total_events": len(self._security_events),
            "recent_events": len(recent_events),
            "recent_event_types": list(set(e["event_type"] for e in recent_events)),
            "config": {
                "max_model_size_mb": self.config.max_model_size_mb,
                "allowed_extensions": list(self.config.allowed_model_extensions),
                "verification_enabled": self.config.enable_model_verification
            }
        }


# Import additional dependencies if needed
try:
    from typing import Union, Any
    import time
except ImportError:
    pass