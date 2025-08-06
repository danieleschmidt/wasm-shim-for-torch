"""Tests for security utilities."""

import pytest
import tempfile
import os
from pathlib import Path
from wasm_torch.security import (
    SecurityConfig,
    validate_model_path,
    verify_model_integrity,
    compute_file_hash,
    create_secure_temp_dir,
    sanitize_environment,
    check_resource_limits,
    audit_log_event
)


class TestSecurityConfig:
    """Test SecurityConfig class."""
    
    def test_default_config(self):
        """Test default security configuration."""
        config = SecurityConfig()
        assert '.wasm' in config.allowed_model_extensions
        assert '.pth' in config.allowed_model_extensions
        assert config.max_model_size_mb == 1000.0
        assert config.enable_model_verification is True
    
    def test_custom_config(self):
        """Test custom security configuration."""
        config = SecurityConfig(
            allowed_model_extensions={'.custom'},
            max_model_size_mb=100.0,
            allowed_directories=['/safe/path'],
            enable_model_verification=False
        )
        assert config.allowed_model_extensions == {'.custom'}
        assert config.max_model_size_mb == 100.0
        assert config.allowed_directories == ['/safe/path']
        assert config.enable_model_verification is False


class TestModelPathValidation:
    """Test model path validation."""
    
    def test_validate_model_path_valid(self, tmp_path):
        """Test validation with valid model path."""
        model_file = tmp_path / "model.wasm"
        model_file.write_text("dummy model content")
        
        config = SecurityConfig()
        result = validate_model_path(model_file, config)
        assert result == model_file.resolve()
    
    def test_validate_model_path_nonexistent(self, tmp_path):
        """Test validation with non-existent file."""
        model_file = tmp_path / "nonexistent.wasm"
        
        config = SecurityConfig()
        with pytest.raises(FileNotFoundError):
            validate_model_path(model_file, config)
    
    def test_validate_model_path_invalid_extension(self, tmp_path):
        """Test validation with invalid file extension."""
        model_file = tmp_path / "model.exe"
        model_file.write_text("malicious content")
        
        config = SecurityConfig()
        with pytest.raises(ValueError, match="File extension .exe not allowed"):
            validate_model_path(model_file, config)
    
    def test_validate_model_path_too_large(self, tmp_path):
        """Test validation with oversized file."""
        model_file = tmp_path / "huge_model.wasm"
        # Create a file that appears large
        model_file.write_text("x" * 100)  # Small file for test
        
        # Use very small limit to trigger size check
        config = SecurityConfig(max_model_size_mb=0.00001)
        with pytest.raises(ValueError, match="Model file too large"):
            validate_model_path(model_file, config)
    
    def test_validate_model_path_directory_restriction(self, tmp_path):
        """Test validation with directory restrictions."""
        safe_dir = tmp_path / "safe"
        safe_dir.mkdir()
        unsafe_dir = tmp_path / "unsafe"
        unsafe_dir.mkdir()
        
        safe_model = safe_dir / "model.wasm"
        safe_model.write_text("safe model")
        
        unsafe_model = unsafe_dir / "model.wasm"
        unsafe_model.write_text("unsafe model")
        
        config = SecurityConfig(allowed_directories=[str(safe_dir)])
        
        # Should pass for safe directory
        result = validate_model_path(safe_model, config)
        assert result == safe_model.resolve()
        
        # Should fail for unsafe directory
        with pytest.raises(ValueError, match="Model file not in allowed directory"):
            validate_model_path(unsafe_model, config)


class TestFileHashing:
    """Test file hashing functions."""
    
    def test_compute_file_hash(self, tmp_path):
        """Test computing file hash."""
        test_file = tmp_path / "test.txt"
        test_content = "Hello, WASM Torch!"
        test_file.write_text(test_content)
        
        hash_value = compute_file_hash(test_file)
        assert len(hash_value) == 64  # SHA-256 hex digest length
        assert isinstance(hash_value, str)
        
        # Same content should produce same hash
        hash_value2 = compute_file_hash(test_file)
        assert hash_value == hash_value2
    
    def test_compute_file_hash_different_algorithms(self, tmp_path):
        """Test different hash algorithms."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("test content")
        
        sha256_hash = compute_file_hash(test_file, 'sha256')
        md5_hash = compute_file_hash(test_file, 'md5')
        
        assert len(sha256_hash) == 64
        assert len(md5_hash) == 32
        assert sha256_hash != md5_hash
    
    def test_verify_model_integrity_pass(self, tmp_path):
        """Test model integrity verification success."""
        model_file = tmp_path / "model.wasm"
        model_file.write_text("model content")
        
        # Compute expected hash
        expected_hash = compute_file_hash(model_file)
        
        # Verification should pass
        result = verify_model_integrity(model_file, expected_hash)
        assert result is True
    
    def test_verify_model_integrity_fail(self, tmp_path):
        """Test model integrity verification failure."""
        model_file = tmp_path / "model.wasm"
        model_file.write_text("model content")
        
        wrong_hash = "0" * 64  # Wrong hash
        
        with pytest.raises(RuntimeError, match="Model integrity verification failed"):
            verify_model_integrity(model_file, wrong_hash)


class TestSecureTemporaryDirectory:
    """Test secure temporary directory creation."""
    
    def test_create_secure_temp_dir(self):
        """Test creating secure temporary directory."""
        temp_dir = create_secure_temp_dir("test_prefix_")
        
        try:
            assert temp_dir.exists()
            assert temp_dir.is_dir()
            assert temp_dir.name.startswith("test_prefix_")
            
            # Check permissions (Unix-style)
            if hasattr(os, 'stat'):
                stat_info = temp_dir.stat()
                # Should be readable/writable/executable by owner only
                assert oct(stat_info.st_mode)[-3:] == '700'
        finally:
            # Clean up
            if temp_dir.exists():
                temp_dir.rmdir()


class TestEnvironmentSanitization:
    """Test environment variable sanitization."""
    
    def test_sanitize_environment(self):
        """Test environment sanitization."""
        sanitized_env = sanitize_environment()
        
        # Should contain essential variables
        assert 'PATH' in sanitized_env
        assert 'HOME' in sanitized_env
        
        # Should not contain potentially dangerous variables
        dangerous_vars = ['LD_PRELOAD', 'LD_LIBRARY_PATH', 'PYTHONPATH']
        for var in dangerous_vars:
            assert var not in sanitized_env
        
        # All values should be strings
        for key, value in sanitized_env.items():
            assert isinstance(key, str)
            assert isinstance(value, str)


class TestResourceLimits:
    """Test resource limit checking."""
    
    def test_check_resource_limits(self):
        """Test resource limit checking."""
        limits = check_resource_limits()
        
        # Should contain expected keys
        expected_keys = [
            'sufficient_disk_space', 'disk_space_gb',
            'sufficient_memory', 'available_memory_gb',
            'process_limit_ok', 'max_processes'
        ]
        
        for key in expected_keys:
            assert key in limits
        
        # Boolean checks should be boolean
        assert isinstance(limits['sufficient_disk_space'], bool)
        assert isinstance(limits['sufficient_memory'], bool)
        assert isinstance(limits['process_limit_ok'], bool)
        
        # Numeric values should be numeric
        assert isinstance(limits['disk_space_gb'], (int, float))
        assert isinstance(limits['available_memory_gb'], (int, float))
        assert isinstance(limits['max_processes'], (int, float))


class TestAuditLogging:
    """Test audit logging functionality."""
    
    def test_audit_log_event(self, caplog):
        """Test audit event logging."""
        import logging
        
        # Set up logging to capture audit events
        audit_logger = logging.getLogger('wasm_torch.security.audit')
        audit_logger.setLevel(logging.INFO)
        
        with caplog.at_level(logging.INFO, logger='wasm_torch.security.audit'):
            audit_log_event("test_event", {
                "user": "test_user",
                "action": "model_export",
                "result": "success"
            })
        
        # Check that the event was logged
        assert len(caplog.records) == 1
        record = caplog.records[0]
        assert "SECURITY_EVENT: test_event" in record.message
        assert "user=test_user" in record.message
        assert "action=model_export" in record.message
        assert "result=success" in record.message