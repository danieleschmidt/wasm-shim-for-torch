"""Comprehensive Security Fortress for WASM-Torch v5.0

Multi-layered security framework with threat detection, input validation,
cryptographic integrity, and compliance management for enterprise deployments.
"""

import asyncio
import hashlib
import hmac
import secrets
import time
import json
import logging
import re
from typing import Dict, Any, List, Optional, Tuple, Union, Set
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import threading
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import base64

logger = logging.getLogger(__name__)

class ThreatLevel(Enum):
    """Security threat levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class SecurityEvent(Enum):
    """Types of security events."""
    UNAUTHORIZED_ACCESS = "unauthorized_access"
    INPUT_VALIDATION_FAILURE = "input_validation_failure"
    MODEL_INTEGRITY_VIOLATION = "model_integrity_violation"
    RESOURCE_ABUSE = "resource_abuse"
    SUSPICIOUS_ACTIVITY = "suspicious_activity"
    INJECTION_ATTEMPT = "injection_attempt"
    PRIVILEGE_ESCALATION = "privilege_escalation"
    DATA_EXFILTRATION_ATTEMPT = "data_exfiltration_attempt"

@dataclass
class SecurityIncident:
    """Security incident record."""
    event_type: SecurityEvent
    threat_level: ThreatLevel
    timestamp: float
    source_ip: str
    user_agent: str
    details: Dict[str, Any]
    mitigation_action: str = ""
    resolved: bool = False
    resolution_time: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'event_type': self.event_type.value,
            'threat_level': self.threat_level.value,
            'timestamp': self.timestamp,
            'source_ip': self.source_ip,
            'user_agent': self.user_agent,
            'details': self.details,
            'mitigation_action': self.mitigation_action,
            'resolved': self.resolved,
            'resolution_time': self.resolution_time
        }

class InputSanitizer:
    """Advanced input sanitization and validation."""
    
    def __init__(self):
        self.validation_rules = {
            'model_path': re.compile(r'^[a-zA-Z0-9_/\-\.]+\.wasm$'),
            'model_name': re.compile(r'^[a-zA-Z0-9_\-]+$'),
            'api_key': re.compile(r'^[a-zA-Z0-9]+$'),
            'user_id': re.compile(r'^[a-zA-Z0-9_\-]+$')
        }
        
        self.dangerous_patterns = [
            re.compile(r'<script[^>]*>.*?</script>', re.IGNORECASE | re.DOTALL),
            re.compile(r'javascript:', re.IGNORECASE),
            re.compile(r'on\w+\s*=', re.IGNORECASE),
            re.compile(r'\.\./'),  # Path traversal
            re.compile(r'eval\s*\(', re.IGNORECASE),
            re.compile(r'exec\s*\(', re.IGNORECASE),
            re.compile(r'system\s*\(', re.IGNORECASE),
            re.compile(r'subprocess\s*\.', re.IGNORECASE)
        ]
        
        self.max_sizes = {
            'model_path': 256,
            'model_name': 64,
            'description': 1024,
            'config': 4096,
            'tensor_data': 100_000_000  # 100MB
        }
    
    def sanitize_input(self, input_type: str, value: Any) -> Tuple[Any, List[str]]:
        """Sanitize input and return cleaned value and any issues found."""
        issues = []
        
        if value is None:
            return None, issues
        
        # Convert to string for pattern matching
        str_value = str(value)
        
        # Check for dangerous patterns
        for pattern in self.dangerous_patterns:
            if pattern.search(str_value):
                issues.append(f"Dangerous pattern detected: {pattern.pattern}")
        
        # Check size limits
        max_size = self.max_sizes.get(input_type, 1024)
        if len(str_value) > max_size:
            issues.append(f"Input too large: {len(str_value)} > {max_size}")
            str_value = str_value[:max_size]
        
        # Apply type-specific validation
        if input_type in self.validation_rules:
            if not self.validation_rules[input_type].match(str_value):
                issues.append(f"Invalid format for {input_type}")
        
        # Path traversal protection
        if 'path' in input_type.lower():
            cleaned_path = self._sanitize_path(str_value)
            if cleaned_path != str_value:
                issues.append("Path traversal attempt sanitized")
            str_value = cleaned_path
        
        return str_value, issues
    
    def _sanitize_path(self, path: str) -> str:
        """Sanitize file paths to prevent traversal attacks."""
        # Remove path traversal attempts
        path = re.sub(r'\.\./', '', path)
        path = re.sub(r'\.\.\\\\', '', path)
        
        # Normalize path
        path = Path(path).as_posix()
        
        # Ensure path doesn't start with /
        if path.startswith('/'):
            path = path[1:]
        
        return path
    
    def validate_tensor_data(self, tensor_data: Any) -> Tuple[bool, List[str]]:
        """Validate tensor data for security issues."""
        issues = []
        
        try:
            # Check if it's a reasonable tensor-like structure
            if hasattr(tensor_data, 'shape'):
                # Check tensor dimensions aren't excessive
                if len(tensor_data.shape) > 8:  # More than 8D is suspicious
                    issues.append("Tensor has excessive dimensions")
                
                # Check tensor size isn't excessive
                import numpy as np
                if hasattr(tensor_data, 'size'):
                    if tensor_data.size > 50_000_000:  # 50M elements
                        issues.append("Tensor size exceeds safety limits")
            
            # Check data type
            if hasattr(tensor_data, 'dtype'):
                allowed_dtypes = ['float32', 'float64', 'int32', 'int64', 'uint8']
                if str(tensor_data.dtype) not in allowed_dtypes:
                    issues.append(f"Unsafe tensor dtype: {tensor_data.dtype}")
            
        except Exception as e:
            issues.append(f"Error validating tensor: {str(e)}")
        
        return len(issues) == 0, issues

class CryptographicManager:
    """Manages cryptographic operations for secure model handling."""
    
    def __init__(self, master_key: Optional[bytes] = None):
        self.master_key = master_key or Fernet.generate_key()
        self.cipher_suite = Fernet(self.master_key)
        self.integrity_checksums = {}
        self._lock = threading.RLock()
    
    def encrypt_model_data(self, model_data: bytes) -> bytes:
        """Encrypt model data for secure storage."""
        with self._lock:
            return self.cipher_suite.encrypt(model_data)
    
    def decrypt_model_data(self, encrypted_data: bytes) -> bytes:
        """Decrypt model data."""
        with self._lock:
            try:
                return self.cipher_suite.decrypt(encrypted_data)
            except Exception as e:
                logger.error(f"Decryption failed: {e}")
                raise SecurityException("Model decryption failed")
    
    def generate_model_hash(self, model_data: bytes) -> str:
        """Generate cryptographic hash for model integrity verification."""
        return hashlib.sha256(model_data).hexdigest()
    
    def verify_model_integrity(self, model_id: str, model_data: bytes) -> bool:
        """Verify model integrity using stored checksum."""
        with self._lock:
            if model_id not in self.integrity_checksums:
                # First time - store the checksum
                self.integrity_checksums[model_id] = self.generate_model_hash(model_data)
                return True
            
            expected_hash = self.integrity_checksums[model_id]
            actual_hash = self.generate_model_hash(model_data)
            
            return hmac.compare_digest(expected_hash, actual_hash)
    
    def generate_secure_token(self, length: int = 32) -> str:
        """Generate cryptographically secure random token."""
        return secrets.token_urlsafe(length)
    
    def sign_response(self, response_data: str, secret_key: bytes) -> str:
        """Create HMAC signature for response integrity."""
        signature = hmac.new(
            secret_key,
            response_data.encode(),
            hashlib.sha256
        ).hexdigest()
        return signature

class ThreatDetector:
    """Advanced threat detection system."""
    
    def __init__(self):
        self.request_patterns = {}
        self.rate_limits = {}
        self.suspicious_ips = set()
        self.threat_scores = {}
        self._lock = threading.RLock()
        
        # Initialize threat detection patterns
        self._setup_threat_patterns()
    
    def _setup_threat_patterns(self) -> None:
        """Setup threat detection patterns."""
        self.threat_patterns = {
            'sql_injection': [
                r"union\s+select", r"or\s+1\s*=\s*1", r"drop\s+table",
                r"insert\s+into", r"delete\s+from"
            ],
            'xss_attack': [
                r"<script", r"javascript:", r"onerror=", r"onload=",
                r"eval\s*\(", r"document\.cookie"
            ],
            'path_traversal': [
                r"\.\./", r"\.\.\\", r"/etc/passwd", r"\.\.%2f",
                r"%2e%2e%2f", r"..%252f"
            ],
            'command_injection': [
                r";\s*cat\s+", r";\s*ls\s+", r";\s*rm\s+", r"\|\s*nc\s+",
                r"&&\s*cat", r"\|\s*curl"
            ]
        }
        
        # Compile patterns for efficiency
        self.compiled_patterns = {}
        for category, patterns in self.threat_patterns.items():
            self.compiled_patterns[category] = [
                re.compile(pattern, re.IGNORECASE) for pattern in patterns
            ]
    
    async def analyze_request(self, 
                            request_data: Dict[str, Any], 
                            source_ip: str) -> Tuple[ThreatLevel, List[str]]:
        """Analyze request for potential threats."""
        
        threats = []
        threat_level = ThreatLevel.LOW
        
        # Rate limiting check
        rate_threat = self._check_rate_limit(source_ip)
        if rate_threat:
            threats.append("Rate limit exceeded")
            threat_level = ThreatLevel.MEDIUM
        
        # Pattern-based threat detection
        for key, value in request_data.items():
            if isinstance(value, str):
                pattern_threats = self._check_threat_patterns(value)
                threats.extend(pattern_threats)
                
                if pattern_threats:
                    threat_level = ThreatLevel.HIGH
        
        # Check for suspicious IP
        if source_ip in self.suspicious_ips:
            threats.append("Request from suspicious IP")
            threat_level = ThreatLevel.HIGH
        
        # Behavioral analysis
        behavioral_threats = await self._analyze_behavior(source_ip, request_data)
        threats.extend(behavioral_threats)
        
        if behavioral_threats:
            threat_level = max(threat_level, ThreatLevel.MEDIUM)
        
        # Update threat score
        with self._lock:
            self.threat_scores[source_ip] = self.threat_scores.get(source_ip, 0) + len(threats)
            
            # Mark as suspicious if threat score is high
            if self.threat_scores[source_ip] > 10:
                self.suspicious_ips.add(source_ip)
                threat_level = ThreatLevel.CRITICAL
        
        return threat_level, threats
    
    def _check_rate_limit(self, source_ip: str, limit: int = 100, window: int = 3600) -> bool:
        """Check if IP has exceeded rate limit."""
        current_time = time.time()
        
        with self._lock:
            if source_ip not in self.rate_limits:
                self.rate_limits[source_ip] = []
            
            # Clean old requests
            self.rate_limits[source_ip] = [
                timestamp for timestamp in self.rate_limits[source_ip]
                if current_time - timestamp < window
            ]
            
            # Add current request
            self.rate_limits[source_ip].append(current_time)
            
            return len(self.rate_limits[source_ip]) > limit
    
    def _check_threat_patterns(self, input_string: str) -> List[str]:
        """Check input string against threat patterns."""
        detected_threats = []
        
        for category, patterns in self.compiled_patterns.items():
            for pattern in patterns:
                if pattern.search(input_string):
                    detected_threats.append(f"Potential {category} detected")
                    break  # One detection per category is enough
        
        return detected_threats
    
    async def _analyze_behavior(self, source_ip: str, request_data: Dict[str, Any]) -> List[str]:
        """Analyze behavioral patterns for anomalies."""
        threats = []
        
        # Check request frequency patterns
        with self._lock:
            if source_ip in self.request_patterns:
                pattern = self.request_patterns[source_ip]
                
                # Check for rapid successive requests
                recent_requests = [
                    t for t in pattern['timestamps'][-10:]  # Last 10 requests
                    if time.time() - t < 60  # Within last minute
                ]
                
                if len(recent_requests) > 50:  # More than 50 requests per minute
                    threats.append("Abnormal request frequency")
                
                # Check for repeated identical requests
                if len(set(pattern['request_hashes'][-5:])) == 1:  # Last 5 requests identical
                    threats.append("Suspicious request repetition")
            else:
                self.request_patterns[source_ip] = {
                    'timestamps': [],
                    'request_hashes': []
                }
            
            # Record current request
            request_hash = hashlib.md5(json.dumps(request_data, sort_keys=True).encode()).hexdigest()
            pattern = self.request_patterns[source_ip]
            pattern['timestamps'].append(time.time())
            pattern['request_hashes'].append(request_hash)
            
            # Keep only recent history
            pattern['timestamps'] = pattern['timestamps'][-100:]
            pattern['request_hashes'] = pattern['request_hashes'][-100:]
        
        return threats

class AccessControlManager:
    """Role-based access control system."""
    
    def __init__(self):
        self.user_roles = {}
        self.role_permissions = {
            'guest': ['read_models'],
            'user': ['read_models', 'run_inference'],
            'developer': ['read_models', 'run_inference', 'upload_models'],
            'admin': ['read_models', 'run_inference', 'upload_models', 'manage_users', 'view_security_logs'],
            'system': ['*']  # System has all permissions
        }
        self.session_tokens = {}
        self._lock = threading.RLock()
    
    def authenticate_user(self, username: str, password: str) -> Optional[str]:
        """Authenticate user and return session token."""
        # In real implementation, this would verify against secure user store
        # For now, simulate authentication
        
        if self._verify_credentials(username, password):
            token = secrets.token_urlsafe(32)
            
            with self._lock:
                self.session_tokens[token] = {
                    'username': username,
                    'role': self.user_roles.get(username, 'user'),
                    'created_at': time.time(),
                    'last_used': time.time()
                }
            
            return token
        
        return None
    
    def _verify_credentials(self, username: str, password: str) -> bool:
        """Verify user credentials (simplified for demo)."""
        # In production, use proper password hashing
        valid_users = {
            'admin': 'admin123',
            'developer': 'dev123',
            'user': 'user123'
        }
        
        return username in valid_users and valid_users[username] == password
    
    def authorize_action(self, token: str, action: str) -> bool:
        """Check if token has permission for action."""
        with self._lock:
            if token not in self.session_tokens:
                return False
            
            session = self.session_tokens[token]
            
            # Check if token is expired (1 hour)
            if time.time() - session['created_at'] > 3600:
                del self.session_tokens[token]
                return False
            
            # Update last used
            session['last_used'] = time.time()
            
            user_role = session['role']
            permissions = self.role_permissions.get(user_role, [])
            
            return '*' in permissions or action in permissions
    
    def revoke_token(self, token: str) -> None:
        """Revoke session token."""
        with self._lock:
            self.session_tokens.pop(token, None)

class SecurityException(Exception):
    """Security-related exception."""
    pass

class ComprehensiveSecurityFortress:
    """Main security fortress coordinating all security components."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        
        self.input_sanitizer = InputSanitizer()
        self.crypto_manager = CryptographicManager()
        self.threat_detector = ThreatDetector()
        self.access_control = AccessControlManager()
        
        self.security_incidents = []
        self.security_metrics = {
            'total_requests': 0,
            'blocked_requests': 0,
            'security_incidents': 0,
            'threats_detected': 0
        }
        
        self._active = False
        self._monitoring_task = None
    
    async def initialize(self) -> None:
        """Initialize the security fortress."""
        logger.info("🛡️ Initializing Comprehensive Security Fortress")
        
        try:
            # Start security monitoring
            await self._start_security_monitoring()
            
            self._active = True
            logger.info("✅ Security Fortress initialized")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize security fortress: {e}")
            raise
    
    async def validate_request(self, 
                             request_data: Dict[str, Any],
                             source_ip: str = "127.0.0.1",
                             user_agent: str = "unknown",
                             auth_token: Optional[str] = None) -> Tuple[bool, Dict[str, Any]]:
        """Comprehensive request validation."""
        
        self.security_metrics['total_requests'] += 1
        validation_result = {
            'allowed': False,
            'sanitized_data': {},
            'security_issues': [],
            'threat_level': ThreatLevel.LOW,
            'mitigation_actions': []
        }
        
        try:
            # 1. Input Sanitization
            sanitized_data = {}
            all_issues = []
            
            for key, value in request_data.items():
                sanitized_value, issues = self.input_sanitizer.sanitize_input(key, value)
                sanitized_data[key] = sanitized_value
                all_issues.extend(issues)
            
            validation_result['sanitized_data'] = sanitized_data
            validation_result['security_issues'].extend(all_issues)
            
            # 2. Threat Detection
            threat_level, threats = await self.threat_detector.analyze_request(
                request_data, source_ip
            )
            validation_result['threat_level'] = threat_level
            validation_result['security_issues'].extend(threats)
            
            # 3. Authentication & Authorization
            if auth_token:
                # Check token validity and permissions
                if not self._validate_auth_token(auth_token, request_data):
                    validation_result['security_issues'].append("Invalid or expired authentication")
                    threat_level = ThreatLevel.HIGH
            
            # 4. Determine if request should be allowed
            if threat_level in [ThreatLevel.HIGH, ThreatLevel.CRITICAL]:
                validation_result['allowed'] = False
                validation_result['mitigation_actions'].append("Request blocked due to high threat level")
                self.security_metrics['blocked_requests'] += 1
                
                # Log security incident
                await self._log_security_incident(
                    SecurityEvent.SUSPICIOUS_ACTIVITY,
                    threat_level,
                    source_ip,
                    user_agent,
                    {
                        'request_data': request_data,
                        'issues': validation_result['security_issues']
                    }
                )
                
            elif len(all_issues) > 0:
                # Allow but with warnings
                validation_result['allowed'] = True
                validation_result['mitigation_actions'].append("Request allowed with sanitization")
            else:
                # Clean request
                validation_result['allowed'] = True
            
            if validation_result['security_issues']:
                self.security_metrics['threats_detected'] += 1
            
        except Exception as e:
            logger.error(f"Security validation error: {e}")
            validation_result['allowed'] = False
            validation_result['security_issues'].append(f"Validation error: {str(e)}")
        
        return validation_result['allowed'], validation_result
    
    def _validate_auth_token(self, token: str, request_data: Dict[str, Any]) -> bool:
        """Validate authentication token."""
        # Determine required action based on request
        required_action = self._determine_required_permission(request_data)
        
        # Check authorization
        return self.access_control.authorize_action(token, required_action)
    
    def _determine_required_permission(self, request_data: Dict[str, Any]) -> str:
        """Determine required permission based on request."""
        # Simplified permission mapping
        if 'model_upload' in request_data:
            return 'upload_models'
        elif 'inference' in request_data:
            return 'run_inference'
        elif 'admin' in request_data:
            return 'manage_users'
        else:
            return 'read_models'
    
    async def _log_security_incident(self,
                                   event_type: SecurityEvent,
                                   threat_level: ThreatLevel,
                                   source_ip: str,
                                   user_agent: str,
                                   details: Dict[str, Any]) -> None:
        """Log security incident."""
        
        incident = SecurityIncident(
            event_type=event_type,
            threat_level=threat_level,
            timestamp=time.time(),
            source_ip=source_ip,
            user_agent=user_agent,
            details=details
        )
        
        self.security_incidents.append(incident)
        self.security_metrics['security_incidents'] += 1
        
        logger.warning(f"🚨 Security incident: {event_type.value} from {source_ip}")
        
        # Keep only recent incidents (last 1000)
        if len(self.security_incidents) > 1000:
            self.security_incidents = self.security_incidents[-1000:]
    
    async def _start_security_monitoring(self) -> None:
        """Start background security monitoring."""
        logger.debug("Starting security monitoring...")
        
        async def security_monitor():
            while self._active:
                try:
                    await self._perform_security_maintenance()
                    await asyncio.sleep(300)  # Run every 5 minutes
                except Exception as e:
                    logger.error(f"Security monitoring error: {e}")
                    await asyncio.sleep(60)  # Brief delay before retry
        
        self._monitoring_task = asyncio.create_task(security_monitor())
    
    async def _perform_security_maintenance(self) -> None:
        """Perform periodic security maintenance."""
        current_time = time.time()
        
        # Clean up old rate limit data
        with self.threat_detector._lock:
            for ip in list(self.threat_detector.rate_limits.keys()):
                self.threat_detector.rate_limits[ip] = [
                    timestamp for timestamp in self.threat_detector.rate_limits[ip]
                    if current_time - timestamp < 3600  # Keep last hour
                ]
                
                if not self.threat_detector.rate_limits[ip]:
                    del self.threat_detector.rate_limits[ip]
        
        # Clean up old session tokens
        with self.access_control._lock:
            expired_tokens = [
                token for token, session in self.access_control.session_tokens.items()
                if current_time - session['created_at'] > 3600  # 1 hour expiry
            ]
            
            for token in expired_tokens:
                del self.access_control.session_tokens[token]
        
        logger.debug("Security maintenance completed")
    
    def get_security_status(self) -> Dict[str, Any]:
        """Get comprehensive security status."""
        return {
            'metrics': self.security_metrics.copy(),
            'threat_summary': {
                'suspicious_ips': len(self.threat_detector.suspicious_ips),
                'high_threat_ips': len([
                    ip for ip, score in self.threat_detector.threat_scores.items()
                    if score > 5
                ]),
                'active_sessions': len(self.access_control.session_tokens)
            },
            'recent_incidents': [
                incident.to_dict()
                for incident in self.security_incidents[-10:]  # Last 10 incidents
            ],
            'fortress_status': 'active' if self._active else 'inactive'
        }
    
    async def cleanup(self) -> None:
        """Clean up security fortress resources."""
        logger.info("🧹 Cleaning up Security Fortress")
        
        self._active = False
        
        if self._monitoring_task:
            self._monitoring_task.cancel()
            try:
                await self._monitoring_task
            except asyncio.CancelledError:
                pass
        
        logger.info("✅ Security Fortress cleanup complete")

# Global security fortress instance
_security_fortress: Optional[ComprehensiveSecurityFortress] = None

async def get_security_fortress(config: Optional[Dict[str, Any]] = None) -> ComprehensiveSecurityFortress:
    """Get or create the global security fortress."""
    global _security_fortress
    
    if _security_fortress is None:
        _security_fortress = ComprehensiveSecurityFortress(config)
        await _security_fortress.initialize()
    
    return _security_fortress

# Export public API
__all__ = [
    'ComprehensiveSecurityFortress',
    'InputSanitizer',
    'CryptographicManager',
    'ThreatDetector',
    'AccessControlManager',
    'SecurityIncident',
    'SecurityEvent',
    'ThreatLevel',
    'SecurityException',
    'get_security_fortress'
]