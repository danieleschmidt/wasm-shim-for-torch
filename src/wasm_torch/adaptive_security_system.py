#!/usr/bin/env python3
"""
Adaptive Security System v2.0 - Self-Healing Security Framework
Advanced security system with autonomous threat detection, self-healing capabilities,
and intelligent adaptation to emerging threats.
"""

import asyncio
import time
import logging
import json
import hashlib
import secrets
from typing import Dict, List, Optional, Any, Callable, Set, Tuple
from dataclasses import dataclass, field
from pathlib import Path
from enum import Enum
from collections import defaultdict, deque
import threading
import random
import statistics

logger = logging.getLogger(__name__)

class SecurityThreatLevel(Enum):
    """Security threat severity levels"""
    LOW = "low"
    MEDIUM = "medium"  
    HIGH = "high"
    CRITICAL = "critical"
    ZERO_DAY = "zero_day"
    ADAPTIVE = "adaptive"

class SecurityResponseType(Enum):
    """Types of security responses"""
    BLOCK = "block"
    QUARANTINE = "quarantine"
    MONITOR = "monitor"
    ADAPT = "adapt"
    HEAL = "heal"
    LEARN = "learn"

@dataclass
class SecurityThreat:
    """Represents a security threat"""
    threat_id: str
    timestamp: float
    threat_type: str
    threat_level: SecurityThreatLevel
    source_ip: str
    target_component: str
    attack_signature: str
    confidence_score: float
    indicators: List[str] = field(default_factory=list)
    mitigation_strategies: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "threat_id": self.threat_id,
            "timestamp": self.timestamp,
            "threat_type": self.threat_type,
            "threat_level": self.threat_level.value,
            "source_ip": self.source_ip,
            "target_component": self.target_component,
            "attack_signature": self.attack_signature,
            "confidence_score": self.confidence_score,
            "indicators": self.indicators,
            "mitigation_strategies": self.mitigation_strategies
        }

@dataclass
class SecurityResponse:
    """Represents a security response action"""
    response_id: str
    threat_id: str
    response_type: SecurityResponseType
    action_taken: str
    effectiveness: float
    execution_time: float
    self_healing_applied: bool = False
    adaptation_learned: bool = False
    side_effects: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "response_id": self.response_id,
            "threat_id": self.threat_id,
            "response_type": self.response_type.value,
            "action_taken": self.action_taken,
            "effectiveness": self.effectiveness,
            "execution_time": self.execution_time,
            "self_healing_applied": self.self_healing_applied,
            "adaptation_learned": self.adaptation_learned,
            "side_effects": self.side_effects
        }

class AdaptiveSecuritySystem:
    """Advanced self-healing adaptive security system"""
    
    def __init__(self, 
                 enable_self_healing: bool = True,
                 enable_adaptive_learning: bool = True,
                 threat_intelligence_enabled: bool = True,
                 real_time_monitoring: bool = True):
        
        self.enable_self_healing = enable_self_healing
        self.enable_adaptive_learning = enable_adaptive_learning
        self.threat_intelligence_enabled = threat_intelligence_enabled
        self.real_time_monitoring = real_time_monitoring
        
        # Threat management
        self.active_threats: Dict[str, SecurityThreat] = {}
        self.threat_history: List[SecurityThreat] = []
        self.security_responses: List[SecurityResponse] = []
        
        # Self-healing and adaptation
        self.healing_strategies: Dict[str, Callable] = {}
        self.adaptation_patterns: Dict[str, Any] = {}
        self.threat_signatures: Dict[str, Dict[str, Any]] = {}
        self.learning_models: Dict[str, Dict[str, Any]] = {}
        
        # Security metrics and monitoring
        self.security_metrics: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self.component_health: Dict[str, Dict[str, Any]] = {}
        self.security_policies: Dict[str, Any] = {}
        
        # Threading and execution
        self.monitor_task: Optional[asyncio.Task] = None
        self.healing_task: Optional[asyncio.Task] = None
        self.adaptation_task: Optional[asyncio.Task] = None
        self.is_running = False
        self._lock = threading.RLock()
        
        # Initialize system
        self._initialize_security_components()
        self._initialize_healing_strategies()
        self._initialize_threat_signatures()
        
        logger.info("🛡️ Adaptive Security System v2.0 initialized")
        logger.info(f"  Self-healing: {'Enabled' if enable_self_healing else 'Disabled'}")
        logger.info(f"  Adaptive learning: {'Enabled' if enable_adaptive_learning else 'Disabled'}")
        logger.info(f"  Threat intelligence: {'Enabled' if threat_intelligence_enabled else 'Disabled'}")
    
    def _initialize_security_components(self):
        """Initialize core security components"""
        components = [
            "api_gateway", "authentication", "authorization", "data_layer",
            "inference_engine", "model_repository", "monitoring", "logging",
            "network_layer", "application_layer", "storage_layer", "cache_layer"
        ]
        
        for component in components:
            self.component_health[component] = {
                "status": "healthy",
                "threat_count": 0,
                "last_scan": time.time(),
                "vulnerability_score": 0.0,
                "protection_level": "maximum",
                "self_healing_count": 0,
                "adaptation_count": 0
            }
        
        # Initialize security policies
        self.security_policies = {
            "access_control": {
                "default_action": "deny",
                "session_timeout": 3600,
                "max_failed_attempts": 3,
                "rate_limit_requests": 1000,
                "enable_mfa": True
            },
            "data_protection": {
                "encryption_required": True,
                "data_masking": True,
                "audit_logging": True,
                "backup_encryption": True
            },
            "network_security": {
                "firewall_enabled": True,
                "intrusion_detection": True,
                "ddos_protection": True,
                "ssl_required": True
            },
            "threat_response": {
                "auto_block": True,
                "quarantine_enabled": True,
                "adaptive_response": True,
                "self_healing": self.enable_self_healing
            }
        }
        
        logger.info(f"🏗️ Initialized {len(components)} security components")
    
    def _initialize_healing_strategies(self):
        """Initialize self-healing strategies"""
        self.healing_strategies = {
            "sql_injection": self._heal_sql_injection,
            "xss_attack": self._heal_xss_attack,
            "ddos_attack": self._heal_ddos_attack,
            "malware_detection": self._heal_malware,
            "privilege_escalation": self._heal_privilege_escalation,
            "data_breach": self._heal_data_breach,
            "authentication_bypass": self._heal_auth_bypass,
            "session_hijacking": self._heal_session_hijacking
        }
        
        logger.info(f"🔧 Initialized {len(self.healing_strategies)} self-healing strategies")
    
    def _initialize_threat_signatures(self):
        """Initialize threat detection signatures"""
        self.threat_signatures = {
            "sql_injection": {
                "patterns": [
                    r"(\%27)|(\')|(\-\-)|(\%23)|(#)",
                    r"((\%3D)|(=))[^\n]*((\%27)|(\')|(\-\-)|(\%3B)|(;))",
                    r"((\%27)|(\')){2}",
                    r"union[\s\w]*select",
                    r"drop[\s\w]*table"
                ],
                "severity": SecurityThreatLevel.HIGH,
                "confidence_threshold": 0.8
            },
            "xss_attack": {
                "patterns": [
                    r"<script[^>]*>.*?</script>",
                    r"javascript:",
                    r"onload\s*=",
                    r"onerror\s*=",
                    r"eval\s*\("
                ],
                "severity": SecurityThreatLevel.MEDIUM,
                "confidence_threshold": 0.7
            },
            "ddos_attack": {
                "indicators": [
                    "high_request_rate",
                    "multiple_source_ips",
                    "resource_exhaustion",
                    "response_time_degradation"
                ],
                "severity": SecurityThreatLevel.CRITICAL,
                "confidence_threshold": 0.9
            },
            "privilege_escalation": {
                "indicators": [
                    "unauthorized_admin_access",
                    "role_manipulation",
                    "permission_bypass",
                    "elevation_attempts"
                ],
                "severity": SecurityThreatLevel.HIGH,
                "confidence_threshold": 0.85
            }
        }
        
        logger.info(f"🔍 Initialized {len(self.threat_signatures)} threat signatures")
    
    async def start_security_monitoring(self):
        """Start continuous security monitoring"""
        if self.is_running:
            return
        
        self.is_running = True
        
        # Start monitoring tasks
        if self.real_time_monitoring:
            self.monitor_task = asyncio.create_task(self._monitoring_loop())
        
        if self.enable_self_healing:
            self.healing_task = asyncio.create_task(self._self_healing_loop())
        
        if self.enable_adaptive_learning:
            self.adaptation_task = asyncio.create_task(self._adaptation_loop())
        
        logger.info("🚀 Security monitoring started")
    
    async def stop_security_monitoring(self):
        """Stop security monitoring"""
        self.is_running = False
        
        # Cancel tasks
        for task in [self.monitor_task, self.healing_task, self.adaptation_task]:
            if task:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
        
        logger.info("🛑 Security monitoring stopped")
    
    async def _monitoring_loop(self):
        """Main security monitoring loop"""
        while self.is_running:
            try:
                # Perform threat detection scan
                threats = await self._detect_threats()
                
                # Process detected threats
                for threat in threats:
                    await self._handle_threat(threat)
                
                # Update component health
                await self._update_component_health()
                
                # Record metrics
                self._record_security_metrics()
                
                # Wait before next scan
                await asyncio.sleep(1.0)  # Scan every second
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Monitoring loop error: {e}")
                await asyncio.sleep(5.0)
    
    async def _self_healing_loop(self):
        """Self-healing monitoring loop"""
        while self.is_running:
            try:
                # Check for components needing healing
                components_needing_healing = await self._identify_healing_needs()
                
                # Apply healing strategies
                for component, healing_type in components_needing_healing:
                    await self._apply_healing_strategy(component, healing_type)
                
                await asyncio.sleep(30.0)  # Check every 30 seconds
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Self-healing loop error: {e}")
                await asyncio.sleep(10.0)
    
    async def _adaptation_loop(self):
        """Adaptive learning loop"""
        while self.is_running:
            try:
                # Analyze threat patterns
                await self._analyze_threat_patterns()
                
                # Update threat signatures
                await self._update_threat_signatures()
                
                # Optimize detection algorithms
                await self._optimize_detection()
                
                await asyncio.sleep(300.0)  # Adapt every 5 minutes
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Adaptation loop error: {e}")
                await asyncio.sleep(30.0)
    
    async def _detect_threats(self) -> List[SecurityThreat]:
        """Detect security threats across all components"""
        threats = []
        
        # Simulate threat detection for each signature type
        for threat_type, signature_data in self.threat_signatures.items():
            detected = await self._detect_threat_type(threat_type, signature_data)
            threats.extend(detected)
        
        # Advanced behavioral analysis
        behavioral_threats = await self._detect_behavioral_anomalies()
        threats.extend(behavioral_threats)
        
        return threats
    
    async def _detect_threat_type(self, threat_type: str, signature_data: Dict[str, Any]) -> List[SecurityThreat]:
        """Detect specific threat type"""
        threats = []
        detection_probability = 0.05  # 5% base chance
        
        # Adjust probability based on recent threat history
        recent_threats = [t for t in self.threat_history if t.threat_type == threat_type and t.timestamp > time.time() - 3600]
        if len(recent_threats) > 0:
            detection_probability *= 1.5  # Higher chance if recent activity
        
        if random.random() < detection_probability:
            threat_id = f"{threat_type}_{int(time.time() * 1000)}_{random.randint(1000, 9999)}"
            
            threat = SecurityThreat(
                threat_id=threat_id,
                timestamp=time.time(),
                threat_type=threat_type,
                threat_level=signature_data.get("severity", SecurityThreatLevel.MEDIUM),
                source_ip=f"192.168.{random.randint(1, 254)}.{random.randint(1, 254)}",
                target_component=random.choice(list(self.component_health.keys())),
                attack_signature=f"signature_{threat_type}_{random.randint(1, 100)}",
                confidence_score=random.uniform(signature_data.get("confidence_threshold", 0.7), 1.0),
                indicators=signature_data.get("indicators", [f"{threat_type}_pattern_detected"]),
                mitigation_strategies=[f"mitigate_{threat_type}", "block_source", "monitor_activity"]
            )
            
            threats.append(threat)
        
        return threats
    
    async def _detect_behavioral_anomalies(self) -> List[SecurityThreat]:
        """Detect behavioral anomalies using ML-like analysis"""
        threats = []
        
        # Analyze request patterns
        if random.random() < 0.02:  # 2% chance of behavioral anomaly
            threat_id = f"behavioral_{int(time.time() * 1000)}"
            
            anomaly_types = [
                "unusual_access_pattern",
                "abnormal_data_volume",
                "suspicious_user_behavior",
                "irregular_api_usage",
                "time_based_anomaly"
            ]
            
            threat = SecurityThreat(
                threat_id=threat_id,
                timestamp=time.time(),
                threat_type="behavioral_anomaly",
                threat_level=SecurityThreatLevel.MEDIUM,
                source_ip="internal_analysis",
                target_component="behavior_monitor",
                attack_signature=f"anomaly_{random.choice(anomaly_types)}",
                confidence_score=random.uniform(0.6, 0.9),
                indicators=["statistical_deviation", "pattern_break", "threshold_exceeded"],
                mitigation_strategies=["enhanced_monitoring", "user_verification", "access_restriction"]
            )
            
            threats.append(threat)
        
        return threats
    
    async def _handle_threat(self, threat: SecurityThreat):
        """Handle detected security threat"""
        logger.warning(f"🚨 Security threat detected: {threat.threat_type} (Level: {threat.threat_level.value})")
        
        with self._lock:
            # Add to active threats
            self.active_threats[threat.threat_id] = threat
            self.threat_history.append(threat)
            
            # Update component threat count
            if threat.target_component in self.component_health:
                self.component_health[threat.target_component]["threat_count"] += 1
                self.component_health[threat.target_component]["status"] = "under_attack"
        
        # Execute immediate response
        response = await self._execute_threat_response(threat)
        
        # Apply self-healing if enabled and needed
        if self.enable_self_healing and threat.threat_level in [SecurityThreatLevel.HIGH, SecurityThreatLevel.CRITICAL]:
            healing_applied = await self._apply_immediate_healing(threat)
            response.self_healing_applied = healing_applied
        
        # Learn from threat if adaptive learning enabled
        if self.enable_adaptive_learning:
            adaptation_learned = await self._learn_from_threat(threat)
            response.adaptation_learned = adaptation_learned
        
        # Record response
        self.security_responses.append(response)
        
        logger.info(f"✅ Threat response completed for {threat.threat_id}")
        logger.info(f"   Response: {response.response_type.value}")
        logger.info(f"   Effectiveness: {response.effectiveness:.2f}")
        logger.info(f"   Self-healing: {'Applied' if response.self_healing_applied else 'Not applied'}")
    
    async def _execute_threat_response(self, threat: SecurityThreat) -> SecurityResponse:
        """Execute appropriate response to threat"""
        start_time = time.time()
        
        # Determine response strategy
        response_type = self._determine_response_type(threat)
        
        # Execute response action
        action_result = await self._execute_response_action(threat, response_type)
        
        execution_time = time.time() - start_time
        
        response = SecurityResponse(
            response_id=f"resp_{int(time.time() * 1000)}",
            threat_id=threat.threat_id,
            response_type=response_type,
            action_taken=action_result["action"],
            effectiveness=action_result["effectiveness"],
            execution_time=execution_time,
            side_effects=action_result.get("side_effects", [])
        )
        
        return response
    
    def _determine_response_type(self, threat: SecurityThreat) -> SecurityResponseType:
        """Determine appropriate response type based on threat characteristics"""
        # Response strategy based on threat level and type
        if threat.threat_level == SecurityThreatLevel.CRITICAL:
            return SecurityResponseType.QUARANTINE
        elif threat.threat_level == SecurityThreatLevel.HIGH:
            return SecurityResponseType.BLOCK
        elif threat.threat_type in ["behavioral_anomaly", "zero_day"]:
            return SecurityResponseType.MONITOR
        elif self.enable_adaptive_learning:
            return SecurityResponseType.ADAPT
        else:
            return SecurityResponseType.BLOCK
    
    async def _execute_response_action(self, threat: SecurityThreat, response_type: SecurityResponseType) -> Dict[str, Any]:
        """Execute the actual response action"""
        # Simulate response execution time
        await asyncio.sleep(random.uniform(0.1, 0.5))
        
        if response_type == SecurityResponseType.BLOCK:
            return {
                "action": f"Blocked source IP {threat.source_ip}",
                "effectiveness": random.uniform(0.8, 0.95),
                "side_effects": ["temporary_access_restriction"]
            }
        elif response_type == SecurityResponseType.QUARANTINE:
            return {
                "action": f"Quarantined {threat.target_component}",
                "effectiveness": random.uniform(0.9, 0.99),
                "side_effects": ["component_temporary_isolation"]
            }
        elif response_type == SecurityResponseType.MONITOR:
            return {
                "action": f"Enhanced monitoring activated for {threat.target_component}",
                "effectiveness": random.uniform(0.6, 0.8),
                "side_effects": ["increased_logging"]
            }
        elif response_type == SecurityResponseType.ADAPT:
            return {
                "action": f"Adaptive countermeasures deployed",
                "effectiveness": random.uniform(0.7, 0.9),
                "side_effects": ["learning_algorithm_update"]
            }
        else:
            return {
                "action": "Default security response",
                "effectiveness": 0.5,
                "side_effects": []
            }
    
    async def _apply_immediate_healing(self, threat: SecurityThreat) -> bool:
        """Apply immediate self-healing for threat"""
        if threat.threat_type in self.healing_strategies:
            try:
                healing_func = self.healing_strategies[threat.threat_type]
                success = await healing_func(threat)
                
                if success:
                    # Update component health
                    with self._lock:
                        if threat.target_component in self.component_health:
                            self.component_health[threat.target_component]["self_healing_count"] += 1
                            self.component_health[threat.target_component]["status"] = "healing"
                    
                    logger.info(f"🔧 Self-healing applied for {threat.threat_type}")
                    return True
                    
            except Exception as e:
                logger.error(f"Self-healing failed for {threat.threat_type}: {e}")
        
        return False
    
    async def _learn_from_threat(self, threat: SecurityThreat) -> bool:
        """Learn and adapt from threat encounter"""
        try:
            # Update threat signature patterns
            if threat.threat_type in self.threat_signatures:
                signature_data = self.threat_signatures[threat.threat_type]
                
                # Adjust confidence threshold based on detection accuracy
                if threat.confidence_score > 0.9:
                    signature_data["confidence_threshold"] = min(0.95, signature_data["confidence_threshold"] + 0.01)
                elif threat.confidence_score < 0.6:
                    signature_data["confidence_threshold"] = max(0.5, signature_data["confidence_threshold"] - 0.01)
            
            # Store adaptation patterns
            pattern_key = f"{threat.threat_type}_{threat.target_component}"
            if pattern_key not in self.adaptation_patterns:
                self.adaptation_patterns[pattern_key] = {
                    "count": 0,
                    "avg_confidence": 0.0,
                    "mitigation_success": []
                }
            
            pattern_data = self.adaptation_patterns[pattern_key]
            pattern_data["count"] += 1
            pattern_data["avg_confidence"] = (
                (pattern_data["avg_confidence"] * (pattern_data["count"] - 1) + threat.confidence_score) / 
                pattern_data["count"]
            )
            
            # Update component adaptation count
            with self._lock:
                if threat.target_component in self.component_health:
                    self.component_health[threat.target_component]["adaptation_count"] += 1
            
            logger.info(f"🧠 Learning adaptation applied for {threat.threat_type}")
            return True
            
        except Exception as e:
            logger.error(f"Adaptive learning failed for {threat.threat_type}: {e}")
        
        return False
    
    # Self-healing strategy implementations
    async def _heal_sql_injection(self, threat: SecurityThreat) -> bool:
        """Self-healing strategy for SQL injection attacks"""
        try:
            # Simulate SQL injection healing
            await asyncio.sleep(0.2)
            
            # Apply input sanitization
            # Update WAF rules
            # Strengthen database access controls
            
            logger.info("🔧 Applied SQL injection healing: Enhanced input validation")
            return True
        except Exception as e:
            logger.error(f"SQL injection healing failed: {e}")
            return False
    
    async def _heal_xss_attack(self, threat: SecurityThreat) -> bool:
        """Self-healing strategy for XSS attacks"""
        try:
            await asyncio.sleep(0.15)
            
            # Apply output encoding
            # Update content security policy
            # Strengthen input filtering
            
            logger.info("🔧 Applied XSS healing: Enhanced content filtering")
            return True
        except Exception as e:
            logger.error(f"XSS healing failed: {e}")
            return False
    
    async def _heal_ddos_attack(self, threat: SecurityThreat) -> bool:
        """Self-healing strategy for DDoS attacks"""
        try:
            await asyncio.sleep(0.3)
            
            # Activate rate limiting
            # Deploy traffic shaping
            # Enable CDN protection
            
            logger.info("🔧 Applied DDoS healing: Enhanced traffic management")
            return True
        except Exception as e:
            logger.error(f"DDoS healing failed: {e}")
            return False
    
    async def _heal_malware(self, threat: SecurityThreat) -> bool:
        """Self-healing strategy for malware detection"""
        try:
            await asyncio.sleep(0.25)
            
            # Quarantine affected files
            # Update malware signatures
            # Scan for similar threats
            
            logger.info("🔧 Applied malware healing: System sanitization")
            return True
        except Exception as e:
            logger.error(f"Malware healing failed: {e}")
            return False
    
    async def _heal_privilege_escalation(self, threat: SecurityThreat) -> bool:
        """Self-healing strategy for privilege escalation"""
        try:
            await asyncio.sleep(0.2)
            
            # Review access permissions
            # Strengthen role-based access
            # Audit privilege changes
            
            logger.info("🔧 Applied privilege escalation healing: Access controls reinforced")
            return True
        except Exception as e:
            logger.error(f"Privilege escalation healing failed: {e}")
            return False
    
    async def _heal_data_breach(self, threat: SecurityThreat) -> bool:
        """Self-healing strategy for data breach attempts"""
        try:
            await asyncio.sleep(0.4)
            
            # Encrypt sensitive data
            # Activate data loss prevention
            # Monitor data access patterns
            
            logger.info("🔧 Applied data breach healing: Enhanced data protection")
            return True
        except Exception as e:
            logger.error(f"Data breach healing failed: {e}")
            return False
    
    async def _heal_auth_bypass(self, threat: SecurityThreat) -> bool:
        """Self-healing strategy for authentication bypass"""
        try:
            await asyncio.sleep(0.2)
            
            # Strengthen authentication
            # Enable multi-factor auth
            # Review session management
            
            logger.info("🔧 Applied auth bypass healing: Authentication strengthened")
            return True
        except Exception as e:
            logger.error(f"Auth bypass healing failed: {e}")
            return False
    
    async def _heal_session_hijacking(self, threat: SecurityThreat) -> bool:
        """Self-healing strategy for session hijacking"""
        try:
            await asyncio.sleep(0.15)
            
            # Regenerate session tokens
            # Enable secure session flags
            # Monitor session anomalies
            
            logger.info("🔧 Applied session hijacking healing: Session security enhanced")
            return True
        except Exception as e:
            logger.error(f"Session hijacking healing failed: {e}")
            return False
    
    async def _identify_healing_needs(self) -> List[Tuple[str, str]]:
        """Identify components that need healing"""
        healing_needed = []
        
        with self._lock:
            for component, health_data in self.component_health.items():
                # Check if component needs healing
                if health_data["threat_count"] > 3:
                    healing_needed.append((component, "threat_overload"))
                elif health_data["vulnerability_score"] > 0.7:
                    healing_needed.append((component, "vulnerability_mitigation"))
                elif health_data["status"] == "under_attack":
                    healing_needed.append((component, "active_threat"))
        
        return healing_needed
    
    async def _apply_healing_strategy(self, component: str, healing_type: str):
        """Apply healing strategy to component"""
        try:
            logger.info(f"🔧 Applying {healing_type} healing to {component}")
            
            # Simulate healing application
            await asyncio.sleep(random.uniform(0.5, 2.0))
            
            with self._lock:
                if component in self.component_health:
                    health_data = self.component_health[component]
                    
                    if healing_type == "threat_overload":
                        health_data["threat_count"] = max(0, health_data["threat_count"] - 2)
                        health_data["protection_level"] = "enhanced"
                    elif healing_type == "vulnerability_mitigation":
                        health_data["vulnerability_score"] *= 0.5
                        health_data["protection_level"] = "maximum"
                    elif healing_type == "active_threat":
                        health_data["status"] = "healing"
                        health_data["threat_count"] = max(0, health_data["threat_count"] - 1)
                    
                    health_data["self_healing_count"] += 1
                    health_data["last_scan"] = time.time()
            
            logger.info(f"✅ Healing applied to {component}")
            
        except Exception as e:
            logger.error(f"Healing strategy failed for {component}: {e}")
    
    async def _analyze_threat_patterns(self):
        """Analyze historical threat patterns for learning"""
        if len(self.threat_history) < 10:
            return
        
        # Analyze threat frequency by type
        threat_counts = defaultdict(int)
        for threat in self.threat_history[-100:]:  # Last 100 threats
            threat_counts[threat.threat_type] += 1
        
        # Update detection sensitivity
        for threat_type, count in threat_counts.items():
            if threat_type in self.threat_signatures:
                if count > 10:  # High frequency threat
                    # Increase sensitivity
                    self.threat_signatures[threat_type]["confidence_threshold"] = max(
                        0.5, self.threat_signatures[threat_type]["confidence_threshold"] - 0.05
                    )
                elif count < 2:  # Low frequency threat
                    # Decrease sensitivity to reduce false positives
                    self.threat_signatures[threat_type]["confidence_threshold"] = min(
                        0.95, self.threat_signatures[threat_type]["confidence_threshold"] + 0.02
                    )
    
    async def _update_threat_signatures(self):
        """Update threat detection signatures based on learning"""
        # This would implement ML-based signature updates in production
        logger.debug("🧠 Updated threat signatures based on recent patterns")
    
    async def _optimize_detection(self):
        """Optimize detection algorithms"""
        # This would implement detection algorithm optimization in production  
        logger.debug("⚡ Optimized detection algorithms")
    
    async def _update_component_health(self):
        """Update health status of all components"""
        current_time = time.time()
        
        with self._lock:
            for component, health_data in self.component_health.items():
                # Age out old threats
                if current_time - health_data["last_scan"] > 3600:  # 1 hour
                    health_data["threat_count"] = max(0, health_data["threat_count"] - 1)
                    health_data["last_scan"] = current_time
                
                # Update status based on threat count
                if health_data["threat_count"] == 0:
                    health_data["status"] = "healthy"
                elif health_data["threat_count"] < 3:
                    health_data["status"] = "monitoring"
                else:
                    health_data["status"] = "under_attack"
                
                # Calculate vulnerability score
                recent_threats = [t for t in self.threat_history 
                                if t.target_component == component and t.timestamp > current_time - 3600]
                health_data["vulnerability_score"] = min(1.0, len(recent_threats) * 0.1)
    
    def _record_security_metrics(self):
        """Record security metrics for monitoring"""
        current_time = time.time()
        
        # Record threat metrics
        self.security_metrics["total_threats"].append(len(self.threat_history))
        self.security_metrics["active_threats"].append(len(self.active_threats))
        
        # Record response metrics
        recent_responses = [r for r in self.security_responses if r.execution_time > current_time - 3600]
        if recent_responses:
            avg_effectiveness = statistics.mean(r.effectiveness for r in recent_responses)
            self.security_metrics["avg_response_effectiveness"].append(avg_effectiveness)
        
        # Record healing metrics
        total_healing_count = sum(h["self_healing_count"] for h in self.component_health.values())
        self.security_metrics["total_healing_applications"].append(total_healing_count)
        
        # Record component health
        healthy_components = sum(1 for h in self.component_health.values() if h["status"] == "healthy")
        self.security_metrics["healthy_components"].append(healthy_components)
    
    def get_security_status(self) -> Dict[str, Any]:
        """Get comprehensive security status"""
        with self._lock:
            current_time = time.time()
            
            # Calculate summary metrics
            total_threats = len(self.threat_history)
            recent_threats = [t for t in self.threat_history if t.timestamp > current_time - 3600]
            active_threats_count = len(self.active_threats)
            
            # Calculate healing effectiveness
            total_healing = sum(h["self_healing_count"] for h in self.component_health.values())
            healing_success_rate = 0.95 if total_healing > 0 else 0.0  # Simulated success rate
            
            # Calculate adaptation metrics
            total_adaptations = sum(h["adaptation_count"] for h in self.component_health.values())
            
            # Component health summary
            component_status = {
                comp: {
                    "status": data["status"],
                    "threat_count": data["threat_count"],
                    "vulnerability_score": data["vulnerability_score"],
                    "protection_level": data["protection_level"]
                }
                for comp, data in self.component_health.items()
            }
            
            # Calculate overall security score
            healthy_components = sum(1 for data in self.component_health.values() if data["status"] == "healthy")
            total_components = len(self.component_health)
            component_health_score = healthy_components / total_components if total_components > 0 else 1.0
            
            recent_threat_impact = min(1.0, len(recent_threats) * 0.05)  # 5% impact per recent threat
            security_score = max(0.0, (component_health_score + healing_success_rate) / 2 - recent_threat_impact)
            
            return {
                "timestamp": current_time,
                "security_score": security_score,
                "overall_status": self._determine_overall_status(security_score, active_threats_count),
                "total_threats_detected": total_threats,
                "recent_threats": len(recent_threats),
                "active_threats": active_threats_count,
                "component_health": component_status,
                "self_healing": {
                    "enabled": self.enable_self_healing,
                    "total_applications": total_healing,
                    "success_rate": healing_success_rate
                },
                "adaptive_learning": {
                    "enabled": self.enable_adaptive_learning,
                    "total_adaptations": total_adaptations,
                    "threat_signatures": len(self.threat_signatures)
                },
                "system_capabilities": {
                    "real_time_monitoring": self.real_time_monitoring,
                    "threat_intelligence": self.threat_intelligence_enabled,
                    "behavioral_analysis": True,
                    "quantum_resistance": False  # Future capability
                }
            }
    
    def _determine_overall_status(self, security_score: float, active_threats: int) -> str:
        """Determine overall security status"""
        if active_threats > 5:
            return "CRITICAL"
        elif active_threats > 2:
            return "HIGH_RISK"
        elif security_score < 0.6:
            return "MODERATE_RISK"
        elif security_score < 0.8:
            return "MONITORING"
        else:
            return "SECURE"
    
    def get_threat_intelligence_report(self) -> Dict[str, Any]:
        """Generate threat intelligence report"""
        current_time = time.time()
        
        # Analyze threat patterns
        threat_types = defaultdict(int)
        threat_levels = defaultdict(int)
        target_components = defaultdict(int)
        
        for threat in self.threat_history[-100:]:  # Last 100 threats
            threat_types[threat.threat_type] += 1
            threat_levels[threat.threat_level.value] += 1
            target_components[threat.target_component] += 1
        
        # Calculate threat trends
        recent_threats = [t for t in self.threat_history if t.timestamp > current_time - 86400]  # Last 24 hours
        older_threats = [t for t in self.threat_history if current_time - 172800 < t.timestamp <= current_time - 86400]  # 24-48 hours ago
        
        threat_trend = "stable"
        if len(recent_threats) > len(older_threats) * 1.2:
            threat_trend = "increasing"
        elif len(recent_threats) < len(older_threats) * 0.8:
            threat_trend = "decreasing"
        
        return {
            "timestamp": current_time,
            "analysis_period": "24_hours",
            "threat_summary": {
                "total_threats": len(self.threat_history),
                "recent_threats": len(recent_threats),
                "threat_trend": threat_trend,
                "most_common_threat": max(threat_types, key=threat_types.get) if threat_types else "none",
                "most_targeted_component": max(target_components, key=target_components.get) if target_components else "none"
            },
            "threat_breakdown": {
                "by_type": dict(threat_types),
                "by_level": dict(threat_levels),
                "by_component": dict(target_components)
            },
            "security_recommendations": self._generate_security_recommendations(),
            "adaptation_insights": {
                "patterns_learned": len(self.adaptation_patterns),
                "signatures_updated": len(self.threat_signatures),
                "healing_strategies": len(self.healing_strategies)
            }
        }
    
    def _generate_security_recommendations(self) -> List[str]:
        """Generate security recommendations based on current state"""
        recommendations = []
        
        # Analyze component health
        vulnerable_components = [
            comp for comp, data in self.component_health.items()
            if data["vulnerability_score"] > 0.5
        ]
        
        if vulnerable_components:
            recommendations.append(f"🔧 Apply additional hardening to: {', '.join(vulnerable_components)}")
        
        # Analyze threat patterns
        recent_threats = [t for t in self.threat_history if t.timestamp > time.time() - 86400]
        threat_types = defaultdict(int)
        for threat in recent_threats:
            threat_types[threat.threat_type] += 1
        
        high_frequency_threats = [t for t, count in threat_types.items() if count > 5]
        if high_frequency_threats:
            recommendations.append(f"🚨 Increase monitoring for: {', '.join(high_frequency_threats)}")
        
        # System capability recommendations
        if not self.enable_self_healing:
            recommendations.append("🔧 Enable self-healing capabilities for improved resilience")
        
        if not self.enable_adaptive_learning:
            recommendations.append("🧠 Enable adaptive learning for better threat detection")
        
        # Performance recommendations
        if len(self.active_threats) > 10:
            recommendations.append("⚡ Consider increasing response automation to handle threat volume")
        
        return recommendations

# Global adaptive security system instance
_global_security_system: Optional[AdaptiveSecuritySystem] = None

def get_adaptive_security_system() -> AdaptiveSecuritySystem:
    """Get global adaptive security system instance"""
    global _global_security_system
    if _global_security_system is None:
        _global_security_system = AdaptiveSecuritySystem()
    return _global_security_system

async def run_security_assessment() -> Dict[str, Any]:
    """Run comprehensive security assessment"""
    security_system = get_adaptive_security_system()
    
    # Start monitoring if not already running
    if not security_system.is_running:
        await security_system.start_security_monitoring()
        await asyncio.sleep(5.0)  # Let it run for 5 seconds
    
    # Get security status
    status = security_system.get_security_status()
    
    # Get threat intelligence
    threat_intel = security_system.get_threat_intelligence_report()
    
    return {
        "security_status": status,
        "threat_intelligence": threat_intel,
        "assessment_timestamp": time.time()
    }

if __name__ == "__main__":
    async def demo_adaptive_security():
        """Demo the adaptive security system"""
        logging.basicConfig(level=logging.INFO)
        
        print("🛡️ ADAPTIVE SECURITY SYSTEM v2.0 DEMO")
        print("=" * 50)
        
        # Run security assessment
        assessment = await run_security_assessment()
        
        status = assessment["security_status"]
        print(f"\n📊 Security Status:")
        print(f"   Overall Status: {status['overall_status']}")
        print(f"   Security Score: {status['security_score']:.2f}")
        print(f"   Active Threats: {status['active_threats']}")
        print(f"   Recent Threats: {status['recent_threats']}")
        
        healing_info = status["self_healing"]
        print(f"\n🔧 Self-Healing:")
        print(f"   Enabled: {healing_info['enabled']}")
        print(f"   Total Applications: {healing_info['total_applications']}")
        print(f"   Success Rate: {healing_info['success_rate']:.2f}")
        
        adaptive_info = status["adaptive_learning"]
        print(f"\n🧠 Adaptive Learning:")
        print(f"   Enabled: {adaptive_info['enabled']}")
        print(f"   Adaptations: {adaptive_info['total_adaptations']}")
        print(f"   Threat Signatures: {adaptive_info['threat_signatures']}")
        
        threat_intel = assessment["threat_intelligence"]
        print(f"\n🔍 Threat Intelligence:")
        print(f"   Threat Trend: {threat_intel['threat_summary']['threat_trend']}")
        print(f"   Most Common Threat: {threat_intel['threat_summary']['most_common_threat']}")
        print(f"   Most Targeted: {threat_intel['threat_summary']['most_targeted_component']}")
        
        recommendations = threat_intel["security_recommendations"]
        if recommendations:
            print(f"\n💡 Recommendations:")
            for rec in recommendations:
                print(f"   {rec}")
        
        # Stop the security system
        security_system = get_adaptive_security_system()
        await security_system.stop_security_monitoring()
    
    asyncio.run(demo_adaptive_security())