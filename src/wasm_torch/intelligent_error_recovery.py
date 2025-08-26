"""Intelligent error recovery system with adaptive resilience strategies."""

import asyncio
import time
import logging
import traceback
from typing import Dict, List, Optional, Union, Any, Callable, Type
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, deque
import threading
import weakref
from functools import wraps


logger = logging.getLogger(__name__)


class ErrorSeverity(Enum):
    """Error severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class RecoveryAction(Enum):
    """Possible recovery actions."""
    RETRY = "retry"
    FALLBACK = "fallback"
    RESTART = "restart"
    CIRCUIT_BREAK = "circuit_break"
    DEGRADE = "degrade"
    ALERT = "alert"


@dataclass
class ErrorContext:
    """Context information for error analysis and recovery."""
    error_type: str
    error_message: str
    stack_trace: str
    severity: ErrorSeverity
    component: str
    operation: str
    retry_count: int = 0
    first_occurrence: float = field(default_factory=time.time)
    last_occurrence: float = field(default_factory=time.time)
    frequency: int = 1
    context_data: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RecoveryPlan:
    """Recovery plan with ordered actions."""
    actions: List[RecoveryAction]
    retry_delays: List[float] = field(default_factory=lambda: [0.1, 0.5, 1.0, 2.0, 5.0])
    max_retries: int = 3
    fallback_strategy: Optional[str] = None
    escalation_threshold: int = 5
    recovery_timeout: float = 30.0
    success_criteria: Optional[Callable] = None


class ErrorPatternAnalyzer:
    """Analyzes error patterns to identify recovery strategies."""
    
    def __init__(self):
        self.error_history: Dict[str, List[ErrorContext]] = defaultdict(list)
        self.pattern_rules = {
            'network_error': RecoveryPlan([RecoveryAction.RETRY, RecoveryAction.FALLBACK]),
            'memory_error': RecoveryPlan([RecoveryAction.DEGRADE, RecoveryAction.RESTART]),
            'timeout_error': RecoveryPlan([RecoveryAction.RETRY, RecoveryAction.CIRCUIT_BREAK]),
            'validation_error': RecoveryPlan([RecoveryAction.FALLBACK, RecoveryAction.ALERT]),
            'resource_exhaustion': RecoveryPlan([RecoveryAction.DEGRADE, RecoveryAction.RESTART]),
            'corruption_error': RecoveryPlan([RecoveryAction.RESTART, RecoveryAction.ALERT])
        }
    
    def analyze_error(self, error: Exception, context: Dict[str, Any]) -> ErrorContext:
        """Analyze error and create context."""
        error_type = type(error).__name__
        error_message = str(error)
        stack_trace = traceback.format_exc()
        
        # Determine severity based on error type and context
        severity = self._determine_severity(error_type, error_message, context)
        
        error_context = ErrorContext(
            error_type=error_type,
            error_message=error_message,
            stack_trace=stack_trace,
            severity=severity,
            component=context.get('component', 'unknown'),
            operation=context.get('operation', 'unknown'),
            context_data=context
        )
        
        # Update history
        error_key = f"{error_type}:{context.get('component', '')}:{context.get('operation', '')}"
        if error_key in self.error_history:
            # Update existing error pattern
            last_error = self.error_history[error_key][-1]
            error_context.retry_count = last_error.retry_count
            error_context.first_occurrence = last_error.first_occurrence
            error_context.frequency = last_error.frequency + 1
        
        self.error_history[error_key].append(error_context)
        
        return error_context
    
    def _determine_severity(self, error_type: str, message: str, context: Dict[str, Any]) -> ErrorSeverity:
        """Determine error severity based on type and context."""
        critical_patterns = ['systemerror', 'memory', 'corrupt', 'fatal']
        high_patterns = ['timeout', 'connection', 'permission', 'security']
        medium_patterns = ['validation', 'format', 'parse', 'conversion']
        
        error_text = (error_type + message).lower()
        
        for pattern in critical_patterns:
            if pattern in error_text:
                return ErrorSeverity.CRITICAL
        
        for pattern in high_patterns:
            if pattern in error_text:
                return ErrorSeverity.HIGH
        
        for pattern in medium_patterns:
            if pattern in error_text:
                return ErrorSeverity.MEDIUM
        
        return ErrorSeverity.LOW
    
    def get_recovery_plan(self, error_context: ErrorContext) -> RecoveryPlan:
        """Get recovery plan based on error analysis."""
        # Check for specific error patterns
        error_text = (error_context.error_type + error_context.error_message).lower()
        
        for pattern, plan in self.pattern_rules.items():
            if pattern.replace('_', '') in error_text:
                return self._customize_plan(plan, error_context)
        
        # Default recovery plan based on severity
        if error_context.severity == ErrorSeverity.CRITICAL:
            return RecoveryPlan([RecoveryAction.RESTART, RecoveryAction.ALERT])
        elif error_context.severity == ErrorSeverity.HIGH:
            return RecoveryPlan([RecoveryAction.RETRY, RecoveryAction.FALLBACK, RecoveryAction.ALERT])
        elif error_context.severity == ErrorSeverity.MEDIUM:
            return RecoveryPlan([RecoveryAction.RETRY, RecoveryAction.FALLBACK])
        else:
            return RecoveryPlan([RecoveryAction.RETRY])
    
    def _customize_plan(self, base_plan: RecoveryPlan, context: ErrorContext) -> RecoveryPlan:
        """Customize recovery plan based on error context."""
        customized_plan = RecoveryPlan(
            actions=base_plan.actions.copy(),
            retry_delays=base_plan.retry_delays.copy(),
            max_retries=base_plan.max_retries,
            fallback_strategy=base_plan.fallback_strategy,
            escalation_threshold=base_plan.escalation_threshold,
            recovery_timeout=base_plan.recovery_timeout
        )
        
        # Adjust based on frequency
        if context.frequency > 3:
            customized_plan.max_retries = max(1, customized_plan.max_retries - 1)
            customized_plan.retry_delays = [d * 0.5 for d in customized_plan.retry_delays]
        
        return customized_plan


class CircuitBreakerManager:
    """Manages circuit breakers for different components."""
    
    def __init__(self):
        self.circuit_breakers: Dict[str, Dict[str, Any]] = {}
        self._lock = threading.Lock()
    
    def get_circuit_breaker(self, component: str) -> Dict[str, Any]:
        """Get or create circuit breaker for component."""
        with self._lock:
            if component not in self.circuit_breakers:
                self.circuit_breakers[component] = {
                    'state': 'closed',  # closed, open, half_open
                    'failure_count': 0,
                    'success_count': 0,
                    'last_failure_time': 0.0,
                    'failure_threshold': 5,
                    'success_threshold': 3,
                    'timeout': 60.0
                }
            return self.circuit_breakers[component]
    
    def record_success(self, component: str) -> None:
        """Record successful operation."""
        breaker = self.get_circuit_breaker(component)
        breaker['success_count'] += 1
        breaker['failure_count'] = 0
        
        if breaker['state'] == 'half_open' and breaker['success_count'] >= breaker['success_threshold']:
            breaker['state'] = 'closed'
            breaker['success_count'] = 0
            logger.info(f"Circuit breaker for {component} closed")
    
    def record_failure(self, component: str) -> None:
        """Record failed operation."""
        breaker = self.get_circuit_breaker(component)
        breaker['failure_count'] += 1
        breaker['last_failure_time'] = time.time()
        breaker['success_count'] = 0
        
        if breaker['state'] == 'closed' and breaker['failure_count'] >= breaker['failure_threshold']:
            breaker['state'] = 'open'
            logger.warning(f"Circuit breaker for {component} opened")
        elif breaker['state'] == 'half_open':
            breaker['state'] = 'open'
            logger.warning(f"Circuit breaker for {component} reopened")
    
    def can_execute(self, component: str) -> bool:
        """Check if operation can be executed."""
        breaker = self.get_circuit_breaker(component)
        
        if breaker['state'] == 'closed':
            return True
        elif breaker['state'] == 'open':
            if time.time() - breaker['last_failure_time'] > breaker['timeout']:
                breaker['state'] = 'half_open'
                logger.info(f"Circuit breaker for {component} half-opened")
                return True
            return False
        else:  # half_open
            return True


class IntelligentErrorRecovery:
    """Intelligent error recovery system with adaptive strategies."""
    
    def __init__(self, enable_circuit_breaker: bool = True, enable_alerting: bool = True):
        self.enable_circuit_breaker = enable_circuit_breaker
        self.enable_alerting = enable_alerting
        self.error_analyzer = ErrorPatternAnalyzer()
        self.circuit_manager = CircuitBreakerManager() if enable_circuit_breaker else None
        self.recovery_history: List[Dict[str, Any]] = []
        self.active_recoveries: Dict[str, Dict[str, Any]] = {}
        self._lock = threading.Lock()
        
        # Fallback strategies registry
        self.fallback_strategies: Dict[str, Callable] = {}
        
        logger.info("Intelligent Error Recovery system initialized")
    
    def register_fallback(self, component: str, fallback_func: Callable) -> None:
        """Register fallback function for a component."""
        self.fallback_strategies[component] = fallback_func
        logger.info(f"Registered fallback strategy for {component}")
    
    async def execute_with_recovery(
        self,
        operation: Callable,
        component: str,
        operation_name: str,
        context: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> Any:
        """Execute operation with intelligent error recovery."""
        context = context or {}
        context.update({'component': component, 'operation': operation_name})
        
        # Check circuit breaker
        if self.circuit_manager and not self.circuit_manager.can_execute(component):
            raise RuntimeError(f"Circuit breaker open for {component}")
        
        recovery_id = f"{component}:{operation_name}:{time.time()}"
        
        try:
            # Execute operation
            if asyncio.iscoroutinefunction(operation):
                result = await operation(**kwargs)
            else:
                result = operation(**kwargs)
            
            # Record success
            if self.circuit_manager:
                self.circuit_manager.record_success(component)
            
            return result
        
        except Exception as error:
            # Analyze error and create recovery plan
            error_context = self.error_analyzer.analyze_error(error, context)
            recovery_plan = self.error_analyzer.get_recovery_plan(error_context)
            
            # Record failure
            if self.circuit_manager:
                self.circuit_manager.record_failure(component)
            
            # Execute recovery
            return await self._execute_recovery(
                recovery_id, operation, error_context, recovery_plan, **kwargs
            )
    
    async def _execute_recovery(
        self,
        recovery_id: str,
        operation: Callable,
        error_context: ErrorContext,
        recovery_plan: RecoveryPlan,
        **kwargs
    ) -> Any:
        """Execute recovery plan."""
        with self._lock:
            self.active_recoveries[recovery_id] = {
                'error_context': error_context,
                'recovery_plan': recovery_plan,
                'start_time': time.time(),
                'current_action_index': 0
            }
        
        try:
            for action_index, action in enumerate(recovery_plan.actions):
                # Update recovery status
                with self._lock:
                    if recovery_id in self.active_recoveries:
                        self.active_recoveries[recovery_id]['current_action_index'] = action_index
                
                try:
                    result = await self._execute_action(
                        action, operation, error_context, recovery_plan, **kwargs
                    )
                    
                    # Recovery successful
                    self._record_recovery_success(recovery_id, action, error_context)
                    return result
                
                except Exception as recovery_error:
                    logger.warning(f"Recovery action {action} failed: {recovery_error}")
                    
                    # If this is the last action, re-raise the original error
                    if action_index == len(recovery_plan.actions) - 1:
                        self._record_recovery_failure(recovery_id, error_context)
                        raise error_context.error_message
        
        finally:
            with self._lock:
                self.active_recoveries.pop(recovery_id, None)
    
    async def _execute_action(
        self,
        action: RecoveryAction,
        operation: Callable,
        error_context: ErrorContext,
        recovery_plan: RecoveryPlan,
        **kwargs
    ) -> Any:
        """Execute specific recovery action."""
        if action == RecoveryAction.RETRY:
            return await self._retry_operation(operation, error_context, recovery_plan, **kwargs)
        
        elif action == RecoveryAction.FALLBACK:
            return await self._execute_fallback(error_context, **kwargs)
        
        elif action == RecoveryAction.DEGRADE:
            return await self._execute_degraded_operation(operation, error_context, **kwargs)
        
        elif action == RecoveryAction.CIRCUIT_BREAK:
            if self.circuit_manager:
                self.circuit_manager.record_failure(error_context.component)
            raise RuntimeError("Circuit breaker activated")
        
        elif action == RecoveryAction.RESTART:
            return await self._restart_component(error_context.component, operation, **kwargs)
        
        elif action == RecoveryAction.ALERT:
            await self._send_alert(error_context)
            raise RuntimeError("Alert sent, manual intervention required")
        
        else:
            raise ValueError(f"Unknown recovery action: {action}")
    
    async def _retry_operation(
        self, operation: Callable, error_context: ErrorContext, 
        recovery_plan: RecoveryPlan, **kwargs
    ) -> Any:
        """Retry operation with exponential backoff."""
        for retry_count in range(recovery_plan.max_retries):
            if retry_count < len(recovery_plan.retry_delays):
                delay = recovery_plan.retry_delays[retry_count]
            else:
                delay = recovery_plan.retry_delays[-1] * (2 ** (retry_count - len(recovery_plan.retry_delays) + 1))
            
            await asyncio.sleep(delay)
            
            try:
                if asyncio.iscoroutinefunction(operation):
                    return await operation(**kwargs)
                else:
                    return operation(**kwargs)
            
            except Exception as retry_error:
                logger.warning(f"Retry {retry_count + 1} failed: {retry_error}")
                error_context.retry_count += 1
                
                if retry_count == recovery_plan.max_retries - 1:
                    raise retry_error
    
    async def _execute_fallback(self, error_context: ErrorContext, **kwargs) -> Any:
        """Execute fallback strategy."""
        component = error_context.component
        if component in self.fallback_strategies:
            fallback_func = self.fallback_strategies[component]
            
            if asyncio.iscoroutinefunction(fallback_func):
                return await fallback_func(**kwargs)
            else:
                return fallback_func(**kwargs)
        
        # Default fallback: return empty/safe result
        return None
    
    async def _execute_degraded_operation(self, operation: Callable, error_context: ErrorContext, **kwargs) -> Any:
        """Execute operation in degraded mode."""
        # Simplify kwargs or reduce complexity
        degraded_kwargs = {k: v for k, v in kwargs.items() if k in ['input_data', 'model_id']}
        
        if asyncio.iscoroutinefunction(operation):
            return await operation(**degraded_kwargs)
        else:
            return operation(**degraded_kwargs)
    
    async def _restart_component(self, component: str, operation: Callable, **kwargs) -> Any:
        """Restart component (mock implementation)."""
        logger.info(f"Restarting component: {component}")
        
        # Simulate component restart
        await asyncio.sleep(1.0)
        
        # Try operation again after restart
        if asyncio.iscoroutinefunction(operation):
            return await operation(**kwargs)
        else:
            return operation(**kwargs)
    
    async def _send_alert(self, error_context: ErrorContext) -> None:
        """Send alert about critical error."""
        if self.enable_alerting:
            alert_message = (
                f"CRITICAL ERROR in {error_context.component}\n"
                f"Operation: {error_context.operation}\n"
                f"Error: {error_context.error_type}: {error_context.error_message}\n"
                f"Frequency: {error_context.frequency} occurrences"
            )
            logger.critical(alert_message)
            # In production, this would integrate with alerting systems
    
    def _record_recovery_success(self, recovery_id: str, action: RecoveryAction, error_context: ErrorContext) -> None:
        """Record successful recovery."""
        self.recovery_history.append({
            'recovery_id': recovery_id,
            'error_context': error_context,
            'successful_action': action,
            'success': True,
            'timestamp': time.time()
        })
        
        logger.info(f"Recovery successful for {error_context.component} using {action}")
    
    def _record_recovery_failure(self, recovery_id: str, error_context: ErrorContext) -> None:
        """Record failed recovery."""
        self.recovery_history.append({
            'recovery_id': recovery_id,
            'error_context': error_context,
            'successful_action': None,
            'success': False,
            'timestamp': time.time()
        })
        
        logger.error(f"Recovery failed for {error_context.component}")
    
    def get_recovery_statistics(self) -> Dict[str, Any]:
        """Get recovery system statistics."""
        if not self.recovery_history:
            return {'message': 'No recovery history available'}
        
        successful_recoveries = [r for r in self.recovery_history if r['success']]
        
        stats = {
            'total_recoveries': len(self.recovery_history),
            'successful_recoveries': len(successful_recoveries),
            'success_rate': len(successful_recoveries) / len(self.recovery_history),
            'active_recoveries': len(self.active_recoveries),
            'most_common_errors': {},
            'most_successful_actions': {}
        }
        
        # Analyze error patterns
        error_types = [r['error_context'].error_type for r in self.recovery_history]
        for error_type in set(error_types):
            stats['most_common_errors'][error_type] = error_types.count(error_type)
        
        # Analyze successful actions
        successful_actions = [r['successful_action'] for r in successful_recoveries if r['successful_action']]
        for action in set(successful_actions):
            stats['most_successful_actions'][str(action)] = successful_actions.count(action)
        
        return stats
    
    def get_system_health(self) -> Dict[str, Any]:
        """Get overall system health based on error patterns."""
        health = {'status': 'healthy', 'issues': [], 'recommendations': []}
        
        if self.circuit_manager:
            for component, breaker in self.circuit_manager.circuit_breakers.items():
                if breaker['state'] == 'open':
                    health['status'] = 'degraded'
                    health['issues'].append(f"Circuit breaker open for {component}")
                    health['recommendations'].append(f"Investigate {component} failures")
        
        # Check recent error frequency
        recent_errors = [r for r in self.recovery_history 
                        if time.time() - r['timestamp'] < 3600]  # Last hour
        
        if len(recent_errors) > 10:
            health['status'] = 'degraded' if health['status'] == 'healthy' else 'critical'
            health['issues'].append("High error rate in last hour")
            health['recommendations'].append("Review system logs and error patterns")
        
        return health


def resilient_operation(component: str, operation_name: str, recovery_system: IntelligentErrorRecovery):
    """Decorator for making operations resilient."""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            return await recovery_system.execute_with_recovery(
                func, component, operation_name, context=kwargs
            )
        return wrapper
    return decorator
