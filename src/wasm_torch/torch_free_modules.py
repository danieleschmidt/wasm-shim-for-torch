"""PyTorch-free modules that can run without PyTorch dependencies."""

import logging
import time
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass
from pathlib import Path


logger = logging.getLogger(__name__)


# Re-export the advanced modules that don't depend on PyTorch
try:
    from .advanced_error_recovery import (
        AdvancedErrorRecovery, 
        with_error_recovery, 
        get_global_recovery_system,
        ErrorSeverity,
        ErrorCategory
    )
    ERROR_RECOVERY_AVAILABLE = True
except Exception as e:
    logger.warning(f"Error recovery module not available: {e}")
    ERROR_RECOVERY_AVAILABLE = False


try:
    from .comprehensive_validation import (
        ComprehensiveValidator,
        ValidationLevel, 
        ValidationCategory,
        get_global_validator
    )
    VALIDATION_AVAILABLE = True
except Exception as e:
    logger.warning(f"Validation module not available: {e}")  
    VALIDATION_AVAILABLE = False


try:
    from .quantum_optimization_engine import (
        QuantumOptimizationEngine,
        OptimizationStrategy,
        PerformanceMetric,
        get_global_optimization_engine
    )
    OPTIMIZATION_AVAILABLE = True
except Exception as e:
    logger.warning(f"Optimization engine not available: {e}")
    OPTIMIZATION_AVAILABLE = False


try:
    from .autonomous_scaling_system import (
        AutonomousScalingSystem,
        ScalingStrategy,
        ResourceType,
        get_global_scaling_system
    )
    SCALING_AVAILABLE = True
except Exception as e:
    logger.warning(f"Scaling system not available: {e}")
    SCALING_AVAILABLE = False


@dataclass
class SystemStatus:
    """Status of available systems."""
    error_recovery: bool
    validation: bool
    optimization: bool
    scaling: bool
    
    @property
    def available_systems(self) -> List[str]:
        """Get list of available systems."""
        systems = []
        if self.error_recovery:
            systems.append("error_recovery")
        if self.validation:
            systems.append("validation") 
        if self.optimization:
            systems.append("optimization")
        if self.scaling:
            systems.append("scaling")
        return systems
    
    @property
    def system_count(self) -> int:
        """Get count of available systems."""
        return len(self.available_systems)


def get_system_status() -> SystemStatus:
    """Get status of all available systems."""
    return SystemStatus(
        error_recovery=ERROR_RECOVERY_AVAILABLE,
        validation=VALIDATION_AVAILABLE,
        optimization=OPTIMIZATION_AVAILABLE,
        scaling=SCALING_AVAILABLE
    )


class WASMTorchLite:
    """Lightweight WASM-Torch implementation without PyTorch dependencies."""
    
    def __init__(self):
        self.status = get_system_status()
        self.systems = {}
        self._initialize_available_systems()
        
        logger.info(f"WASM-Torch Lite initialized with {self.status.system_count} systems")
    
    def _initialize_available_systems(self):
        """Initialize all available systems."""
        if self.status.error_recovery:
            try:
                self.systems["error_recovery"] = AdvancedErrorRecovery()
            except Exception as e:
                logger.error(f"Failed to initialize error recovery: {e}")
        
        if self.status.validation:
            try:
                self.systems["validation"] = ComprehensiveValidator()
            except Exception as e:
                logger.error(f"Failed to initialize validation: {e}")
        
        if self.status.optimization:
            try:
                self.systems["optimization"] = QuantumOptimizationEngine()
            except Exception as e:
                logger.error(f"Failed to initialize optimization: {e}")
        
        if self.status.scaling:
            try:
                self.systems["scaling"] = AutonomousScalingSystem()
            except Exception as e:
                logger.error(f"Failed to initialize scaling: {e}")
    
    def get_system(self, system_name: str) -> Optional[Any]:
        """Get a specific system by name."""
        return self.systems.get(system_name)
    
    async def validate_input(self, data: Any, context: Optional[Dict[str, Any]] = None) -> bool:
        """Validate input data if validation system is available."""
        if "validation" not in self.systems:
            logger.warning("Validation system not available")
            return True  # Default to allowing input
        
        try:
            validator = self.systems["validation"]
            report = await validator.validate_comprehensive(
                data, 
                context or {}, 
                [ValidationCategory.SECURITY, ValidationCategory.USER_INPUT]
            )
            return report.overall_status != "fail"
        except Exception as e:
            logger.error(f"Validation failed: {e}")
            return False
    
    async def handle_error(self, error: Exception, operation: str, context: Optional[Dict[str, Any]] = None) -> Any:
        """Handle error if error recovery system is available."""
        if "error_recovery" not in self.systems:
            logger.error(f"Error recovery not available for {operation}: {error}")
            raise error
        
        try:
            recovery_system = self.systems["error_recovery"]
            return await recovery_system.handle_error(error, operation, context or {})
        except Exception as recovery_error:
            logger.error(f"Error recovery failed: {recovery_error}")
            raise error
    
    async def optimize_performance(self, objective_function: Callable, **kwargs) -> Optional[Any]:
        """Optimize performance if optimization system is available."""
        if "optimization" not in self.systems:
            logger.warning("Optimization system not available")
            return None
        
        try:
            optimizer = self.systems["optimization"]
            return await optimizer.optimize(objective_function, **kwargs)
        except Exception as e:
            logger.error(f"Performance optimization failed: {e}")
            return None
    
    async def scale_resources(self, **kwargs) -> bool:
        """Scale resources if scaling system is available."""
        if "scaling" not in self.systems:
            logger.warning("Scaling system not available")
            return False
        
        try:
            scaler = self.systems["scaling"]
            if not scaler.is_running:
                await scaler.start_autonomous_scaling()
            return True
        except Exception as e:
            logger.error(f"Resource scaling failed: {e}")
            return False
    
    def get_comprehensive_status(self) -> Dict[str, Any]:
        """Get comprehensive status of all systems."""
        status = {
            "system_status": {
                "error_recovery": self.status.error_recovery,
                "validation": self.status.validation,
                "optimization": self.status.optimization,
                "scaling": self.status.scaling
            },
            "available_systems": self.status.available_systems,
            "system_count": self.status.system_count,
            "initialized_systems": list(self.systems.keys())
        }
        
        # Add system-specific status
        for system_name, system in self.systems.items():
            try:
                if hasattr(system, 'get_statistics'):
                    status[f"{system_name}_stats"] = system.get_statistics()
                elif hasattr(system, 'get_optimization_summary'):
                    status[f"{system_name}_stats"] = system.get_optimization_summary()
                elif hasattr(system, 'get_scaling_statistics'):
                    status[f"{system_name}_stats"] = system.get_scaling_statistics()
            except Exception as e:
                logger.warning(f"Could not get stats for {system_name}: {e}")
        
        return status


# Global instance
_global_wasm_torch_lite = None


def get_wasm_torch_lite() -> WASMTorchLite:
    """Get global WASM-Torch Lite instance."""
    global _global_wasm_torch_lite
    if _global_wasm_torch_lite is None:
        _global_wasm_torch_lite = WASMTorchLite()
    return _global_wasm_torch_lite


def run_system_diagnostics() -> Dict[str, Any]:
    """Run comprehensive system diagnostics."""
    start_time = time.time()
    
    try:
        wasm_torch = get_wasm_torch_lite()
        status = wasm_torch.get_comprehensive_status()
        
        # Add diagnostic information
        diagnostics = {
            "timestamp": time.time(),
            "diagnostic_time": time.time() - start_time,
            "python_version": f"{__import__('sys').version_info.major}.{__import__('sys').version_info.minor}",
            "system_status": status,
            "module_availability": {
                "advanced_error_recovery": ERROR_RECOVERY_AVAILABLE,
                "comprehensive_validation": VALIDATION_AVAILABLE,
                "quantum_optimization_engine": OPTIMIZATION_AVAILABLE,
                "autonomous_scaling_system": SCALING_AVAILABLE,
            }
        }
        
        return diagnostics
        
    except Exception as e:
        return {
            "timestamp": time.time(),
            "diagnostic_time": time.time() - start_time,
            "error": str(e),
            "status": "failed"
        }


if __name__ == "__main__":
    # Run diagnostics when module is executed directly
    print("🔍 Running WASM-Torch System Diagnostics...")
    diagnostics = run_system_diagnostics()
    
    print(f"System Status: {diagnostics['system_status']['system_count']} systems available")
    print(f"Available Systems: {', '.join(diagnostics['system_status']['available_systems'])}")
    
    if diagnostics.get("error"):
        print(f"❌ Diagnostics failed: {diagnostics['error']}")
    else:
        print("✅ Diagnostics completed successfully")
        
    import json
    print(json.dumps(diagnostics, indent=2, default=str))