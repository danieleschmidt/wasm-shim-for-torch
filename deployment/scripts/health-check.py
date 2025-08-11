#!/usr/bin/env python3
"""
Health check script for WASM-Torch production deployment
Performs comprehensive health checks and returns appropriate exit codes
"""

import sys
import time
import json
import logging
import asyncio
from typing import Dict, Any, List, Tuple
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, '/app/src')

try:
    import requests
    import psutil
    from wasm_torch.validation import validate_system_resources
    from wasm_torch.security import check_resource_limits
    from wasm_torch.performance import get_performance_monitor
except ImportError as e:
    print(f"Import error during health check: {e}")
    sys.exit(1)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class HealthChecker:
    """Comprehensive health checker for WASM-Torch deployment."""
    
    def __init__(self, api_url: str = "http://localhost:8080"):
        self.api_url = api_url
        self.timeout = 10
        self.checks = []
        
    def add_check(self, name: str, status: bool, details: str = "") -> None:
        """Add a health check result."""
        self.checks.append({
            "name": name,
            "status": "pass" if status else "fail",
            "details": details,
            "timestamp": time.time()
        })
        
    def check_api_endpoints(self) -> bool:
        """Check if API endpoints are responding."""
        endpoints = [
            ("/health", "Health endpoint"),
            ("/ready", "Readiness endpoint"),
            ("/metrics", "Metrics endpoint")
        ]
        
        all_passed = True
        
        for endpoint, description in endpoints:
            try:
                response = requests.get(f"{self.api_url}{endpoint}", timeout=self.timeout)
                if response.status_code == 200:
                    self.add_check(f"api_{endpoint.strip('/')}", True, f"{description} responsive")
                else:
                    self.add_check(f"api_{endpoint.strip('/')}", False, f"{description} returned {response.status_code}")
                    all_passed = False
            except requests.exceptions.RequestException as e:
                self.add_check(f"api_{endpoint.strip('/')}", False, f"{description} error: {str(e)}")
                all_passed = False
        
        return all_passed
    
    def check_system_resources(self) -> bool:
        """Check system resource availability."""
        try:
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)
            cpu_ok = cpu_percent < 90
            self.add_check("cpu_usage", cpu_ok, f"CPU usage: {cpu_percent:.1f}%")
            
            # Memory usage
            memory = psutil.virtual_memory()
            memory_ok = memory.percent < 90
            self.add_check("memory_usage", memory_ok, f"Memory usage: {memory.percent:.1f}%")
            
            # Disk usage
            disk = psutil.disk_usage('/')
            disk_ok = (disk.used / disk.total) < 0.9
            self.add_check("disk_usage", disk_ok, f"Disk usage: {(disk.used / disk.total) * 100:.1f}%")
            
            # Load average
            load_avg = psutil.getloadavg()[0]  # 1-minute load average
            cpu_count = psutil.cpu_count()
            load_ok = load_avg < cpu_count * 2  # Allow up to 2x CPU count
            self.add_check("load_average", load_ok, f"Load average: {load_avg:.2f} (CPUs: {cpu_count})")
            
            return cpu_ok and memory_ok and disk_ok and load_ok
            
        except Exception as e:
            self.add_check("system_resources", False, f"Error checking resources: {str(e)}")
            return False
    
    def check_wasm_torch_components(self) -> bool:
        """Check WASM-Torch specific components."""
        try:
            # Check if main modules import correctly
            from wasm_torch import WASMRuntime, export_to_wasm
            self.add_check("wasm_torch_imports", True, "Core modules importable")
            
            # Check performance monitor
            perf_monitor = get_performance_monitor()
            stats = perf_monitor.get_comprehensive_stats()
            self.add_check("performance_monitor", True, f"Operations: {stats['operations']['count']}")
            
            # Check resource limits
            limits = check_resource_limits()
            limits_ok = limits.get("sufficient_disk_space", True) and limits.get("sufficient_memory", True)
            self.add_check("resource_limits", limits_ok, "Resource limits checked")
            
            # Check system validation
            system_resources = validate_system_resources()
            system_ok = system_resources["sufficient"]
            warnings = len(system_resources.get("warnings", []))
            self.add_check("system_validation", system_ok, f"System validation (warnings: {warnings})")
            
            return True
            
        except Exception as e:
            self.add_check("wasm_torch_components", False, f"Component check failed: {str(e)}")
            return False
    
    def check_file_permissions(self) -> bool:
        """Check critical file permissions."""
        critical_paths = [
            ("/app/cache", "Cache directory"),
            ("/app/logs", "Logs directory"),
            ("/tmp", "Temporary directory"),
        ]
        
        all_passed = True
        
        for path_str, description in critical_paths:
            path = Path(path_str)
            
            try:
                if path.exists():
                    # Check read permission
                    readable = path.is_dir() and os.access(path, os.R_OK)
                    # Check write permission  
                    writable = os.access(path, os.W_OK)
                    
                    if readable and writable:
                        self.add_check(f"permissions_{path.name}", True, f"{description} accessible")
                    else:
                        self.add_check(f"permissions_{path.name}", False, f"{description} permission denied")
                        all_passed = False
                else:
                    self.add_check(f"permissions_{path.name}", False, f"{description} does not exist")
                    all_passed = False
                    
            except Exception as e:
                self.add_check(f"permissions_{path.name}", False, f"{description} error: {str(e)}")
                all_passed = False
        
        return all_passed
    
    def check_configuration(self) -> bool:
        """Check configuration file and settings."""
        import os
        
        config_path = os.getenv("CONFIG_PATH", "/app/config/production.yaml")
        
        try:
            config_file = Path(config_path)
            
            if config_file.exists():
                # Check if config is readable
                content = config_file.read_text()
                config_ok = len(content) > 0
                self.add_check("configuration", config_ok, f"Config file readable ({len(content)} bytes)")
            else:
                self.add_check("configuration", False, f"Config file missing: {config_path}")
                return False
            
            # Check environment variables
            required_env = ["PYTHONPATH", "LOG_LEVEL", "WORKERS"]
            env_ok = True
            
            for var in required_env:
                if os.getenv(var):
                    self.add_check(f"env_{var.lower()}", True, f"{var}={os.getenv(var)}")
                else:
                    self.add_check(f"env_{var.lower()}", False, f"{var} not set")
                    env_ok = False
            
            return config_ok and env_ok
            
        except Exception as e:
            self.add_check("configuration", False, f"Config check error: {str(e)}")
            return False
    
    def run_all_checks(self) -> Tuple[bool, Dict[str, Any]]:
        """Run all health checks and return overall status."""
        logger.info("Starting comprehensive health check")
        
        # Run all checks
        checks_passed = []
        
        logger.info("Checking API endpoints...")
        checks_passed.append(self.check_api_endpoints())
        
        logger.info("Checking system resources...")
        checks_passed.append(self.check_system_resources())
        
        logger.info("Checking WASM-Torch components...")
        checks_passed.append(self.check_wasm_torch_components())
        
        logger.info("Checking file permissions...")
        checks_passed.append(self.check_file_permissions())
        
        logger.info("Checking configuration...")
        checks_passed.append(self.check_configuration())
        
        # Calculate overall health
        total_checks = len(self.checks)
        passed_checks = len([c for c in self.checks if c["status"] == "pass"])
        overall_healthy = all(checks_passed)
        
        health_report = {
            "status": "healthy" if overall_healthy else "unhealthy",
            "timestamp": time.time(),
            "checks_total": total_checks,
            "checks_passed": passed_checks,
            "checks_failed": total_checks - passed_checks,
            "success_rate": (passed_checks / total_checks) * 100 if total_checks > 0 else 0,
            "details": self.checks
        }
        
        logger.info(f"Health check complete: {health_report['status']} "
                   f"({passed_checks}/{total_checks} checks passed)")
        
        return overall_healthy, health_report


def main():
    """Main health check execution."""
    # Parse command line arguments
    api_url = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8080"
    
    # Create health checker
    health_checker = HealthChecker(api_url)
    
    try:
        # Run health checks
        is_healthy, report = health_checker.run_all_checks()
        
        # Output results
        if "--json" in sys.argv:
            print(json.dumps(report, indent=2))
        else:
            print(f"Health Status: {report['status'].upper()}")
            print(f"Checks: {report['checks_passed']}/{report['checks_total']} passed "
                  f"({report['success_rate']:.1f}%)")
            
            # Show failed checks
            failed_checks = [c for c in report['details'] if c['status'] == 'fail']
            if failed_checks:
                print("\nFailed Checks:")
                for check in failed_checks:
                    print(f"  ❌ {check['name']}: {check['details']}")
        
        # Return appropriate exit code
        sys.exit(0 if is_healthy else 1)
        
    except Exception as e:
        logger.error(f"Health check failed with exception: {e}")
        print(f"CRITICAL: Health check failed: {e}")
        sys.exit(2)


if __name__ == "__main__":
    # Ensure we have required imports
    import os
    main()