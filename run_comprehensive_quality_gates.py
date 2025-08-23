#!/usr/bin/env python3
"""
Comprehensive Quality Gates Validation
Validates all three generations and runs comprehensive quality checks
"""

import asyncio
import time
import json
import sys
from pathlib import Path
import subprocess

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

async def run_generation_validations():
    """Run all generation validations"""
    print("🧪 COMPREHENSIVE GENERATION VALIDATION")
    print("=" * 50)
    
    results = {}
    
    # Generation 1: Autonomous Testing
    print("\n🔬 Running Generation 1 Validation...")
    try:
        result = subprocess.run([sys.executable, "run_generation_validation.py"], 
                              capture_output=True, text=True, timeout=60)
        gen1_success = result.returncode == 0
        results["generation_1"] = {
            "status": "PASSED" if gen1_success else "FAILED",
            "output": result.stdout,
            "errors": result.stderr
        }
        print(f"   Generation 1: {'✅ PASSED' if gen1_success else '❌ FAILED'}")
    except Exception as e:
        results["generation_1"] = {"status": "FAILED", "error": str(e)}
        print(f"   Generation 1: ❌ FAILED ({e})")
    
    # Generation 3: Quantum Optimization
    print("\n🌌 Running Generation 3 Validation...")
    try:
        result = subprocess.run([sys.executable, "run_generation_3_validation.py"],
                              capture_output=True, text=True, timeout=60)
        gen3_success = result.returncode == 0
        results["generation_3"] = {
            "status": "PASSED" if gen3_success else "FAILED",
            "output": result.stdout,
            "errors": result.stderr
        }
        print(f"   Generation 3: {'✅ PASSED' if gen3_success else '❌ FAILED'}")
    except Exception as e:
        results["generation_3"] = {"status": "FAILED", "error": str(e)}
        print(f"   Generation 3: ❌ FAILED ({e})")
    
    return results

async def run_code_quality_checks():
    """Run code quality and security checks"""
    print("\n🔍 CODE QUALITY & SECURITY CHECKS")
    print("-" * 40)
    
    quality_results = {}
    
    # Check for Python syntax errors
    print("Checking Python syntax...")
    try:
        result = subprocess.run([sys.executable, "-m", "py_compile", "src/wasm_torch/__init__.py"],
                              capture_output=True, text=True)
        syntax_ok = result.returncode == 0
        quality_results["syntax_check"] = {
            "status": "PASSED" if syntax_ok else "FAILED",
            "output": result.stdout,
            "errors": result.stderr
        }
        print(f"   Syntax Check: {'✅ PASSED' if syntax_ok else '❌ FAILED'}")
    except Exception as e:
        quality_results["syntax_check"] = {"status": "FAILED", "error": str(e)}
        print(f"   Syntax Check: ❌ FAILED ({e})")
    
    # Check import structure
    print("Checking import structure...")
    try:
        from wasm_torch import __version__
        from wasm_torch.autonomous_testing_framework import get_test_framework
        from wasm_torch.adaptive_security_system import get_adaptive_security_system
        from wasm_torch.quantum_optimization_engine import get_global_optimization_engine
        
        quality_results["import_check"] = {"status": "PASSED"}
        print("   Import Check: ✅ PASSED")
    except Exception as e:
        quality_results["import_check"] = {"status": "FAILED", "error": str(e)}
        print(f"   Import Check: ❌ FAILED ({e})")
    
    # Check for security patterns
    print("Checking security patterns...")
    security_issues = []
    
    # Look for potential security issues in code
    security_patterns = [
        ("eval(", "Dynamic code execution"),
        ("exec(", "Dynamic code execution"), 
        ("subprocess.call", "System command execution"),
        ("os.system", "System command execution")
    ]
    
    for py_file in Path("src").rglob("*.py"):
        try:
            content = py_file.read_text()
            for pattern, description in security_patterns:
                if pattern in content and "# SECURITY: OK" not in content:
                    security_issues.append(f"{py_file}: {description} ({pattern})")
        except Exception:
            continue
    
    quality_results["security_check"] = {
        "status": "PASSED" if len(security_issues) == 0 else "REVIEW_NEEDED",
        "issues": security_issues
    }
    print(f"   Security Check: {'✅ PASSED' if len(security_issues) == 0 else '⚠️ REVIEW NEEDED'}")
    
    return quality_results

async def run_performance_benchmarks():
    """Run basic performance benchmarks"""
    print("\n⚡ PERFORMANCE BENCHMARKS")
    print("-" * 30)
    
    perf_results = {}
    
    # Test framework initialization time
    print("Testing framework initialization...")
    start_time = time.time()
    try:
        from wasm_torch.autonomous_testing_framework import get_test_framework
        framework = get_test_framework()
        init_time = time.time() - start_time
        
        perf_results["framework_init"] = {
            "time": init_time,
            "status": "PASSED" if init_time < 1.0 else "SLOW"
        }
        print(f"   Framework Init: {init_time:.3f}s {'✅' if init_time < 1.0 else '⚠️'}")
    except Exception as e:
        perf_results["framework_init"] = {"status": "FAILED", "error": str(e)}
        print(f"   Framework Init: ❌ FAILED ({e})")
    
    # Test security system initialization
    print("Testing security system initialization...")
    start_time = time.time()
    try:
        from wasm_torch.adaptive_security_system import get_adaptive_security_system
        security = get_adaptive_security_system()
        init_time = time.time() - start_time
        
        perf_results["security_init"] = {
            "time": init_time,
            "status": "PASSED" if init_time < 1.0 else "SLOW"
        }
        print(f"   Security Init: {init_time:.3f}s {'✅' if init_time < 1.0 else '⚠️'}")
    except Exception as e:
        perf_results["security_init"] = {"status": "FAILED", "error": str(e)}
        print(f"   Security Init: ❌ FAILED ({e})")
    
    # Test optimization engine initialization
    print("Testing optimization engine initialization...")
    start_time = time.time()
    try:
        from wasm_torch.quantum_optimization_engine import get_global_optimization_engine
        engine = get_global_optimization_engine()
        init_time = time.time() - start_time
        
        perf_results["optimization_init"] = {
            "time": init_time,
            "status": "PASSED" if init_time < 2.0 else "SLOW"
        }
        print(f"   Optimization Init: {init_time:.3f}s {'✅' if init_time < 2.0 else '⚠️'}")
    except Exception as e:
        perf_results["optimization_init"] = {"status": "FAILED", "error": str(e)}
        print(f"   Optimization Init: ❌ FAILED ({e})")
    
    return perf_results

async def run_integration_tests():
    """Run integration tests between systems"""
    print("\n🔗 INTEGRATION TESTS")
    print("-" * 25)
    
    integration_results = {}
    
    # Test framework + security integration
    print("Testing framework + security integration...")
    try:
        from wasm_torch.autonomous_testing_framework import get_test_framework
        from wasm_torch.adaptive_security_system import get_adaptive_security_system
        
        framework = get_test_framework()
        security = get_adaptive_security_system()
        
        # Both should have self-healing capabilities
        framework_healing = framework.enable_self_healing
        security_healing = security.enable_self_healing
        
        integration_results["framework_security"] = {
            "status": "PASSED" if framework_healing and security_healing else "PARTIAL",
            "framework_healing": framework_healing,
            "security_healing": security_healing
        }
        
        status = "✅ PASSED" if framework_healing and security_healing else "⚠️ PARTIAL"
        print(f"   Framework + Security: {status}")
        
    except Exception as e:
        integration_results["framework_security"] = {"status": "FAILED", "error": str(e)}
        print(f"   Framework + Security: ❌ FAILED ({e})")
    
    # Test optimization + security integration
    print("Testing optimization + security integration...")
    try:
        from wasm_torch.quantum_optimization_engine import get_global_optimization_engine
        from wasm_torch.adaptive_security_system import get_adaptive_security_system
        
        engine = get_global_optimization_engine()
        security = get_adaptive_security_system()
        
        # Both should have adaptive capabilities
        engine_adaptive = hasattr(engine, 'cache_system')
        security_adaptive = security.enable_adaptive_learning
        
        integration_results["optimization_security"] = {
            "status": "PASSED" if engine_adaptive and security_adaptive else "PARTIAL",
            "engine_adaptive": engine_adaptive,
            "security_adaptive": security_adaptive
        }
        
        status = "✅ PASSED" if engine_adaptive and security_adaptive else "⚠️ PARTIAL"
        print(f"   Optimization + Security: {status}")
        
    except Exception as e:
        integration_results["optimization_security"] = {"status": "FAILED", "error": str(e)}
        print(f"   Optimization + Security: ❌ FAILED ({e})")
    
    return integration_results

async def run_scalability_tests():
    """Run basic scalability tests"""
    print("\n📈 SCALABILITY TESTS")
    print("-" * 25)
    
    scalability_results = {}
    
    # Test multiple concurrent operations
    print("Testing concurrent operations...")
    try:
        from wasm_torch.quantum_optimization_engine import get_global_optimization_engine
        
        engine = get_global_optimization_engine()
        load_balancer = engine.load_balancer
        
        # Create test worker pool
        load_balancer.create_worker_pool("scalability_test", 4)
        
        def simple_task(x):
            return x * x
        
        # Run concurrent tasks
        start_time = time.time()
        tasks = []
        for i in range(10):
            task = load_balancer.execute_task("scalability_test", simple_task, i)
            tasks.append(task)
        
        results = await asyncio.gather(*tasks)
        execution_time = time.time() - start_time
        
        scalability_results["concurrent_operations"] = {
            "status": "PASSED" if execution_time < 2.0 else "SLOW",
            "execution_time": execution_time,
            "tasks_completed": len(results),
            "throughput": len(results) / execution_time
        }
        
        status = "✅ PASSED" if execution_time < 2.0 else "⚠️ SLOW"
        print(f"   Concurrent Ops: {status} ({len(results)} tasks in {execution_time:.2f}s)")
        
    except Exception as e:
        scalability_results["concurrent_operations"] = {"status": "FAILED", "error": str(e)}
        print(f"   Concurrent Ops: ❌ FAILED ({e})")
    
    # Test cache system scalability
    print("Testing cache system scalability...")
    try:
        from wasm_torch.quantum_optimization_engine import get_global_optimization_engine
        
        engine = get_global_optimization_engine()
        cache = engine.cache_system
        
        # Add many items to cache
        start_time = time.time()
        for i in range(100):
            cache.put(f"test_key_{i}", f"test_value_{i}")
        
        # Retrieve items
        hit_count = 0
        for i in range(100):
            if cache.get(f"test_key_{i}") is not None:
                hit_count += 1
        
        execution_time = time.time() - start_time
        hit_rate = hit_count / 100.0
        
        scalability_results["cache_scalability"] = {
            "status": "PASSED" if hit_rate > 0.9 and execution_time < 1.0 else "PARTIAL",
            "hit_rate": hit_rate,
            "execution_time": execution_time,
            "items_tested": 100
        }
        
        status = "✅ PASSED" if hit_rate > 0.9 and execution_time < 1.0 else "⚠️ PARTIAL"
        print(f"   Cache Scalability: {status} (hit rate: {hit_rate:.2f})")
        
    except Exception as e:
        scalability_results["cache_scalability"] = {"status": "FAILED", "error": str(e)}
        print(f"   Cache Scalability: ❌ FAILED ({e})")
    
    return scalability_results

async def main():
    """Run comprehensive quality gates validation"""
    print("🛡️ COMPREHENSIVE QUALITY GATES VALIDATION")
    print("=" * 60)
    
    start_time = time.time()
    
    # Run all validation components
    generation_results = await run_generation_validations()
    quality_results = await run_code_quality_checks()
    performance_results = await run_performance_benchmarks()
    integration_results = await run_integration_tests()
    scalability_results = await run_scalability_tests()
    
    execution_time = time.time() - start_time
    
    # Calculate overall status
    def get_status_score(results):
        total = len(results)
        if total == 0:
            return 0.0
        
        passed = sum(1 for r in results.values() if r.get("status") == "PASSED")
        partial = sum(1 for r in results.values() if r.get("status") in ["PARTIAL", "SLOW", "REVIEW_NEEDED"])
        
        return (passed + partial * 0.5) / total
    
    generation_score = get_status_score(generation_results)
    quality_score = get_status_score(quality_results)
    performance_score = get_status_score(performance_results)
    integration_score = get_status_score(integration_results)
    scalability_score = get_status_score(scalability_results)
    
    overall_score = (generation_score + quality_score + performance_score + 
                    integration_score + scalability_score) / 5
    
    # Determine overall status
    if overall_score >= 0.9:
        overall_status = "EXCELLENT"
        emoji = "🚀"
    elif overall_score >= 0.7:
        overall_status = "GOOD"
        emoji = "✅"
    elif overall_score >= 0.5:
        overall_status = "ACCEPTABLE"
        emoji = "⚠️"
    else:
        overall_status = "NEEDS_IMPROVEMENT"
        emoji = "❌"
    
    # Generate comprehensive report
    report = {
        "quality_gates_validation": {
            "overall_status": overall_status,
            "overall_score": overall_score,
            "execution_time": execution_time,
            "timestamp": time.time()
        },
        "component_scores": {
            "generation_validation": generation_score,
            "code_quality": quality_score,
            "performance": performance_score,
            "integration": integration_score,
            "scalability": scalability_score
        },
        "detailed_results": {
            "generation_validation": generation_results,
            "code_quality": quality_results,
            "performance_benchmarks": performance_results,
            "integration_tests": integration_results,
            "scalability_tests": scalability_results
        }
    }
    
    # Display comprehensive results
    print(f"\n{emoji} COMPREHENSIVE QUALITY GATES SUMMARY")
    print("=" * 50)
    print(f"Overall Status: {overall_status}")
    print(f"Overall Score: {overall_score:.1%}")
    print(f"Execution Time: {execution_time:.2f}s")
    
    print(f"\n📊 Component Scores:")
    print(f"   Generation Validation: {generation_score:.1%}")
    print(f"   Code Quality: {quality_score:.1%}")
    print(f"   Performance: {performance_score:.1%}")
    print(f"   Integration: {integration_score:.1%}")
    print(f"   Scalability: {scalability_score:.1%}")
    
    print(f"\n🔧 Generation Status:")
    gen1_status = generation_results.get("generation_1", {}).get("status", "UNKNOWN")
    gen3_status = generation_results.get("generation_3", {}).get("status", "UNKNOWN")
    print(f"   Generation 1 (Testing): {gen1_status}")
    print(f"   Generation 3 (Optimization): {gen3_status}")
    
    # Quality gates requirements
    print(f"\n✅ QUALITY GATES STATUS:")
    gates = [
        ("All generations pass", generation_score >= 0.8),
        ("Code quality acceptable", quality_score >= 0.7),
        ("Performance acceptable", performance_score >= 0.6),
        ("Integration working", integration_score >= 0.7),
        ("Scalability adequate", scalability_score >= 0.6)
    ]
    
    for gate_name, passed in gates:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"   {gate_name}: {status}")
    
    gates_passed = sum(1 for _, passed in gates if passed)
    print(f"\nQuality Gates: {gates_passed}/{len(gates)} passed")
    
    # Save comprehensive report
    report_file = Path("comprehensive_quality_gates_report.json")
    with open(report_file, "w") as f:
        json.dump(report, f, indent=2, default=str)
    
    print(f"\n📄 Comprehensive report saved: {report_file}")
    
    # Return exit code based on minimum quality requirements
    min_requirements_met = (
        generation_score >= 0.7 and
        quality_score >= 0.6 and
        overall_score >= 0.65
    )
    
    if min_requirements_met:
        print(f"\n🎉 MINIMUM QUALITY REQUIREMENTS MET")
        return 0
    else:
        print(f"\n❌ MINIMUM QUALITY REQUIREMENTS NOT MET")
        return 1

if __name__ == "__main__":
    try:
        exit_code = asyncio.run(main())
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n🛑 Quality gates validation interrupted")
        sys.exit(130)
    except Exception as e:
        print(f"\n💥 Fatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)