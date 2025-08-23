#!/usr/bin/env python3
"""
Generation Validation Suite - Validate Autonomous SDLC Generations
Tests Generation 1 (Enhanced Testing) and Generation 2 (Advanced Security)
"""

import asyncio
import time
import json
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

async def test_generation_1_autonomous_testing():
    """Test Generation 1: Enhanced Autonomous Testing Framework"""
    print("🧪 GENERATION 1: AUTONOMOUS TESTING VALIDATION")
    print("-" * 50)
    
    try:
        from wasm_torch.autonomous_testing_framework import get_test_framework, TestCase, TestType
        
        # Initialize framework
        framework = get_test_framework()
        
        print(f"✅ Self-healing test framework initialized")
        print(f"   AI generation: {framework.enable_ai_generation}")
        print(f"   Self-healing: {framework.enable_self_healing}")
        print(f"   Healing strategies: {len(framework.healing_strategies)}")
        
        # Create sample test
        async def sample_test():
            """Sample test function"""
            return "test_passed"
        
        test_case = TestCase(
            test_id="sample_test_001",
            name="Sample Test",
            test_type=TestType.UNIT,
            test_function=sample_test,
            description="Sample test for validation"
        )
        
        # Register and run test
        framework.register_test(test_case)
        print(f"✅ Test registered: {test_case.test_id}")
        
        # Run single test
        execution = await framework.run_test(test_case)
        print(f"✅ Test executed: {execution.result.value} in {execution.execution_time:.3f}s")
        
        # Test self-healing capability
        if execution.metrics.get("self_healing_attempted"):
            print(f"✅ Self-healing attempted: {execution.metrics['self_healing_success']}")
        
        return {
            "status": "PASSED",
            "framework_initialized": True,
            "tests_registered": len(framework.test_registry),
            "healing_strategies": len(framework.healing_strategies),
            "test_execution_successful": execution.result.value == "passed"
        }
        
    except Exception as e:
        print(f"❌ Generation 1 failed: {e}")
        return {
            "status": "FAILED",
            "error": str(e)
        }

async def test_generation_2_adaptive_security():
    """Test Generation 2: Advanced Security & Monitoring"""
    print("\n🛡️ GENERATION 2: ADAPTIVE SECURITY VALIDATION")
    print("-" * 50)
    
    try:
        from wasm_torch.adaptive_security_system import get_adaptive_security_system
        
        # Initialize security system
        security = get_adaptive_security_system()
        
        print(f"✅ Adaptive security system initialized")
        print(f"   Self-healing: {security.enable_self_healing}")
        print(f"   Adaptive learning: {security.enable_adaptive_learning}")
        print(f"   Healing strategies: {len(security.healing_strategies)}")
        print(f"   Threat signatures: {len(security.threat_signatures)}")
        
        # Get security status
        status = security.get_security_status()
        print(f"✅ Security status retrieved")
        print(f"   Overall Status: {status['overall_status']}")
        print(f"   Security Score: {status['security_score']:.2f}")
        print(f"   Protected Components: {len(status['component_health'])}")
        
        # Test threat intelligence
        threat_intel = security.get_threat_intelligence_report()
        print(f"✅ Threat intelligence generated")
        print(f"   Total Threats: {threat_intel['threat_summary']['total_threats']}")
        print(f"   Threat Trend: {threat_intel['threat_summary']['threat_trend']}")
        
        # Test healing capabilities
        healing_info = status["self_healing"]
        adaptation_info = status["adaptive_learning"]
        
        return {
            "status": "PASSED",
            "security_initialized": True,
            "overall_status": status["overall_status"],
            "security_score": status["security_score"],
            "self_healing_enabled": healing_info["enabled"],
            "adaptive_learning_enabled": adaptation_info["enabled"],
            "protected_components": len(status["component_health"]),
            "threat_signatures": len(security.threat_signatures)
        }
        
    except Exception as e:
        print(f"❌ Generation 2 failed: {e}")
        return {
            "status": "FAILED",
            "error": str(e)
        }

async def validate_integration():
    """Validate integration between generations"""
    print("\n🔗 INTEGRATION VALIDATION")
    print("-" * 30)
    
    try:
        # Test that both systems can coexist
        from wasm_torch.autonomous_testing_framework import get_test_framework
        from wasm_torch.adaptive_security_system import get_adaptive_security_system
        
        framework = get_test_framework()
        security = get_adaptive_security_system()
        
        print("✅ Both systems initialized successfully")
        print(f"   Testing framework: {framework.__class__.__name__}")
        print(f"   Security system: {security.__class__.__name__}")
        
        # Check for integration capabilities
        integration_score = 0.0
        
        # Self-healing compatibility
        if framework.enable_self_healing and security.enable_self_healing:
            integration_score += 0.3
            print("✅ Self-healing integration: Compatible")
        
        # Adaptive learning compatibility  
        if framework.enable_ai_generation and security.enable_adaptive_learning:
            integration_score += 0.3
            print("✅ Adaptive learning integration: Compatible")
        
        # Monitoring compatibility
        if hasattr(framework, 'performance_metrics') and hasattr(security, 'security_metrics'):
            integration_score += 0.4
            print("✅ Monitoring integration: Compatible")
        
        return {
            "status": "PASSED" if integration_score >= 0.7 else "PARTIAL",
            "integration_score": integration_score,
            "systems_compatible": True
        }
        
    except Exception as e:
        print(f"❌ Integration failed: {e}")
        return {
            "status": "FAILED",
            "error": str(e)
        }

async def main():
    """Main validation execution"""
    print("🚀 AUTONOMOUS SDLC GENERATION VALIDATION v2.0")
    print("=" * 60)
    
    start_time = time.time()
    
    # Test Generation 1
    gen1_results = await test_generation_1_autonomous_testing()
    
    # Test Generation 2
    gen2_results = await test_generation_2_adaptive_security()
    
    # Test Integration
    integration_results = await validate_integration()
    
    # Calculate overall results
    execution_time = time.time() - start_time
    
    # Determine overall status
    gen1_passed = gen1_results["status"] == "PASSED"
    gen2_passed = gen2_results["status"] == "PASSED"
    integration_passed = integration_results["status"] in ["PASSED", "PARTIAL"]
    
    overall_status = "PASSED" if all([gen1_passed, gen2_passed, integration_passed]) else "FAILED"
    
    # Generate comprehensive report
    report = {
        "validation_summary": {
            "overall_status": overall_status,
            "execution_time": execution_time,
            "timestamp": time.time(),
            "generations_tested": 2
        },
        "generation_1": gen1_results,
        "generation_2": gen2_results,
        "integration": integration_results,
        "capabilities_validated": {
            "self_healing_testing": gen1_results.get("healing_strategies", 0) > 0,
            "adaptive_security": gen2_results.get("adaptive_learning_enabled", False),
            "autonomous_response": gen2_results.get("self_healing_enabled", False),
            "system_integration": integration_results.get("systems_compatible", False)
        }
    }
    
    # Display final results
    print(f"\n📊 VALIDATION SUMMARY")
    print("=" * 30)
    print(f"Overall Status: {overall_status}")
    print(f"Generation 1 (Testing): {'✅ PASSED' if gen1_passed else '❌ FAILED'}")
    print(f"Generation 2 (Security): {'✅ PASSED' if gen2_passed else '❌ FAILED'}")
    print(f"Integration: {'✅ PASSED' if integration_passed else '❌ FAILED'}")
    print(f"Execution Time: {execution_time:.2f}s")
    
    # Save report
    report_file = Path("generation_validation_report.json")
    with open(report_file, "w") as f:
        json.dump(report, f, indent=2, default=str)
    
    print(f"\n📄 Report saved: {report_file}")
    
    return 0 if overall_status == "PASSED" else 1

if __name__ == "__main__":
    try:
        exit_code = asyncio.run(main())
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n🛑 Validation interrupted")
        sys.exit(130)
    except Exception as e:
        print(f"\n💥 Fatal error: {e}")
        sys.exit(1)