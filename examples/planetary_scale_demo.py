#!/usr/bin/env python3
"""
Planetary Scale Orchestration Demo
Demonstrates hyperdimensional optimization and planetary-scale deployment capabilities.
"""

import asyncio
import time
import logging
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from wasm_torch.planetary_scale_orchestrator import (
    PlanetaryScaleOrchestrator,
    ScaleTarget,
    PerformanceClass,
    ResourceType,
    WorkloadProfile,
    OptimizationTarget
)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


async def demonstrate_planetary_scaling():
    """Demonstrate planetary-scale orchestration capabilities."""
    logger.info("🌍 Starting Planetary Scale Orchestration Demo")
    
    scale_targets = [
        ScaleTarget.EDGE_DEVICES,
        ScaleTarget.REGIONAL_CLUSTERS,
        ScaleTarget.GLOBAL_DISTRIBUTION,
        ScaleTarget.PLANETARY_MESH,
        ScaleTarget.QUANTUM_GRID,
        ScaleTarget.HYPERDIMENSIONAL
    ]
    
    results = {}
    
    for scale_target in scale_targets:
        logger.info(f"🚀 Testing {scale_target.value} deployment")
        
        # Initialize orchestrator for this scale target
        orchestrator = PlanetaryScaleOrchestrator(scale_target)
        
        start_time = time.time()
        
        try:
            # Initialize the orchestrator
            await orchestrator.initialize()
            
            # Create test workloads for this scale
            workloads = await create_test_workloads(scale_target)
            
            deployment_results = []
            
            # Deploy each workload
            for workload in workloads:
                optimization_target = OptimizationTarget(
                    quantum_preference=(scale_target in [ScaleTarget.QUANTUM_GRID, ScaleTarget.HYPERDIMENSIONAL]),
                    edge_preference=(scale_target == ScaleTarget.EDGE_DEVICES)
                )
                
                result = await orchestrator.deploy_workload(workload, optimization_target)
                deployment_results.append(result)
                
                logger.info(f"✅ Deployed {workload.workload_id}: "
                           f"fitness={result['fitness_score']:.4f}, "
                           f"nodes={len(result['assigned_nodes'])}, "
                           f"cost=${result['cost_estimate']:.2f}/hr")
                
            # Get orchestration status
            status = await orchestrator.get_orchestration_status()
            
            # Let the system run for a bit to generate metrics
            await asyncio.sleep(2)
            
            # Get final status
            final_status = await orchestrator.get_orchestration_status()
            
            orchestration_time = time.time() - start_time
            
            results[scale_target.value] = {
                "orchestration_time": orchestration_time,
                "deployment_results": deployment_results,
                "initial_status": status,
                "final_status": final_status,
                "workload_count": len(workloads),
                "success": True
            }
            
            logger.info(f"📊 {scale_target.value} Results:")
            logger.info(f"  Orchestration Time: {orchestration_time:.2f}s")
            logger.info(f"  Workloads Deployed: {len(workloads)}")
            logger.info(f"  Total Nodes: {final_status['metrics']['total_nodes']}")
            logger.info(f"  Average Latency: {final_status['metrics']['average_latency_ms']:.2f}ms")
            logger.info(f"  Availability: {final_status['metrics']['availability_score']:.1f}%")
            logger.info(f"  Quantum Ratio: {final_status['metrics']['quantum_acceleration_ratio']:.1%}")
            logger.info(f"  Edge Ratio: {final_status['metrics']['edge_deployment_ratio']:.1%}")
            
        except Exception as e:
            logger.error(f"❌ {scale_target.value} failed: {e}")
            results[scale_target.value] = {
                "error": str(e),
                "success": False
            }
            
        finally:
            # Cleanup
            await orchestrator.shutdown()
            
        logger.info("=" * 60)
        
    return results


async def create_test_workloads(scale_target: ScaleTarget) -> list:
    """Create appropriate test workloads for the scale target."""
    workloads = []
    
    if scale_target == ScaleTarget.EDGE_DEVICES:
        # Edge workloads: low-latency, real-time
        workloads = [
            WorkloadProfile(
                workload_id="edge_realtime_1",
                performance_requirements=PerformanceClass.REALTIME,
                resource_requirements={
                    ResourceType.CPU_CORES: 4.0,
                    ResourceType.MEMORY_GB: 8.0
                },
                latency_budget_ms=1.0,
                edge_deployment=True,
                geographic_preferences=["US", "Europe"]
            ),
            WorkloadProfile(
                workload_id="edge_interactive_1",
                performance_requirements=PerformanceClass.INTERACTIVE,
                resource_requirements={
                    ResourceType.CPU_CORES: 2.0,
                    ResourceType.MEMORY_GB: 4.0
                },
                latency_budget_ms=10.0,
                edge_deployment=True
            )
        ]
        
    elif scale_target == ScaleTarget.REGIONAL_CLUSTERS:
        # Regional workloads: batch processing, higher resource requirements
        workloads = [
            WorkloadProfile(
                workload_id="regional_batch_1",
                performance_requirements=PerformanceClass.BATCH,
                resource_requirements={
                    ResourceType.CPU_CORES: 32.0,
                    ResourceType.MEMORY_GB: 128.0,
                    ResourceType.STORAGE_TB: 5.0
                },
                latency_budget_ms=1000.0,
                geographic_preferences=["US East", "Europe"]
            ),
            WorkloadProfile(
                workload_id="regional_ml_training",
                performance_requirements=PerformanceClass.MASSIVE,
                resource_requirements={
                    ResourceType.CPU_CORES: 64.0,
                    ResourceType.MEMORY_GB: 256.0,
                    ResourceType.GPU_UNITS: 4.0
                },
                latency_budget_ms=5000.0
            )
        ]
        
    elif scale_target == ScaleTarget.GLOBAL_DISTRIBUTION:
        # Global workloads: mixed requirements
        workloads = [
            WorkloadProfile(
                workload_id="global_cdn_1",
                performance_requirements=PerformanceClass.INTERACTIVE,
                resource_requirements={
                    ResourceType.CPU_CORES: 16.0,
                    ResourceType.MEMORY_GB: 64.0,
                    ResourceType.NETWORK_GBPS: 10.0
                },
                latency_budget_ms=100.0,
                geographic_preferences=["Global"]
            ),
            WorkloadProfile(
                workload_id="global_ai_inference",
                performance_requirements=PerformanceClass.BATCH,
                resource_requirements={
                    ResourceType.CPU_CORES: 32.0,
                    ResourceType.MEMORY_GB: 128.0,
                    ResourceType.GPU_UNITS: 8.0,
                    ResourceType.TENSOR_UNITS: 16.0
                },
                latency_budget_ms=500.0
            ),
            WorkloadProfile(
                workload_id="global_edge_hybrid",
                performance_requirements=PerformanceClass.INTERACTIVE,
                resource_requirements={
                    ResourceType.CPU_CORES: 8.0,
                    ResourceType.MEMORY_GB: 32.0
                },
                latency_budget_ms=50.0,
                edge_deployment=True
            )
        ]
        
    elif scale_target == ScaleTarget.PLANETARY_MESH:
        # Planetary workloads: specialized requirements
        workloads = [
            WorkloadProfile(
                workload_id="planetary_coordination",
                performance_requirements=PerformanceClass.BATCH,
                resource_requirements={
                    ResourceType.CPU_CORES: 64.0,
                    ResourceType.MEMORY_GB: 256.0,
                    ResourceType.NETWORK_GBPS: 100.0
                },
                latency_budget_ms=2000.0,
                geographic_preferences=["Global", "Space", "Undersea"]
            ),
            WorkloadProfile(
                workload_id="satellite_processing",
                performance_requirements=PerformanceClass.INTERACTIVE,
                resource_requirements={
                    ResourceType.CPU_CORES: 16.0,
                    ResourceType.MEMORY_GB: 64.0
                },
                latency_budget_ms=500.0,
                geographic_preferences=["Space", "Mobile"]
            )
        ]
        
    elif scale_target == ScaleTarget.QUANTUM_GRID:
        # Quantum workloads: quantum-enhanced processing
        workloads = [
            WorkloadProfile(
                workload_id="quantum_optimization",
                performance_requirements=PerformanceClass.QUANTUM,
                resource_requirements={
                    ResourceType.CPU_CORES: 128.0,
                    ResourceType.MEMORY_GB: 512.0,
                    ResourceType.QUANTUM_QUBITS: 32.0
                },
                latency_budget_ms=10000.0,
                quantum_acceleration=True
            ),
            WorkloadProfile(
                workload_id="quantum_ml_training",
                performance_requirements=PerformanceClass.QUANTUM,
                resource_requirements={
                    ResourceType.CPU_CORES: 256.0,
                    ResourceType.MEMORY_GB: 1024.0,
                    ResourceType.QUANTUM_QUBITS: 64.0,
                    ResourceType.TENSOR_UNITS: 32.0
                },
                latency_budget_ms=30000.0,
                quantum_acceleration=True
            )
        ]
        
    elif scale_target == ScaleTarget.HYPERDIMENSIONAL:
        # Hyperdimensional workloads: maximum scale and complexity
        workloads = [
            WorkloadProfile(
                workload_id="hyperdimensional_search",
                performance_requirements=PerformanceClass.QUANTUM,
                resource_requirements={
                    ResourceType.CPU_CORES: 512.0,
                    ResourceType.MEMORY_GB: 2048.0,
                    ResourceType.QUANTUM_QUBITS: 128.0,
                    ResourceType.TENSOR_UNITS: 256.0
                },
                latency_budget_ms=60000.0,
                quantum_acceleration=True
            ),
            WorkloadProfile(
                workload_id="hyperdimensional_ai",
                performance_requirements=PerformanceClass.QUANTUM,
                resource_requirements={
                    ResourceType.CPU_CORES: 1024.0,
                    ResourceType.MEMORY_GB: 4096.0,
                    ResourceType.QUANTUM_QUBITS: 256.0,
                    ResourceType.TENSOR_UNITS: 512.0
                },
                latency_budget_ms=120000.0,
                quantum_acceleration=True
            ),
            WorkloadProfile(
                workload_id="planetary_consciousness",
                performance_requirements=PerformanceClass.QUANTUM,
                resource_requirements={
                    ResourceType.CPU_CORES: 2048.0,
                    ResourceType.MEMORY_GB: 8192.0,
                    ResourceType.QUANTUM_QUBITS: 512.0,
                    ResourceType.TENSOR_UNITS: 1024.0
                },
                latency_budget_ms=300000.0,
                quantum_acceleration=True
            )
        ]
        
    return workloads


async def demonstrate_hyperdimensional_optimization():
    """Demonstrate advanced hyperdimensional optimization."""
    logger.info("🌌 Starting Hyperdimensional Optimization Demo")
    
    from wasm_torch.planetary_scale_orchestrator import HyperdimensionalOptimizer, NodeCapacity
    
    # Initialize hyperdimensional optimizer
    optimizer = HyperdimensionalOptimizer(dimensions=2048)
    
    # Create test workload
    workload = WorkloadProfile(
        workload_id="hyperdim_test",
        performance_requirements=PerformanceClass.QUANTUM,
        resource_requirements={
            ResourceType.CPU_CORES: 64.0,
            ResourceType.MEMORY_GB: 256.0,
            ResourceType.QUANTUM_QUBITS: 32.0
        },
        quantum_acceleration=True
    )
    
    # Create test nodes
    nodes = [
        NodeCapacity(
            node_id=f"hyperdim_node_{i}",
            location=f"Dimension_{i}",
            resources={
                ResourceType.CPU_CORES: 128.0,
                ResourceType.MEMORY_GB: 512.0,
                ResourceType.QUANTUM_QUBITS: 64.0
            },
            performance_class=PerformanceClass.QUANTUM,
            quantum_enabled=True,
            health_score=0.95
        )
        for i in range(10)
    ]
    
    # Optimization target
    optimization_target = OptimizationTarget(
        quantum_preference=True,
        weights={
            "latency": 0.25,
            "cost": 0.15,
            "throughput": 0.25,
            "availability": 0.20,
            "energy": 0.10,
            "quantum": 0.05
        }
    )
    
    start_time = time.time()
    
    # Perform hyperdimensional optimization
    optimal_placement, fitness_score = await optimizer.optimize_placement(
        workload, nodes, optimization_target
    )
    
    optimization_time = time.time() - start_time
    
    logger.info("🌌 Hyperdimensional Optimization Results:")
    logger.info(f"  Optimization Time: {optimization_time:.4f}s")
    logger.info(f"  Optimal Placement: {optimal_placement}")
    logger.info(f"  Fitness Score: {fitness_score:.6f}")
    logger.info(f"  Dimensions Processed: {optimizer.dimensions}")
    
    return {
        "optimization_time": optimization_time,
        "optimal_placement": optimal_placement,
        "fitness_score": fitness_score,
        "dimensions": optimizer.dimensions
    }


async def demonstrate_quantum_acceleration():
    """Demonstrate quantum acceleration capabilities."""
    logger.info("⚛️ Starting Quantum Acceleration Demo")
    
    from wasm_torch.planetary_scale_orchestrator import QuantumAccelerator
    
    # Initialize quantum accelerator
    accelerator = QuantumAccelerator()
    
    # Test workloads
    test_workloads = [
        WorkloadProfile(
            workload_id="quantum_search",
            performance_requirements=PerformanceClass.QUANTUM,
            resource_requirements={ResourceType.QUANTUM_QUBITS: 32.0},
            quantum_acceleration=True
        ),
        WorkloadProfile(
            workload_id="classical_batch",
            performance_requirements=PerformanceClass.BATCH,
            resource_requirements={ResourceType.CPU_CORES: 16.0},
            quantum_acceleration=False
        ),
        WorkloadProfile(
            workload_id="quantum_ml_optimization",
            performance_requirements=PerformanceClass.QUANTUM,
            resource_requirements={ResourceType.QUANTUM_QUBITS: 64.0},
            quantum_acceleration=True
        )
    ]
    
    results = []
    
    for workload in test_workloads:
        can_accelerate, speedup = await accelerator.can_accelerate(workload)
        
        logger.info(f"🔬 Testing {workload.workload_id}:")
        logger.info(f"  Can Accelerate: {can_accelerate}")
        logger.info(f"  Expected Speedup: {speedup:.2f}x")
        
        if can_accelerate:
            # Simulate acceleration
            test_data = {"input": "test_quantum_data", "size": 1000}
            result, metrics = await accelerator.accelerate_workload(workload, test_data)
            
            logger.info(f"  Quantum Processing Results:")
            logger.info(f"    Actual Speedup: {metrics['quantum_speedup']:.2f}x")
            logger.info(f"    Processing Time: {metrics['processing_time']:.4f}s")
            logger.info(f"    Quantum Gates Used: {metrics['quantum_gates_used']}")
            logger.info(f"    Coherence Utilized: {metrics['coherence_utilized']:.1%}")
            
            results.append({
                "workload_id": workload.workload_id,
                "can_accelerate": can_accelerate,
                "expected_speedup": speedup,
                "actual_speedup": metrics['quantum_speedup'],
                "processing_time": metrics['processing_time'],
                "quantum_gates_used": metrics['quantum_gates_used']
            })
        else:
            results.append({
                "workload_id": workload.workload_id,
                "can_accelerate": can_accelerate,
                "expected_speedup": speedup
            })
            
    # Calculate quantum advantage summary
    quantum_workloads = [r for r in results if r["can_accelerate"]]
    if quantum_workloads:
        avg_speedup = sum(r["actual_speedup"] for r in quantum_workloads) / len(quantum_workloads)
        logger.info(f"⚛️ Quantum Acceleration Summary:")
        logger.info(f"  Quantum-Accelerated Workloads: {len(quantum_workloads)}")
        logger.info(f"  Average Quantum Speedup: {avg_speedup:.2f}x")
        
    return results


async def demonstrate_edge_optimization():
    """Demonstrate edge optimization capabilities."""
    logger.info("📍 Starting Edge Optimization Demo")
    
    from wasm_torch.planetary_scale_orchestrator import EdgeOptimizer
    
    # Initialize edge optimizer
    optimizer = EdgeOptimizer()
    
    # Simulate user locations around the world
    user_locations = [
        (40.7128, -74.0060),   # New York
        (37.7749, -122.4194),  # San Francisco
        (51.5074, -0.1278),    # London
        (48.8566, 2.3522),     # Paris
        (35.6762, 139.6503),   # Tokyo
        (37.5665, 126.9780),   # Seoul
        (-33.8688, 151.2093),  # Sydney
        (55.7558, 37.6176),    # Moscow
        (-23.5505, -46.6333),  # São Paulo
        (19.4326, -99.1332)    # Mexico City
    ]
    
    # Test workload
    workload = WorkloadProfile(
        workload_id="global_edge_service",
        performance_requirements=PerformanceClass.REALTIME,
        resource_requirements={
            ResourceType.CPU_CORES: 4.0,
            ResourceType.MEMORY_GB: 16.0
        },
        latency_budget_ms=10.0,
        edge_deployment=True
    )
    
    start_time = time.time()
    
    # Optimize edge placement
    placement_result = await optimizer.optimize_edge_placement(workload, user_locations)
    
    optimization_time = time.time() - start_time
    
    logger.info("📍 Edge Optimization Results:")
    logger.info(f"  Optimization Time: {optimization_time:.4f}s")
    logger.info(f"  Optimal Edge Location: {placement_result['optimal_placement']}")
    
    # Show top recommendations
    logger.info("🏆 Top Edge Placement Recommendations:")
    for i, (edge_id, details) in enumerate(placement_result['recommendations']):
        logger.info(f"  {i+1}. {edge_id}:")
        logger.info(f"     Average Distance: {details['average_distance_km']:.0f} km")
        logger.info(f"     Estimated Latency: {details['estimated_latency_ms']:.1f} ms")
        logger.info(f"     Capacity: {details['capacity']} units")
        
    return {
        "optimization_time": optimization_time,
        "optimal_placement": placement_result['optimal_placement'],
        "recommendations": placement_result['recommendations'],
        "user_locations_count": len(user_locations)
    }


async def run_comprehensive_planetary_demo():
    """Run comprehensive planetary scale demonstration."""
    logger.info("🌍 Starting Comprehensive Planetary Scale Demo")
    
    start_time = time.time()
    
    try:
        # Demo 1: Planetary Scaling
        logger.info("=" * 60)
        planetary_results = await demonstrate_planetary_scaling()
        
        # Demo 2: Hyperdimensional Optimization
        logger.info("=" * 60)
        hyperdim_results = await demonstrate_hyperdimensional_optimization()
        
        # Demo 3: Quantum Acceleration
        logger.info("=" * 60)
        quantum_results = await demonstrate_quantum_acceleration()
        
        # Demo 4: Edge Optimization
        logger.info("=" * 60)
        edge_results = await demonstrate_edge_optimization()
        
        total_time = time.time() - start_time
        
        # Summary Report
        logger.info("=" * 60)
        logger.info("🎉 COMPREHENSIVE PLANETARY DEMO COMPLETED")
        logger.info(f"📊 Total Demo Time: {total_time:.2f}s")
        
        # Planetary scaling summary
        successful_scales = sum(1 for result in planetary_results.values() if result.get('success'))
        total_scales = len(planetary_results)
        logger.info(f"✅ Successful Scale Targets: {successful_scales}/{total_scales}")
        
        # Best performing scale target
        best_scale = None
        best_score = 0
        for scale, result in planetary_results.items():
            if result.get('success'):
                avg_fitness = sum(
                    dep['fitness_score'] for dep in result['deployment_results']
                ) / len(result['deployment_results'])
                if avg_fitness > best_score:
                    best_score = avg_fitness
                    best_scale = scale
                    
        if best_scale:
            logger.info(f"🏆 Best Scale Target: {best_scale} (fitness: {best_score:.4f})")
            
        # Hyperdimensional optimization summary
        logger.info(f"🌌 Hyperdimensional Optimization: {hyperdim_results['fitness_score']:.6f} fitness")
        
        # Quantum acceleration summary
        quantum_workloads = [r for r in quantum_results if r.get("can_accelerate")]
        if quantum_workloads:
            avg_quantum_speedup = sum(r["actual_speedup"] for r in quantum_workloads) / len(quantum_workloads)
            logger.info(f"⚛️ Average Quantum Speedup: {avg_quantum_speedup:.2f}x")
            
        # Edge optimization summary
        logger.info(f"📍 Edge Optimization: {edge_results['optimal_placement']} selected")
        
        # Save demo results
        demo_results = {
            "planetary_results": planetary_results,
            "hyperdimensional_results": hyperdim_results,
            "quantum_results": quantum_results,
            "edge_results": edge_results,
            "total_time_seconds": total_time,
            "demo_timestamp": time.time(),
            "summary": {
                "successful_scales": successful_scales,
                "total_scales": total_scales,
                "best_scale_target": best_scale,
                "best_fitness_score": best_score,
                "quantum_acceleration_count": len(quantum_workloads),
                "average_quantum_speedup": avg_quantum_speedup if quantum_workloads else 0.0
            }
        }
        
        results_path = Path("output/planetary_scale_demo_results.json")
        results_path.parent.mkdir(exist_ok=True)
        
        import json
        with open(results_path, 'w') as f:
            json.dump(demo_results, f, indent=2, default=str)
            
        logger.info(f"💾 Demo results saved to {results_path}")
        
        return demo_results
        
    except Exception as e:
        logger.error(f"❌ Planetary demo failed: {e}")
        raise


if __name__ == "__main__":
    logger.info("🌍 Planetary Scale Orchestration Demo")
    logger.info("🚀 Demonstrating hyperdimensional optimization and quantum acceleration")
    
    try:
        # Run the comprehensive demo
        results = asyncio.run(run_comprehensive_planetary_demo())
        logger.info("🎉 Planetary demo completed successfully!")
        
    except KeyboardInterrupt:
        logger.info("🛑 Demo interrupted by user")
    except Exception as e:
        logger.error(f"❌ Demo failed: {e}")
        sys.exit(1)