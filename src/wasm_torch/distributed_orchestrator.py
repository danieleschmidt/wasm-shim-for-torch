"""
Distributed Orchestrator - Generation 3: Make It Scale
Multi-node coordination and distributed processing for planetary-scale deployment.
"""

import asyncio
import logging
import time
import json
import uuid
from typing import Dict, Any, List, Optional, Union, Callable, Set, Tuple
from dataclasses import dataclass, field
from enum import Enum, auto
from collections import defaultdict, deque
import threading
import hashlib
import socket
import weakref

logger = logging.getLogger(__name__)


class NodeStatus(Enum):
    """Node status in distributed system."""
    
    UNKNOWN = auto()      # Status unknown
    HEALTHY = auto()      # Node is healthy and available
    DEGRADED = auto()     # Node is experiencing issues but functional
    UNAVAILABLE = auto()  # Node is not responding
    DRAINING = auto()     # Node is being drained for maintenance


class PartitionStrategy(Enum):
    """Data partitioning strategies."""
    
    HASH = auto()         # Hash-based partitioning
    RANGE = auto()        # Range-based partitioning
    ROUND_ROBIN = auto()  # Round-robin distribution
    LOAD_BASED = auto()   # Load-based assignment


class ConsensusAlgorithm(Enum):
    """Distributed consensus algorithms."""
    
    RAFT = auto()         # Raft consensus
    PBFT = auto()         # Practical Byzantine Fault Tolerance
    GOSSIP = auto()       # Gossip-based consensus


@dataclass
class NodeInfo:
    """Information about a node in the distributed system."""
    
    node_id: str
    address: str
    port: int
    status: NodeStatus = NodeStatus.UNKNOWN
    capabilities: Dict[str, Any] = field(default_factory=dict)
    metrics: Dict[str, float] = field(default_factory=dict)
    last_heartbeat: float = field(default_factory=time.time)
    load: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'node_id': self.node_id,
            'address': self.address,
            'port': self.port,
            'status': self.status.name,
            'capabilities': self.capabilities,
            'metrics': self.metrics,
            'last_heartbeat': self.last_heartbeat,
            'load': self.load
        }


@dataclass
class DistributedTask:
    """Task for distributed processing."""
    
    task_id: str
    task_type: str
    payload: Dict[str, Any]
    priority: int = 1
    created_at: float = field(default_factory=time.time)
    assigned_node: Optional[str] = None
    status: str = "pending"
    result: Optional[Any] = None
    error: Optional[str] = None
    retry_count: int = 0
    max_retries: int = 3
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'task_id': self.task_id,
            'task_type': self.task_type,
            'payload': self.payload,
            'priority': self.priority,
            'created_at': self.created_at,
            'assigned_node': self.assigned_node,
            'status': self.status,
            'result': self.result,
            'error': self.error,
            'retry_count': self.retry_count,
            'max_retries': self.max_retries
        }


class NodeHealthMonitor:
    """
    Monitor health of nodes in distributed system.
    """
    
    def __init__(self, heartbeat_interval: float = 5.0, timeout: float = 15.0):
        self.heartbeat_interval = heartbeat_interval
        self.timeout = timeout
        self._nodes: Dict[str, NodeInfo] = {}
        self._node_callbacks: Dict[str, List[Callable]] = defaultdict(list)
        self._running = False
        self._monitor_task: Optional[asyncio.Task] = None
        self._lock = threading.RLock()
    
    async def start(self) -> None:
        """Start health monitoring."""
        if self._running:
            return
        
        self._running = True
        self._monitor_task = asyncio.create_task(self._monitor_loop())
        logger.info("Node health monitor started")
    
    async def stop(self) -> None:
        """Stop health monitoring."""
        self._running = False
        if self._monitor_task and not self._monitor_task.done():
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except asyncio.CancelledError:
                pass
        logger.info("Node health monitor stopped")
    
    def register_node(self, node_info: NodeInfo) -> None:
        """Register a new node."""
        with self._lock:
            self._nodes[node_info.node_id] = node_info
            logger.info(f"Registered node: {node_info.node_id} at {node_info.address}:{node_info.port}")
    
    def unregister_node(self, node_id: str) -> None:
        """Unregister a node."""
        with self._lock:
            if node_id in self._nodes:
                del self._nodes[node_id]
                logger.info(f"Unregistered node: {node_id}")
    
    def update_node_metrics(self, node_id: str, metrics: Dict[str, float]) -> None:
        """Update metrics for a node."""
        with self._lock:
            if node_id in self._nodes:
                self._nodes[node_id].metrics.update(metrics)
                self._nodes[node_id].last_heartbeat = time.time()
    
    def get_healthy_nodes(self) -> List[NodeInfo]:
        """Get list of healthy nodes."""
        with self._lock:
            return [
                node for node in self._nodes.values()
                if node.status == NodeStatus.HEALTHY
            ]
    
    def get_node_by_id(self, node_id: str) -> Optional[NodeInfo]:
        """Get node by ID."""
        with self._lock:
            return self._nodes.get(node_id)
    
    def get_all_nodes(self) -> List[NodeInfo]:
        """Get all nodes."""
        with self._lock:
            return list(self._nodes.values())
    
    def add_status_callback(self, node_id: str, callback: Callable[[NodeInfo], None]) -> None:
        """Add callback for node status changes."""
        self._node_callbacks[node_id].append(callback)
    
    async def _monitor_loop(self) -> None:
        """Main monitoring loop."""
        while self._running:
            try:
                current_time = time.time()
                
                with self._lock:
                    nodes_to_update = list(self._nodes.values())
                
                for node in nodes_to_update:
                    old_status = node.status
                    
                    # Check if node has timed out
                    time_since_heartbeat = current_time - node.last_heartbeat
                    
                    if time_since_heartbeat > self.timeout:
                        node.status = NodeStatus.UNAVAILABLE
                    elif time_since_heartbeat > self.timeout * 0.5:
                        node.status = NodeStatus.DEGRADED
                    else:
                        node.status = NodeStatus.HEALTHY
                    
                    # Trigger callbacks on status change
                    if old_status != node.status:
                        logger.info(f"Node {node.node_id} status changed: {old_status.name} -> {node.status.name}")
                        for callback in self._node_callbacks.get(node.node_id, []):
                            try:
                                callback(node)
                            except Exception as e:
                                logger.error(f"Status callback error: {e}")
                
                await asyncio.sleep(self.heartbeat_interval)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Health monitor error: {e}")
                await asyncio.sleep(1.0)


class LoadBalancer:
    """
    Intelligent load balancer for distributed tasks.
    """
    
    def __init__(self, strategy: str = "weighted_round_robin"):
        self.strategy = strategy
        self._round_robin_counter = 0
        self._node_weights: Dict[str, float] = {}
        self._lock = threading.RLock()
    
    def select_node(self, nodes: List[NodeInfo], task: DistributedTask) -> Optional[NodeInfo]:
        """Select best node for a task."""
        if not nodes:
            return None
        
        healthy_nodes = [n for n in nodes if n.status == NodeStatus.HEALTHY]
        if not healthy_nodes:
            # Fall back to degraded nodes if no healthy ones
            healthy_nodes = [n for n in nodes if n.status == NodeStatus.DEGRADED]
        
        if not healthy_nodes:
            return None
        
        if self.strategy == "round_robin":
            return self._round_robin_selection(healthy_nodes)
        elif self.strategy == "least_loaded":
            return self._least_loaded_selection(healthy_nodes)
        elif self.strategy == "weighted_round_robin":
            return self._weighted_selection(healthy_nodes)
        elif self.strategy == "capability_based":
            return self._capability_based_selection(healthy_nodes, task)
        else:
            # Default to round robin
            return self._round_robin_selection(healthy_nodes)
    
    def _round_robin_selection(self, nodes: List[NodeInfo]) -> NodeInfo:
        """Round-robin node selection."""
        with self._lock:
            selected = nodes[self._round_robin_counter % len(nodes)]
            self._round_robin_counter += 1
            return selected
    
    def _least_loaded_selection(self, nodes: List[NodeInfo]) -> NodeInfo:
        """Select node with lowest load."""
        return min(nodes, key=lambda n: n.load)
    
    def _weighted_selection(self, nodes: List[NodeInfo]) -> NodeInfo:
        """Weighted selection based on node performance."""
        # Calculate weights based on inverse load and performance metrics
        total_weight = 0
        weights = []
        
        for node in nodes:
            # Base weight is inverse of load (avoid division by zero)
            load_weight = 1.0 / max(node.load, 0.1)
            
            # Adjust based on performance metrics
            perf_multiplier = 1.0
            if 'success_rate' in node.metrics:
                perf_multiplier *= node.metrics['success_rate']
            if 'avg_latency' in node.metrics:
                # Lower latency is better
                perf_multiplier *= max(0.1, 1.0 - min(node.metrics['avg_latency'], 1.0))
            
            weight = load_weight * perf_multiplier
            weights.append(weight)
            total_weight += weight
        
        # Select based on weights
        if total_weight == 0:
            return self._round_robin_selection(nodes)
        
        import random
        r = random.random() * total_weight
        cumulative = 0
        
        for i, weight in enumerate(weights):
            cumulative += weight
            if r <= cumulative:
                return nodes[i]
        
        return nodes[-1]  # Fallback
    
    def _capability_based_selection(self, nodes: List[NodeInfo], task: DistributedTask) -> NodeInfo:
        """Select node based on task requirements and node capabilities."""
        # Filter nodes that can handle this task type
        capable_nodes = []
        
        for node in nodes:
            capabilities = node.capabilities
            task_type = task.task_type
            
            # Check if node can handle this task type
            if 'supported_tasks' in capabilities:
                if task_type in capabilities['supported_tasks']:
                    capable_nodes.append(node)
            else:
                # If no capability info, assume all nodes can handle all tasks
                capable_nodes.append(node)
        
        if not capable_nodes:
            # Fall back to any available node
            capable_nodes = nodes
        
        # Use weighted selection among capable nodes
        return self._weighted_selection(capable_nodes)


class DistributedTaskQueue:
    """
    Distributed task queue with partitioning and fault tolerance.
    """
    
    def __init__(self, partition_strategy: PartitionStrategy = PartitionStrategy.HASH):
        self.partition_strategy = partition_strategy
        self._pending_tasks: Dict[str, DistributedTask] = {}
        self._active_tasks: Dict[str, DistributedTask] = {}
        self._completed_tasks: deque = deque(maxlen=10000)  # Keep recent history
        self._task_assignments: Dict[str, str] = {}  # task_id -> node_id
        self._lock = threading.RLock()
        
        # Statistics
        self._stats = {
            'tasks_submitted': 0,
            'tasks_completed': 0,
            'tasks_failed': 0,
            'tasks_retried': 0
        }
    
    def submit_task(self, task: DistributedTask) -> str:
        """Submit a task to the queue."""
        with self._lock:
            self._pending_tasks[task.task_id] = task
            self._stats['tasks_submitted'] += 1
            logger.debug(f"Submitted task: {task.task_id}")
            return task.task_id
    
    def get_pending_tasks(self, limit: Optional[int] = None) -> List[DistributedTask]:
        """Get pending tasks for assignment."""
        with self._lock:
            tasks = list(self._pending_tasks.values())
            # Sort by priority (higher first) then by creation time
            tasks.sort(key=lambda t: (-t.priority, t.created_at))
            return tasks[:limit] if limit else tasks
    
    def assign_task(self, task_id: str, node_id: str) -> bool:
        """Assign a task to a node."""
        with self._lock:
            if task_id in self._pending_tasks:
                task = self._pending_tasks.pop(task_id)
                task.assigned_node = node_id
                task.status = "assigned"
                self._active_tasks[task_id] = task
                self._task_assignments[task_id] = node_id
                logger.debug(f"Assigned task {task_id} to node {node_id}")
                return True
            return False
    
    def complete_task(self, task_id: str, result: Any = None, error: Optional[str] = None) -> bool:
        """Mark a task as completed."""
        with self._lock:
            if task_id in self._active_tasks:
                task = self._active_tasks.pop(task_id)
                task.result = result
                task.error = error
                task.status = "completed" if error is None else "failed"
                
                self._completed_tasks.append(task)
                
                if task_id in self._task_assignments:
                    del self._task_assignments[task_id]
                
                if error:
                    self._stats['tasks_failed'] += 1
                    logger.warning(f"Task {task_id} failed: {error}")
                else:
                    self._stats['tasks_completed'] += 1
                    logger.debug(f"Task {task_id} completed successfully")
                
                return True
            return False
    
    def retry_task(self, task_id: str) -> bool:
        """Retry a failed task."""
        with self._lock:
            if task_id in self._active_tasks:
                task = self._active_tasks[task_id]
                
                if task.retry_count < task.max_retries:
                    # Move back to pending
                    self._active_tasks.pop(task_id)
                    task.retry_count += 1
                    task.assigned_node = None
                    task.status = "pending"
                    task.error = None
                    self._pending_tasks[task_id] = task
                    
                    if task_id in self._task_assignments:
                        del self._task_assignments[task_id]
                    
                    self._stats['tasks_retried'] += 1
                    logger.info(f"Retrying task {task_id} (attempt {task.retry_count})")
                    return True
            
            return False
    
    def get_tasks_for_node(self, node_id: str) -> List[DistributedTask]:
        """Get all active tasks assigned to a node."""
        with self._lock:
            return [
                task for task in self._active_tasks.values()
                if task.assigned_node == node_id
            ]
    
    def get_queue_stats(self) -> Dict[str, Any]:
        """Get queue statistics."""
        with self._lock:
            return {
                'pending_tasks': len(self._pending_tasks),
                'active_tasks': len(self._active_tasks),
                'completed_tasks': len(self._completed_tasks),
                **self._stats
            }


class DistributedOrchestrator:
    """
    Main distributed orchestrator coordinating multi-node operations.
    Generation 3: Make It Scale - Optimized for planetary-scale distribution.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.node_id = self.config.get('node_id', f"node_{uuid.uuid4().hex[:8]}")
        
        # Core components
        self.health_monitor = NodeHealthMonitor(
            heartbeat_interval=self.config.get('heartbeat_interval', 5.0),
            timeout=self.config.get('node_timeout', 15.0)
        )
        
        self.load_balancer = LoadBalancer(
            strategy=self.config.get('load_balancing', 'weighted_round_robin')
        )
        
        self.task_queue = DistributedTaskQueue(
            partition_strategy=PartitionStrategy(self.config.get('partition_strategy', 1))
        )
        
        # Orchestration state
        self._running = False
        self._orchestration_task: Optional[asyncio.Task] = None
        self._rebalance_task: Optional[asyncio.Task] = None
        
        # Node registry
        self._local_node_info = NodeInfo(
            node_id=self.node_id,
            address=self._get_local_ip(),
            port=self.config.get('port', 8080),
            capabilities=self.config.get('capabilities', {}),
            status=NodeStatus.HEALTHY
        )
        
        # Performance tracking
        self._orchestration_stats = {
            'tasks_orchestrated': 0,
            'nodes_managed': 0,
            'rebalance_operations': 0,
            'failover_operations': 0
        }
        
        self._lock = threading.RLock()
    
    async def start(self) -> bool:
        """Start the distributed orchestrator."""
        try:
            if self._running:
                return True
            
            logger.info(f"Starting distributed orchestrator (node: {self.node_id})")
            
            # Start health monitoring
            await self.health_monitor.start()
            
            # Register local node
            self.health_monitor.register_node(self._local_node_info)
            
            self._running = True
            
            # Start orchestration tasks
            self._orchestration_task = asyncio.create_task(self._orchestration_loop())
            self._rebalance_task = asyncio.create_task(self._rebalance_loop())
            
            logger.info("Distributed orchestrator started successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to start distributed orchestrator: {e}")
            return False
    
    async def stop(self) -> None:
        """Stop the distributed orchestrator."""
        try:
            logger.info("Stopping distributed orchestrator")
            self._running = False
            
            # Cancel tasks
            if self._orchestration_task and not self._orchestration_task.done():
                self._orchestration_task.cancel()
            if self._rebalance_task and not self._rebalance_task.done():
                self._rebalance_task.cancel()
            
            # Wait for tasks to complete
            tasks = [t for t in [self._orchestration_task, self._rebalance_task] if t]
            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)
            
            # Stop health monitor
            await self.health_monitor.stop()
            
            logger.info("Distributed orchestrator stopped")
            
        except Exception as e:
            logger.error(f"Error stopping distributed orchestrator: {e}")
    
    def join_cluster(self, bootstrap_nodes: List[Tuple[str, int]]) -> bool:
        """Join an existing cluster."""
        try:
            for address, port in bootstrap_nodes:
                # In real implementation, this would use network discovery
                node_id = f"node_{address}_{port}"
                node_info = NodeInfo(
                    node_id=node_id,
                    address=address,
                    port=port,
                    status=NodeStatus.HEALTHY
                )
                self.health_monitor.register_node(node_info)
                logger.info(f"Joined cluster via bootstrap node: {address}:{port}")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to join cluster: {e}")
            return False
    
    def add_node(self, node_info: NodeInfo) -> None:
        """Add a new node to the cluster."""
        self.health_monitor.register_node(node_info)
        with self._lock:
            self._orchestration_stats['nodes_managed'] += 1
    
    def remove_node(self, node_id: str) -> None:
        """Remove a node from the cluster."""
        # Reassign tasks from the node being removed
        tasks = self.task_queue.get_tasks_for_node(node_id)
        for task in tasks:
            logger.info(f"Reassigning task {task.task_id} from removed node {node_id}")
            self.task_queue.retry_task(task.task_id)
        
        self.health_monitor.unregister_node(node_id)
    
    async def submit_distributed_task(
        self,
        task_type: str,
        payload: Dict[str, Any],
        priority: int = 1
    ) -> str:
        """Submit a task for distributed processing."""
        task = DistributedTask(
            task_id=f"task_{uuid.uuid4().hex[:8]}",
            task_type=task_type,
            payload=payload,
            priority=priority
        )
        
        task_id = self.task_queue.submit_task(task)
        
        with self._lock:
            self._orchestration_stats['tasks_orchestrated'] += 1
        
        logger.info(f"Submitted distributed task: {task_id}")
        return task_id
    
    async def get_task_result(self, task_id: str, timeout: float = 60.0) -> Optional[DistributedTask]:
        """Get result of a distributed task."""
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            # Check in completed tasks
            for completed_task in self.task_queue._completed_tasks:
                if completed_task.task_id == task_id:
                    return completed_task
            
            # Check if still pending or active
            if (task_id in self.task_queue._pending_tasks or 
                task_id in self.task_queue._active_tasks):
                await asyncio.sleep(0.5)  # Poll every 500ms
                continue
            
            break
        
        return None
    
    def get_cluster_status(self) -> Dict[str, Any]:
        """Get comprehensive cluster status."""
        nodes = self.health_monitor.get_all_nodes()
        healthy_nodes = self.health_monitor.get_healthy_nodes()
        
        return {
            'cluster_id': self.node_id,
            'node_count': len(nodes),
            'healthy_nodes': len(healthy_nodes),
            'nodes': [node.to_dict() for node in nodes],
            'task_queue': self.task_queue.get_queue_stats(),
            'orchestration_stats': self._orchestration_stats,
            'running': self._running
        }
    
    async def _orchestration_loop(self) -> None:
        """Main orchestration loop for task assignment."""
        while self._running:
            try:
                # Get pending tasks
                pending_tasks = self.task_queue.get_pending_tasks(limit=50)
                
                if pending_tasks:
                    healthy_nodes = self.health_monitor.get_healthy_nodes()
                    
                    if healthy_nodes:
                        # Assign tasks to nodes
                        for task in pending_tasks:
                            selected_node = self.load_balancer.select_node(healthy_nodes, task)
                            if selected_node:
                                success = self.task_queue.assign_task(task.task_id, selected_node.node_id)
                                if success:
                                    # In real implementation, send task to node via network
                                    await self._simulate_task_execution(task, selected_node)
                    else:
                        logger.warning("No healthy nodes available for task assignment")
                
                await asyncio.sleep(1.0)  # Check for new tasks every second
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Orchestration loop error: {e}")
                await asyncio.sleep(5.0)
    
    async def _rebalance_loop(self) -> None:
        """Periodic load rebalancing."""
        while self._running:
            try:
                # Perform rebalancing every 30 seconds
                await asyncio.sleep(30.0)
                
                if not self._running:
                    break
                
                await self._perform_load_rebalancing()
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Rebalance loop error: {e}")
                await asyncio.sleep(10.0)
    
    async def _perform_load_rebalancing(self) -> None:
        """Perform load rebalancing across nodes."""
        try:
            nodes = self.health_monitor.get_healthy_nodes()
            if len(nodes) < 2:
                return  # No need to rebalance with less than 2 nodes
            
            # Calculate load distribution
            node_loads = [(node.node_id, node.load) for node in nodes]
            node_loads.sort(key=lambda x: x[1])  # Sort by load
            
            # Check if rebalancing is needed
            if node_loads:
                min_load = node_loads[0][1]
                max_load = node_loads[-1][1]
                
                # If difference is significant, consider rebalancing
                if max_load - min_load > 2.0:  # Arbitrary threshold
                    logger.info(f"Load imbalance detected: min={min_load:.2f}, max={max_load:.2f}")
                    
                    # In real implementation, we would reassign some tasks
                    # from high-load nodes to low-load nodes
                    with self._lock:
                        self._orchestration_stats['rebalance_operations'] += 1
                    
                    logger.info("Load rebalancing completed")
            
        except Exception as e:
            logger.error(f"Load rebalancing error: {e}")
    
    async def _simulate_task_execution(self, task: DistributedTask, node: NodeInfo) -> None:
        """Simulate task execution on a node (for demo purposes)."""
        try:
            # Simulate processing time
            processing_time = 0.1 + (hash(task.task_id) % 100) / 1000
            await asyncio.sleep(processing_time)
            
            # Simulate result
            result = {
                'task_id': task.task_id,
                'processed_by': node.node_id,
                'processing_time': processing_time,
                'result': f"Processed {task.task_type} on {node.node_id}"
            }
            
            # Complete the task
            self.task_queue.complete_task(task.task_id, result=result)
            
            # Update node metrics
            node.load = max(0, node.load - 0.1)  # Simulate load decrease
            
        except Exception as e:
            # Fail the task
            self.task_queue.complete_task(task.task_id, error=str(e))
            logger.error(f"Task execution failed: {e}")
    
    def _get_local_ip(self) -> str:
        """Get local IP address."""
        try:
            # Connect to a remote address to determine local IP
            with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
                s.connect(("8.8.8.8", 80))
                return s.getsockname()[0]
        except Exception:
            return "127.0.0.1"


# Example usage and testing
async def demo_distributed_orchestrator():
    """Demonstration of distributed orchestrator."""
    # Create orchestrator
    config = {
        'node_id': 'master_node',
        'heartbeat_interval': 2.0,
        'node_timeout': 6.0,
        'capabilities': {
            'supported_tasks': ['inference', 'training', 'preprocessing'],
            'max_concurrent_tasks': 10
        }
    }
    
    orchestrator = DistributedOrchestrator(config)
    
    try:
        # Start orchestrator
        success = await orchestrator.start()
        if not success:
            print("Failed to start distributed orchestrator")
            return
        
        print("Distributed orchestrator started")
        
        # Simulate adding nodes to cluster
        for i in range(3):
            node_info = NodeInfo(
                node_id=f"worker_node_{i}",
                address=f"192.168.1.{100 + i}",
                port=8080,
                capabilities={'supported_tasks': ['inference'], 'max_concurrent_tasks': 5},
                status=NodeStatus.HEALTHY
            )
            orchestrator.add_node(node_info)
        
        print("Added worker nodes to cluster")
        
        # Submit distributed tasks
        task_types = ['inference', 'preprocessing', 'training']
        task_ids = []
        
        print("Submitting distributed tasks...")
        for i in range(20):
            task_type = task_types[i % len(task_types)]
            payload = {
                'model_id': f'model_{i % 3}',
                'input_data': f'data_batch_{i}',
                'parameters': {'batch_size': 32}
            }
            
            task_id = await orchestrator.submit_distributed_task(
                task_type=task_type,
                payload=payload,
                priority=1 + (i % 3)
            )
            task_ids.append(task_id)
        
        print(f"Submitted {len(task_ids)} tasks")
        
        # Wait for some tasks to complete
        await asyncio.sleep(5.0)
        
        # Check results for first few tasks
        print("Checking task results...")
        for i in range(min(5, len(task_ids))):
            task_id = task_ids[i]
            result_task = await orchestrator.get_task_result(task_id, timeout=1.0)
            
            if result_task:
                print(f"Task {task_id}: {result_task.status}")
                if result_task.result:
                    print(f"  Result: {result_task.result}")
            else:
                print(f"Task {task_id}: Still processing or not found")
        
        # Show cluster status
        cluster_status = orchestrator.get_cluster_status()
        print(f"\nCluster Status:")
        print(json.dumps(cluster_status, indent=2, default=str))
        
    finally:
        # Stop orchestrator
        await orchestrator.stop()
        print("\nDistributed orchestrator stopped")


if __name__ == "__main__":
    asyncio.run(demo_distributed_orchestrator())