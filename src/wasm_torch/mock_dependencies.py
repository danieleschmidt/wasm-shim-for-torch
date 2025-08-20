"""
Mock implementations for optional dependencies.
Provides basic functionality for demonstration when dependencies are not available.
"""

import random
import time


# Mock numpy implementation
class MockNumPy:
    """Mock numpy for basic mathematical operations."""
    
    @staticmethod
    def zeros(size):
        """Create array of zeros."""
        if isinstance(size, int):
            return [0.0] * size
        else:
            # Multi-dimensional array
            total_size = 1
            for dim in size:
                total_size *= dim
            return [0.0] * total_size
    
    @staticmethod
    def random():
        """Random number generation."""
        class Random:
            @staticmethod
            def randn(*args):
                """Random normal distribution."""
                if len(args) == 1:
                    return [random.gauss(0, 1) for _ in range(args[0])]
                elif len(args) == 2:
                    result = []
                    for i in range(args[0]):
                        result.append([random.gauss(0, 1) for _ in range(args[1])])
                    return result
                else:
                    return random.gauss(0, 1)
        return Random()
    
    @staticmethod
    def eye(n):
        """Identity matrix."""
        result = []
        for i in range(n):
            row = [0.0] * n
            if i < n:
                row[i] = 1.0
            result.append(row)
        return result
    
    @staticmethod
    def dot(a, b):
        """Dot product."""
        if isinstance(a, list) and isinstance(b, list):
            if len(a) != len(b):
                return 0.0
            return sum(x * y for x, y in zip(a, b))
        return 0.0
    
    @staticmethod
    def linalg():
        """Linear algebra operations."""
        class LinAlg:
            @staticmethod
            def norm(vector):
                """Vector norm."""
                if isinstance(vector, list):
                    return sum(x * x for x in vector) ** 0.5
                return abs(vector)
        return LinAlg()


# Mock psutil implementation
class MockPsutil:
    """Mock psutil for system monitoring."""
    
    @staticmethod
    def virtual_memory():
        """Mock virtual memory info."""
        class VirtualMemory:
            percent = 45.0  # Mock 45% usage
            available = 8 * 1024**3  # 8GB available
            total = 16 * 1024**3  # 16GB total
            used = 8 * 1024**3  # 8GB used
        return VirtualMemory()
    
    @staticmethod
    def cpu_percent(interval=None):
        """Mock CPU percentage."""
        if interval:
            time.sleep(interval)
        return random.uniform(20.0, 60.0)
    
    @staticmethod
    def cpu_count():
        """Mock CPU count."""
        return 8
    
    @staticmethod
    def disk_usage(path):
        """Mock disk usage."""
        class DiskUsage:
            total = 1000 * 1024**3  # 1TB
            used = 500 * 1024**3   # 500GB used
            free = 500 * 1024**3   # 500GB free
        return DiskUsage()
    
    @staticmethod
    def net_io_counters():
        """Mock network I/O counters."""
        class NetworkIO:
            bytes_sent = random.randint(1000000, 10000000)
            bytes_recv = random.randint(1000000, 10000000)
            packets_sent = random.randint(1000, 10000)
            packets_recv = random.randint(1000, 10000)
        return NetworkIO()
    
    @staticmethod
    def getloadavg():
        """Mock load average."""
        return (random.uniform(0.5, 2.0), random.uniform(0.5, 2.0), random.uniform(0.5, 2.0))
    
    @staticmethod
    def Process(pid=None):
        """Mock process information."""
        class MockProcess:
            def __init__(self, pid=None):
                self.pid = pid or 12345
                
            def memory_info(self):
                class MemoryInfo:
                    rss = random.randint(100000000, 500000000)  # 100-500MB
                    vms = random.randint(200000000, 1000000000)  # 200MB-1GB
                return MemoryInfo()
                
            def cpu_percent(self):
                return random.uniform(5.0, 25.0)
                
            def status(self):
                return "running"
                
        return MockProcess(pid)
    
    @staticmethod
    def boot_time():
        """Mock system boot time."""
        return time.time() - random.randint(3600, 86400)  # 1-24 hours ago


# Mock aiohttp implementation
class MockAiohttp:
    """Mock aiohttp for HTTP operations."""
    
    class ClientSession:
        """Mock client session."""
        
        async def __aenter__(self):
            return self
            
        async def __aexit__(self, exc_type, exc_val, exc_tb):
            pass
            
        async def get(self, url, **kwargs):
            """Mock GET request."""
            class Response:
                status = 200
                
                async def text(self):
                    return '{"status": "ok", "mock": true}'
                    
                async def json(self):
                    return {"status": "ok", "mock": True}
                    
            await asyncio.sleep(0.01)  # Simulate network delay
            return Response()


# Export mock implementations
np = MockNumPy()
psutil = MockPsutil()

try:
    import asyncio
except ImportError:
    asyncio = None