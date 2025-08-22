"""
Basic Model Loader - Generation 1: Make It Work
Simple, reliable model loading system for WASM-Torch without external dependencies.
"""

import json
import logging
import time
import hashlib
from typing import Dict, Any, List, Optional, Union, BinaryIO
from pathlib import Path
from dataclasses import dataclass, field
import tempfile
import shutil
import threading
from contextlib import contextmanager
import os

logger = logging.getLogger(__name__)


@dataclass
class ModelInfo:
    """Basic model information structure."""
    
    model_id: str
    model_path: str
    model_type: str
    size_bytes: int
    checksum: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    loaded_at: Optional[float] = None
    access_count: int = 0
    last_accessed: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            'model_id': self.model_id,
            'model_path': self.model_path,
            'model_type': self.model_type,
            'size_bytes': self.size_bytes,
            'checksum': self.checksum,
            'metadata': self.metadata,
            'loaded_at': self.loaded_at,
            'access_count': self.access_count,
            'last_accessed': self.last_accessed
        }


class BasicModelStorage:
    """
    Basic model storage system with file management and caching.
    """
    
    def __init__(self, storage_dir: Optional[str] = None):
        self.storage_dir = Path(storage_dir) if storage_dir else Path(tempfile.gettempdir()) / "wasm_torch_models"
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._model_cache: Dict[str, Any] = {}
        self._model_info: Dict[str, ModelInfo] = {}
        
    def store_model(
        self, 
        model_id: str, 
        model_data: Union[bytes, BinaryIO, str], 
        model_type: str = "wasm",
        metadata: Optional[Dict[str, Any]] = None
    ) -> Optional[str]:
        """Store a model in the storage system."""
        try:
            with self._lock:
                model_path = self.storage_dir / f"{model_id}.{model_type}"
                
                # Write model data to file
                if isinstance(model_data, bytes):
                    model_path.write_bytes(model_data)
                    size_bytes = len(model_data)
                    checksum = self._calculate_checksum(model_data)
                    
                elif isinstance(model_data, str):
                    # Treat as file path to copy
                    source_path = Path(model_data)
                    if not source_path.exists():
                        logger.error(f"Source model file not found: {model_data}")
                        return None
                    
                    shutil.copy2(source_path, model_path)
                    size_bytes = model_path.stat().st_size
                    checksum = self._calculate_file_checksum(model_path)
                    
                else:
                    # Treat as file-like object
                    model_data.seek(0)
                    content = model_data.read()
                    if isinstance(content, str):
                        content = content.encode('utf-8')
                    
                    model_path.write_bytes(content)
                    size_bytes = len(content)
                    checksum = self._calculate_checksum(content)
                
                # Store model info
                model_info = ModelInfo(
                    model_id=model_id,
                    model_path=str(model_path),
                    model_type=model_type,
                    size_bytes=size_bytes,
                    checksum=checksum,
                    metadata=metadata or {},
                    loaded_at=time.time()
                )
                
                self._model_info[model_id] = model_info
                
                # Save metadata file
                metadata_path = self.storage_dir / f"{model_id}.metadata.json"
                metadata_path.write_text(json.dumps(model_info.to_dict(), indent=2))
                
                logger.info(f"Stored model {model_id} at {model_path}")
                return str(model_path)
                
        except Exception as e:
            logger.error(f"Failed to store model {model_id}: {e}")
            return None
    
    def load_model_data(self, model_id: str) -> Optional[bytes]:
        """Load raw model data."""
        try:
            with self._lock:
                if model_id not in self._model_info:
                    # Try to restore from metadata file
                    if not self._restore_model_info(model_id):
                        return None
                
                model_info = self._model_info[model_id]
                model_path = Path(model_info.model_path)
                
                if not model_path.exists():
                    logger.error(f"Model file not found: {model_path}")
                    return None
                
                # Update access info
                model_info.access_count += 1
                model_info.last_accessed = time.time()
                
                # Read and verify model data
                model_data = model_path.read_bytes()
                
                # Verify checksum
                if self._calculate_checksum(model_data) != model_info.checksum:
                    logger.warning(f"Checksum mismatch for model {model_id}")
                
                return model_data
                
        except Exception as e:
            logger.error(f"Failed to load model data for {model_id}: {e}")
            return None
    
    def get_model_info(self, model_id: str) -> Optional[ModelInfo]:
        """Get model information."""
        with self._lock:
            if model_id in self._model_info:
                return self._model_info[model_id]
            
            # Try to restore from metadata file
            if self._restore_model_info(model_id):
                return self._model_info[model_id]
            
            return None
    
    def list_models(self) -> List[str]:
        """List all stored models."""
        with self._lock:
            # Scan directory for metadata files
            model_ids = set()
            
            for metadata_file in self.storage_dir.glob("*.metadata.json"):
                model_id = metadata_file.stem.replace(".metadata", "")
                model_ids.add(model_id)
                
                # Restore info if not loaded
                if model_id not in self._model_info:
                    self._restore_model_info(model_id)
            
            return list(model_ids)
    
    def delete_model(self, model_id: str) -> bool:
        """Delete a stored model."""
        try:
            with self._lock:
                model_info = self.get_model_info(model_id)
                if not model_info:
                    return False
                
                # Delete model file
                model_path = Path(model_info.model_path)
                if model_path.exists():
                    model_path.unlink()
                
                # Delete metadata file
                metadata_path = self.storage_dir / f"{model_id}.metadata.json"
                if metadata_path.exists():
                    metadata_path.unlink()
                
                # Remove from cache
                if model_id in self._model_info:
                    del self._model_info[model_id]
                
                if model_id in self._model_cache:
                    del self._model_cache[model_id]
                
                logger.info(f"Deleted model {model_id}")
                return True
                
        except Exception as e:
            logger.error(f"Failed to delete model {model_id}: {e}")
            return False
    
    def _restore_model_info(self, model_id: str) -> bool:
        """Restore model info from metadata file."""
        try:
            metadata_path = self.storage_dir / f"{model_id}.metadata.json"
            if not metadata_path.exists():
                return False
            
            metadata = json.loads(metadata_path.read_text())
            model_info = ModelInfo(
                model_id=metadata['model_id'],
                model_path=metadata['model_path'],
                model_type=metadata['model_type'],
                size_bytes=metadata['size_bytes'],
                checksum=metadata['checksum'],
                metadata=metadata.get('metadata', {}),
                loaded_at=metadata.get('loaded_at'),
                access_count=metadata.get('access_count', 0),
                last_accessed=metadata.get('last_accessed')
            )
            
            self._model_info[model_id] = model_info
            return True
            
        except Exception as e:
            logger.error(f"Failed to restore model info for {model_id}: {e}")
            return False
    
    def _calculate_checksum(self, data: bytes) -> str:
        """Calculate SHA-256 checksum of data."""
        return hashlib.sha256(data).hexdigest()
    
    def _calculate_file_checksum(self, file_path: Path) -> str:
        """Calculate SHA-256 checksum of file."""
        hasher = hashlib.sha256()
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hasher.update(chunk)
        return hasher.hexdigest()


class BasicModelLoader:
    """
    Basic model loader with simple caching and validation.
    Generation 1: Focus on core functionality and reliability.
    """
    
    def __init__(self, storage_dir: Optional[str] = None):
        self.storage = BasicModelStorage(storage_dir)
        self._loaded_models: Dict[str, Any] = {}
        self._load_stats: Dict[str, Dict[str, Any]] = {}
        self._lock = threading.RLock()
        
    def load_model(
        self, 
        model_id: str, 
        force_reload: bool = False
    ) -> Optional[Any]:
        """
        Load a model by ID.
        
        Args:
            model_id: Unique identifier for the model
            force_reload: Force reload even if already cached
            
        Returns:
            Loaded model object or None if loading failed
        """
        try:
            with self._lock:
                # Check if already loaded and not forcing reload
                if model_id in self._loaded_models and not force_reload:
                    self._update_load_stats(model_id, cache_hit=True)
                    logger.debug(f"Returning cached model: {model_id}")
                    return self._loaded_models[model_id]
                
                # Get model info
                model_info = self.storage.get_model_info(model_id)
                if not model_info:
                    logger.error(f"Model not found: {model_id}")
                    return None
                
                # Load model data
                model_data = self.storage.load_model_data(model_id)
                if not model_data:
                    logger.error(f"Failed to load model data: {model_id}")
                    return None
                
                # Parse model based on type
                model_object = self._parse_model_data(model_data, model_info)
                if model_object is None:
                    logger.error(f"Failed to parse model: {model_id}")
                    return None
                
                # Cache the loaded model
                self._loaded_models[model_id] = model_object
                self._update_load_stats(model_id, cache_hit=False)
                
                logger.info(f"Successfully loaded model: {model_id}")
                return model_object
                
        except Exception as e:
            logger.error(f"Failed to load model {model_id}: {e}")
            return None
    
    def unload_model(self, model_id: str) -> bool:
        """Unload a model from cache."""
        try:
            with self._lock:
                if model_id in self._loaded_models:
                    del self._loaded_models[model_id]
                    logger.info(f"Unloaded model: {model_id}")
                    return True
                return False
                
        except Exception as e:
            logger.error(f"Failed to unload model {model_id}: {e}")
            return False
    
    def register_model_from_file(
        self, 
        model_id: str, 
        file_path: str,
        model_type: str = "wasm",
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Register a model from a file path."""
        try:
            file_path = Path(file_path)
            if not file_path.exists():
                logger.error(f"Model file not found: {file_path}")
                return False
            
            # Store in storage system
            stored_path = self.storage.store_model(
                model_id=model_id,
                model_data=str(file_path),
                model_type=model_type,
                metadata=metadata
            )
            
            if stored_path:
                logger.info(f"Registered model {model_id} from {file_path}")
                return True
            else:
                return False
                
        except Exception as e:
            logger.error(f"Failed to register model from file: {e}")
            return False
    
    def register_model_from_bytes(
        self,
        model_id: str,
        model_data: bytes,
        model_type: str = "wasm",
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Register a model from raw bytes."""
        try:
            stored_path = self.storage.store_model(
                model_id=model_id,
                model_data=model_data,
                model_type=model_type,
                metadata=metadata
            )
            
            if stored_path:
                logger.info(f"Registered model {model_id} from bytes ({len(model_data)} bytes)")
                return True
            else:
                return False
                
        except Exception as e:
            logger.error(f"Failed to register model from bytes: {e}")
            return False
    
    def list_registered_models(self) -> List[str]:
        """List all registered models."""
        return self.storage.list_models()
    
    def list_loaded_models(self) -> List[str]:
        """List all currently loaded (cached) models."""
        with self._lock:
            return list(self._loaded_models.keys())
    
    def get_model_info(self, model_id: str) -> Optional[Dict[str, Any]]:
        """Get information about a model."""
        model_info = self.storage.get_model_info(model_id)
        if not model_info:
            return None
        
        info_dict = model_info.to_dict()
        
        # Add loading statistics
        if model_id in self._load_stats:
            info_dict['load_stats'] = self._load_stats[model_id]
        
        # Add cache status
        info_dict['is_cached'] = model_id in self._loaded_models
        
        return info_dict
    
    def get_loader_statistics(self) -> Dict[str, Any]:
        """Get loader statistics."""
        with self._lock:
            total_models = len(self.storage.list_models())
            loaded_models = len(self._loaded_models)
            
            total_loads = sum(
                stats['load_count'] for stats in self._load_stats.values()
            )
            total_cache_hits = sum(
                stats['cache_hits'] for stats in self._load_stats.values()
            )
            
            cache_hit_rate = (
                total_cache_hits / total_loads if total_loads > 0 else 0.0
            )
            
            return {
                'total_models': total_models,
                'loaded_models': loaded_models,
                'total_loads': total_loads,
                'cache_hits': total_cache_hits,
                'cache_hit_rate': cache_hit_rate,
                'memory_usage_models': loaded_models
            }
    
    def cleanup_cache(self, max_age_seconds: float = 3600) -> int:
        """
        Clean up old models from cache.
        
        Args:
            max_age_seconds: Maximum age for cached models
            
        Returns:
            Number of models removed from cache
        """
        removed_count = 0
        current_time = time.time()
        
        try:
            with self._lock:
                models_to_remove = []
                
                for model_id in self._loaded_models.keys():
                    if model_id in self._load_stats:
                        last_access = self._load_stats[model_id].get('last_access', 0)
                        if current_time - last_access > max_age_seconds:
                            models_to_remove.append(model_id)
                
                for model_id in models_to_remove:
                    self.unload_model(model_id)
                    removed_count += 1
                
                if removed_count > 0:
                    logger.info(f"Cleaned up {removed_count} cached models")
                
                return removed_count
                
        except Exception as e:
            logger.error(f"Error during cache cleanup: {e}")
            return 0
    
    def _parse_model_data(self, model_data: bytes, model_info: ModelInfo) -> Optional[Any]:
        """
        Parse model data into a model object.
        This is a simplified implementation for Generation 1.
        """
        try:
            model_type = model_info.model_type.lower()
            
            if model_type == 'wasm':
                # For WASM models, return the raw bytes with metadata
                return {
                    'type': 'wasm',
                    'data': model_data,
                    'size': len(model_data),
                    'metadata': model_info.metadata
                }
            
            elif model_type == 'json':
                # For JSON models, parse the JSON
                json_str = model_data.decode('utf-8')
                parsed_data = json.loads(json_str)
                return {
                    'type': 'json',
                    'data': parsed_data,
                    'metadata': model_info.metadata
                }
            
            elif model_type in ['bin', 'binary']:
                # For binary models, return structured data
                return {
                    'type': 'binary',
                    'data': model_data,
                    'size': len(model_data),
                    'metadata': model_info.metadata
                }
            
            else:
                # Default: return raw data
                return {
                    'type': model_type,
                    'data': model_data,
                    'size': len(model_data),
                    'metadata': model_info.metadata
                }
                
        except Exception as e:
            logger.error(f"Failed to parse model data: {e}")
            return None
    
    def _update_load_stats(self, model_id: str, cache_hit: bool) -> None:
        """Update loading statistics for a model."""
        current_time = time.time()
        
        if model_id not in self._load_stats:
            self._load_stats[model_id] = {
                'load_count': 0,
                'cache_hits': 0,
                'first_load': current_time,
                'last_access': current_time
            }
        
        stats = self._load_stats[model_id]
        stats['load_count'] += 1
        stats['last_access'] = current_time
        
        if cache_hit:
            stats['cache_hits'] += 1


# Utility functions for testing and demonstration
async def demo_basic_model_loader():
    """Demonstration of the basic model loader."""
    loader = BasicModelLoader()
    
    try:
        print("Basic Model Loader Demo")
        print("=" * 50)
        
        # Create some test model data
        test_models = {
            'simple_classifier': {
                'data': json.dumps({'weights': [1, 2, 3], 'bias': 0.5}).encode(),
                'type': 'json'
            },
            'binary_model': {
                'data': b'\x00\x01\x02\x03\x04\x05',
                'type': 'bin'
            },
            'wasm_model': {
                'data': b'\x00asm\x01\x00\x00\x00',  # WASM magic bytes
                'type': 'wasm'
            }
        }
        
        # Register test models
        print("Registering test models...")
        for model_id, info in test_models.items():
            success = loader.register_model_from_bytes(
                model_id=model_id,
                model_data=info['data'],
                model_type=info['type'],
                metadata={'demo': True, 'version': '1.0'}
            )
            print(f"Registered {model_id}: {'✓' if success else '✗'}")
        
        # List registered models
        print(f"\nRegistered models: {loader.list_registered_models()}")
        
        # Load models
        print("\nLoading models...")
        for model_id in test_models.keys():
            model = loader.load_model(model_id)
            if model and isinstance(model, dict):
                size = model.get('size', len(model.get('data', b'')))
                print(f"Loaded {model_id}: {model.get('type', 'unknown')} ({size} bytes)")
            else:
                print(f"Failed to load {model_id}")
        
        # Test cache hit
        print("\nTesting cache (should be faster)...")
        model = loader.load_model('simple_classifier')
        if model:
            print(f"Cache hit for simple_classifier: {model['type']}")
        
        # Show statistics
        stats = loader.get_loader_statistics()
        print(f"\nLoader Statistics:")
        print(f"Total models: {stats['total_models']}")
        print(f"Loaded models: {stats['loaded_models']}")
        print(f"Total loads: {stats['total_loads']}")
        print(f"Cache hit rate: {stats['cache_hit_rate']:.2%}")
        
        # Show detailed model info
        print(f"\nDetailed model info:")
        for model_id in loader.list_registered_models():
            info = loader.get_model_info(model_id)
            if info:
                print(f"{model_id}: {info['size_bytes']} bytes, "
                      f"accessed {info['access_count']} times")
        
    except Exception as e:
        print(f"Demo error: {e}")


# Mock implementations for when PyTorch is not available
class MockExporter:
    """Mock model exporter for testing without PyTorch."""
    
    def export_to_wasm(self, model, output_path, example_input=None, **kwargs):
        """Mock WASM export implementation."""
        try:
            from .mock_torch import torch
            
            # Create a simple mock WASM file
            output_path = Path(output_path)
            
            # Mock WASM file content (simplified)
            mock_wasm_content = b'\x00asm\x01\x00\x00\x00'  # WASM magic bytes
            mock_wasm_content += b'\x00' * 100  # Dummy content
            
            output_path.write_bytes(mock_wasm_content)
            
            # Also create a JSON metadata file
            metadata = {
                "model_info": {
                    "input_shape": getattr(example_input, 'shape', [1, 10]) if example_input else [1, 10],
                    "input_dtype": "float32"
                },
                "graph": {
                    "operations": [
                        {"kind": "aten::linear", "attributes": {}},
                        {"kind": "aten::relu", "attributes": {}}
                    ],
                    "parameters": {}
                },
                "optimization": {
                    "use_simd": kwargs.get('use_simd', True),
                    "use_threads": kwargs.get('use_threads', True)
                },
                "metadata": {
                    "torch_version": "mock-2.4.0",
                    "export_version": "0.1.0"
                }
            }
            
            metadata_path = output_path.with_suffix('.json')
            metadata_path.write_text(json.dumps(metadata, indent=2))
            
            logger.info(f"Mock WASM export completed: {output_path}")
            return str(output_path)
            
        except Exception as e:
            logger.error(f"Mock WASM export failed: {e}")
            raise RuntimeError(f"Mock export failed: {e}")


class MockWASMRuntime:
    """Mock WASM runtime for testing without PyTorch."""
    
    def __init__(self, simd=True, threads=4, memory_limit_mb=1024, enable_monitoring=True, **kwargs):
        self.simd = simd
        self.threads = threads 
        self.memory_limit_mb = memory_limit_mb
        self.enable_monitoring = enable_monitoring
        self._initialized = False
        self._models = {}
        self._stats = {
            'inference_count': 0,
            'total_inference_time': 0.0,
            'memory_peak_mb': 0.0
        }
        
    async def init(self):
        """Initialize the mock runtime."""
        self._initialized = True
        logger.info("Mock WASM runtime initialized")
        return self
        
    async def load_model(self, model_path):
        """Load a mock model."""
        if not self._initialized:
            await self.init()
            
        model_path = Path(model_path)
        model_id = model_path.stem
        
        # Create mock model
        mock_model = MockWASMModel(model_path, self)
        await mock_model._load_model_data()
        
        self._models[model_id] = mock_model
        logger.info(f"Mock model loaded: {model_id}")
        return mock_model
        
    async def cleanup(self):
        """Cleanup mock runtime."""
        self._models.clear()
        logger.info("Mock WASM runtime cleaned up")
        
    def get_runtime_stats(self):
        """Get mock runtime statistics."""
        return {
            'uptime_seconds': 100.0,
            'inference_count': self._stats['inference_count'],
            'total_inference_time': self._stats['total_inference_time'],
            'average_inference_time': 0.05,
            'memory_peak_mb': self._stats['memory_peak_mb'],
            'health_status': 'healthy'
        }


class MockWASMModel:
    """Mock WASM model for testing."""
    
    def __init__(self, model_path, runtime):
        self.model_path = Path(model_path)
        self.runtime = runtime
        self._is_loaded = False
        
    async def _load_model_data(self):
        """Mock model data loading."""
        self._is_loaded = True
        
    async def forward(self, input_tensor):
        """Mock forward pass."""
        from .mock_torch import MockTensor
        
        if not self._is_loaded:
            raise RuntimeError("Model not loaded")
            
        # Update stats
        self.runtime._stats['inference_count'] += 1
        self.runtime._stats['total_inference_time'] += 0.05
        
        # Return mock output tensor
        if hasattr(input_tensor, 'data'):
            output_data = [x * 0.5 + 0.1 for x in input_tensor.data]
        else:
            output_data = [0.5, 0.3, 0.2]  # Default output
            
        return MockTensor(output_data)


class MockOptimizer:
    """Mock optimizer for testing without PyTorch."""
    
    def optimize_for_browser(self, model, **kwargs):
        """Mock browser optimization."""
        logger.info("Mock browser optimization applied")
        return model  # Return unchanged
        
    def quantize_for_wasm(self, model, **kwargs):
        """Mock WASM quantization.""" 
        logger.info("Mock WASM quantization applied")
        return model  # Return unchanged


if __name__ == "__main__":
    import asyncio
    asyncio.run(demo_basic_model_loader())