"""Global model registry with comprehensive versioning and metadata management.

This module implements a sophisticated model registry system that provides
versioning, dependency management, and distributed deployment capabilities
for WASM PyTorch models.
"""

import hashlib
import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union
import uuid

import torch
import torch.nn as nn

from ..security import log_security_event, validate_path
from ..validation import validate_tensor_safe


logger = logging.getLogger(__name__)


class ModelStatus(Enum):
    """Model status in registry."""
    
    DRAFT = "draft"
    TESTING = "testing"
    STABLE = "stable"
    DEPRECATED = "deprecated"
    ARCHIVED = "archived"


class License(Enum):
    """Supported model licenses."""
    
    MIT = "mit"
    APACHE_2_0 = "apache-2.0"
    BSD_3_CLAUSE = "bsd-3-clause"
    GPL_V3 = "gpl-v3"
    CREATIVE_COMMONS = "cc-by-4.0"
    PROPRIETARY = "proprietary"
    CUSTOM = "custom"


@dataclass
class ModelMetadata:
    """Comprehensive model metadata."""
    
    # Basic information
    model_id: str
    name: str
    description: str
    version: str
    author: str
    organization: Optional[str] = None
    
    # Technical specifications
    architecture: str = "unknown"
    framework_version: str = "torch>=2.4.0"
    input_shape: Tuple[int, ...] = ()
    output_shape: Tuple[int, ...] = ()
    parameter_count: int = 0
    model_size_mb: float = 0.0
    
    # Performance metrics
    accuracy_metrics: Dict[str, float] = field(default_factory=dict)
    benchmark_results: Dict[str, float] = field(default_factory=dict)
    hardware_requirements: Dict[str, Any] = field(default_factory=dict)
    
    # Deployment information
    supported_backends: List[str] = field(default_factory=list)
    optimization_flags: Dict[str, Any] = field(default_factory=dict)
    quantization_info: Optional[Dict[str, Any]] = None
    
    # Legal and compliance
    license: License = License.MIT
    license_url: Optional[str] = None
    ethical_considerations: List[str] = field(default_factory=list)
    data_sources: List[str] = field(default_factory=list)
    
    # Timestamps and tracking
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    download_count: int = 0
    rating: float = 0.0
    
    # Dependencies and relationships
    dependencies: List[str] = field(default_factory=list)
    parent_model: Optional[str] = None
    derived_models: List[str] = field(default_factory=list)
    
    # Tags and categorization
    tags: Set[str] = field(default_factory=set)
    categories: List[str] = field(default_factory=list)
    use_cases: List[str] = field(default_factory=list)


@dataclass
class VersionInfo:
    """Version information with semantic versioning."""
    
    version: str
    major: int
    minor: int
    patch: int
    pre_release: Optional[str] = None
    build_metadata: Optional[str] = None
    
    # Version-specific information
    changelog: str = ""
    compatibility_notes: List[str] = field(default_factory=list)
    breaking_changes: List[str] = field(default_factory=list)
    migration_guide: Optional[str] = None
    
    # Technical details
    checksum: str = ""
    file_size_bytes: int = 0
    build_info: Dict[str, str] = field(default_factory=dict)
    
    # Status and availability
    status: ModelStatus = ModelStatus.DRAFT
    is_latest: bool = False
    published_at: Optional[float] = None
    deprecated_at: Optional[float] = None


@dataclass
class ModelEntry:
    """Complete model registry entry."""
    
    metadata: ModelMetadata
    versions: Dict[str, VersionInfo] = field(default_factory=dict)
    latest_version: Optional[str] = None
    download_urls: Dict[str, str] = field(default_factory=dict)
    verification_info: Dict[str, Any] = field(default_factory=dict)
    access_control: Dict[str, Any] = field(default_factory=dict)


class ModelRegistry:
    """Global model registry with comprehensive management capabilities."""
    
    def __init__(self, registry_path: Optional[Path] = None):
        self.registry_path = registry_path or Path.cwd() / ".model_registry"
        self.registry_path.mkdir(exist_ok=True)
        
        # Core data structures
        self.models: Dict[str, ModelEntry] = {}
        self.model_index: Dict[str, Set[str]] = {}  # tag -> model_ids
        self.category_index: Dict[str, Set[str]] = {}  # category -> model_ids
        self.author_index: Dict[str, Set[str]] = {}  # author -> model_ids
        
        # Registry metadata
        self.registry_info = {
            "version": "1.0.0",
            "created_at": time.time(),
            "last_updated": time.time(),
            "total_models": 0,
            "total_downloads": 0,
        }
        
        # Load existing registry
        self._load_registry()
    
    def _load_registry(self) -> None:
        """Load existing registry from disk."""
        
        registry_file = self.registry_path / "registry.json"
        
        try:
            if registry_file.exists():
                with open(registry_file, "r") as f:
                    data = json.load(f)
                
                # Load registry info
                self.registry_info.update(data.get("registry_info", {}))
                
                # Load models
                for model_id, model_data in data.get("models", {}).items():
                    entry = self._deserialize_model_entry(model_data)
                    self.models[model_id] = entry
                    self._update_indices(model_id, entry.metadata)
                
                logger.info(f"Loaded registry with {len(self.models)} models")
            
        except Exception as e:
            logger.error(f"Failed to load registry: {e}")
            self.models = {}
    
    def _save_registry(self) -> None:
        """Save registry to disk."""
        
        registry_file = self.registry_path / "registry.json"
        
        try:
            # Update registry info
            self.registry_info["last_updated"] = time.time()
            self.registry_info["total_models"] = len(self.models)
            self.registry_info["total_downloads"] = sum(
                entry.metadata.download_count for entry in self.models.values()
            )
            
            # Serialize data
            data = {
                "registry_info": self.registry_info,
                "models": {
                    model_id: self._serialize_model_entry(entry)
                    for model_id, entry in self.models.items()
                },
            }
            
            with open(registry_file, "w") as f:
                json.dump(data, f, indent=2, default=str)
            
            logger.debug(f"Saved registry with {len(self.models)} models")
            
        except Exception as e:
            logger.error(f"Failed to save registry: {e}")
    
    def register_model(
        self,
        model: nn.Module,
        metadata: ModelMetadata,
        model_path: Path,
        replace_existing: bool = False
    ) -> bool:
        """Register new model in the registry."""
        
        if metadata.model_id in self.models and not replace_existing:
            logger.warning(f"Model {metadata.model_id} already exists")
            return False
        
        try:
            # Validate model and compute metadata
            self._validate_model(model, metadata)
            enhanced_metadata = self._enhance_metadata(model, metadata, model_path)
            
            # Create initial version
            version_info = VersionInfo(
                version=enhanced_metadata.version,
                major=1, minor=0, patch=0,
                changelog="Initial model registration",
                status=ModelStatus.STABLE,
                is_latest=True,
                published_at=time.time(),
                checksum=self._compute_model_checksum(model_path),
                file_size_bytes=model_path.stat().st_size if model_path.exists() else 0,
            )
            
            # Create model entry
            model_entry = ModelEntry(
                metadata=enhanced_metadata,
                versions={enhanced_metadata.version: version_info},
                latest_version=enhanced_metadata.version,
                download_urls={enhanced_metadata.version: str(model_path)},
                verification_info={
                    "checksum": version_info.checksum,
                    "verified_at": time.time(),
                    "verification_status": "passed",
                },
            )
            
            # Add to registry
            self.models[metadata.model_id] = model_entry
            self._update_indices(metadata.model_id, enhanced_metadata)
            
            # Save registry
            self._save_registry()
            
            log_security_event("model_registered", {
                "model_id": metadata.model_id,
                "version": enhanced_metadata.version,
                "author": enhanced_metadata.author,
                "size_mb": enhanced_metadata.model_size_mb,
            })
            
            logger.info(f"Registered model {metadata.model_id} v{enhanced_metadata.version}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to register model {metadata.model_id}: {e}")
            return False
    
    def _validate_model(self, model: nn.Module, metadata: ModelMetadata) -> None:
        """Validate model and metadata."""
        
        # Basic model validation
        if not isinstance(model, nn.Module):
            raise ValueError("Invalid model: must be PyTorch nn.Module")
        
        # Metadata validation
        if not metadata.model_id or not metadata.name:
            raise ValueError("Model ID and name are required")
        
        if not metadata.version:
            raise ValueError("Model version is required")
        
        # Test model forward pass if input shape provided
        if metadata.input_shape:
            try:
                test_input = torch.randn(1, *metadata.input_shape)
                with torch.no_grad():
                    output = model(test_input)
                validate_tensor_safe(output, "model_output")
                
                # Update output shape if not provided
                if not metadata.output_shape:
                    metadata.output_shape = tuple(output.shape[1:])  # Exclude batch dimension
                    
            except Exception as e:
                logger.warning(f"Model validation test failed: {e}")
    
    def _enhance_metadata(
        self, 
        model: nn.Module, 
        metadata: ModelMetadata,
        model_path: Path
    ) -> ModelMetadata:
        """Enhance metadata with computed information."""
        
        # Compute parameter count
        if metadata.parameter_count == 0:
            metadata.parameter_count = sum(p.numel() for p in model.parameters())
        
        # Compute model size
        if metadata.model_size_mb == 0.0 and model_path.exists():
            metadata.model_size_mb = model_path.stat().st_size / (1024 * 1024)
        
        # Add default supported backends
        if not metadata.supported_backends:
            metadata.supported_backends = ["wasm", "cpu"]
        
        # Set framework version
        if metadata.framework_version == "torch>=2.4.0":
            metadata.framework_version = f"torch=={torch.__version__}"
        
        # Generate unique model ID if needed
        if not metadata.model_id:
            metadata.model_id = str(uuid.uuid4())
        
        # Update timestamps
        metadata.updated_at = time.time()
        
        return metadata
    
    def _compute_model_checksum(self, model_path: Path) -> str:
        """Compute model file checksum."""
        
        if not model_path.exists():
            return ""
        
        sha256_hash = hashlib.sha256()
        with open(model_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                sha256_hash.update(chunk)
        
        return sha256_hash.hexdigest()
    
    def _update_indices(self, model_id: str, metadata: ModelMetadata) -> None:
        """Update search indices."""
        
        # Tag index
        for tag in metadata.tags:
            if tag not in self.model_index:
                self.model_index[tag] = set()
            self.model_index[tag].add(model_id)
        
        # Category index
        for category in metadata.categories:
            if category not in self.category_index:
                self.category_index[category] = set()
            self.category_index[category].add(model_id)
        
        # Author index
        if metadata.author not in self.author_index:
            self.author_index[metadata.author] = set()
        self.author_index[metadata.author].add(model_id)
    
    def add_model_version(
        self,
        model_id: str,
        version: str,
        model_path: Path,
        changelog: str = "",
        breaking_changes: Optional[List[str]] = None
    ) -> bool:
        """Add new version of existing model."""
        
        if model_id not in self.models:
            logger.error(f"Model {model_id} not found")
            return False
        
        if version in self.models[model_id].versions:
            logger.warning(f"Version {version} already exists for model {model_id}")
            return False
        
        try:
            # Parse version
            major, minor, patch = self._parse_version(version)
            
            # Create version info
            version_info = VersionInfo(
                version=version,
                major=major, minor=minor, patch=patch,
                changelog=changelog,
                breaking_changes=breaking_changes or [],
                status=ModelStatus.TESTING,  # New versions start as testing
                is_latest=False,  # Will be updated if this becomes latest
                published_at=time.time(),
                checksum=self._compute_model_checksum(model_path),
                file_size_bytes=model_path.stat().st_size if model_path.exists() else 0,
            )
            
            # Add version
            self.models[model_id].versions[version] = version_info
            self.models[model_id].download_urls[version] = str(model_path)
            
            # Update latest version if this is newer
            current_latest = self.models[model_id].latest_version
            if self._is_newer_version(version, current_latest):
                # Mark old latest as not latest
                if current_latest and current_latest in self.models[model_id].versions:
                    self.models[model_id].versions[current_latest].is_latest = False
                
                # Set new latest
                self.models[model_id].latest_version = version
                version_info.is_latest = True
            
            # Update model metadata timestamp
            self.models[model_id].metadata.updated_at = time.time()
            
            # Save registry
            self._save_registry()
            
            log_security_event("model_version_added", {
                "model_id": model_id,
                "version": version,
                "is_latest": version_info.is_latest,
            })
            
            logger.info(f"Added version {version} for model {model_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to add version {version} for model {model_id}: {e}")
            return False
    
    def _parse_version(self, version: str) -> Tuple[int, int, int]:
        """Parse semantic version string."""
        
        parts = version.split('.')
        if len(parts) < 3:
            raise ValueError(f"Invalid version format: {version}")
        
        try:
            major = int(parts[0])
            minor = int(parts[1])
            patch = int(parts[2])
            return major, minor, patch
        except ValueError:
            raise ValueError(f"Invalid version numbers in: {version}")
    
    def _is_newer_version(self, version1: str, version2: Optional[str]) -> bool:
        """Check if version1 is newer than version2."""
        
        if not version2:
            return True
        
        try:
            v1_parts = self._parse_version(version1)
            v2_parts = self._parse_version(version2)
            
            return v1_parts > v2_parts
        except ValueError:
            return False
    
    def search_models(
        self,
        query: Optional[str] = None,
        tags: Optional[List[str]] = None,
        categories: Optional[List[str]] = None,
        author: Optional[str] = None,
        min_rating: float = 0.0,
        limit: int = 50
    ) -> List[ModelEntry]:
        """Search models in registry."""
        
        candidate_ids = set(self.models.keys())
        
        # Filter by tags
        if tags:
            tag_matches = set()
            for tag in tags:
                if tag in self.model_index:
                    tag_matches.update(self.model_index[tag])
            candidate_ids = candidate_ids.intersection(tag_matches)
        
        # Filter by categories
        if categories:
            category_matches = set()
            for category in categories:
                if category in self.category_index:
                    category_matches.update(self.category_index[category])
            candidate_ids = candidate_ids.intersection(category_matches)
        
        # Filter by author
        if author:
            if author in self.author_index:
                candidate_ids = candidate_ids.intersection(self.author_index[author])
            else:
                candidate_ids = set()
        
        # Apply filters and collect results
        results = []
        
        for model_id in candidate_ids:
            entry = self.models[model_id]
            
            # Rating filter
            if entry.metadata.rating < min_rating:
                continue
            
            # Text query filter (basic implementation)
            if query:
                query_lower = query.lower()
                searchable_text = (
                    entry.metadata.name.lower() + " " +
                    entry.metadata.description.lower() + " " +
                    " ".join(entry.metadata.tags).lower()
                )
                
                if query_lower not in searchable_text:
                    continue
            
            results.append(entry)
        
        # Sort by relevance (rating * download_count)
        results.sort(
            key=lambda x: x.metadata.rating * (1 + x.metadata.download_count),
            reverse=True
        )
        
        return results[:limit]
    
    def get_model(self, model_id: str, version: Optional[str] = None) -> Optional[ModelEntry]:
        """Get specific model from registry."""
        
        if model_id not in self.models:
            return None
        
        entry = self.models[model_id]
        
        # Return copy to prevent external modification
        return ModelEntry(
            metadata=entry.metadata,
            versions=entry.versions.copy(),
            latest_version=entry.latest_version,
            download_urls=entry.download_urls.copy(),
            verification_info=entry.verification_info.copy(),
        )
    
    def download_model(self, model_id: str, version: Optional[str] = None) -> Optional[Path]:
        """Get download path for model."""
        
        if model_id not in self.models:
            logger.error(f"Model {model_id} not found")
            return None
        
        entry = self.models[model_id]
        
        # Determine version to download
        target_version = version or entry.latest_version
        if not target_version or target_version not in entry.versions:
            logger.error(f"Version {target_version} not found for model {model_id}")
            return None
        
        # Get download URL
        if target_version not in entry.download_urls:
            logger.error(f"No download URL for model {model_id} v{target_version}")
            return None
        
        download_path = Path(entry.download_urls[target_version])
        
        # Verify file exists and checksum
        if not download_path.exists():
            logger.error(f"Model file not found: {download_path}")
            return None
        
        version_info = entry.versions[target_version]
        if version_info.checksum:
            actual_checksum = self._compute_model_checksum(download_path)
            if actual_checksum != version_info.checksum:
                logger.error(f"Checksum mismatch for model {model_id} v{target_version}")
                return None
        
        # Update download count
        entry.metadata.download_count += 1
        entry.metadata.updated_at = time.time()
        self._save_registry()
        
        log_security_event("model_downloaded", {
            "model_id": model_id,
            "version": target_version,
            "download_count": entry.metadata.download_count,
        })
        
        logger.info(f"Downloaded model {model_id} v{target_version}")
        return download_path
    
    def update_model_status(
        self, 
        model_id: str, 
        version: str, 
        status: ModelStatus
    ) -> bool:
        """Update model version status."""
        
        if model_id not in self.models:
            logger.error(f"Model {model_id} not found")
            return False
        
        if version not in self.models[model_id].versions:
            logger.error(f"Version {version} not found for model {model_id}")
            return False
        
        old_status = self.models[model_id].versions[version].status
        self.models[model_id].versions[version].status = status
        
        # Handle deprecation
        if status == ModelStatus.DEPRECATED:
            self.models[model_id].versions[version].deprecated_at = time.time()
        
        self.models[model_id].metadata.updated_at = time.time()
        self._save_registry()
        
        log_security_event("model_status_updated", {
            "model_id": model_id,
            "version": version,
            "old_status": old_status.value,
            "new_status": status.value,
        })
        
        logger.info(f"Updated model {model_id} v{version} status: {old_status.value} -> {status.value}")
        return True
    
    def delete_model(self, model_id: str, confirm: bool = False) -> bool:
        """Delete model from registry."""
        
        if not confirm:
            logger.warning("Model deletion requires explicit confirmation")
            return False
        
        if model_id not in self.models:
            logger.error(f"Model {model_id} not found")
            return False
        
        try:
            # Remove from indices
            entry = self.models[model_id]
            metadata = entry.metadata
            
            for tag in metadata.tags:
                if tag in self.model_index:
                    self.model_index[tag].discard(model_id)
                    if not self.model_index[tag]:
                        del self.model_index[tag]
            
            for category in metadata.categories:
                if category in self.category_index:
                    self.category_index[category].discard(model_id)
                    if not self.category_index[category]:
                        del self.category_index[category]
            
            if metadata.author in self.author_index:
                self.author_index[metadata.author].discard(model_id)
                if not self.author_index[metadata.author]:
                    del self.author_index[metadata.author]
            
            # Remove from models
            del self.models[model_id]
            
            # Save registry
            self._save_registry()
            
            log_security_event("model_deleted", {
                "model_id": model_id,
                "author": metadata.author,
            })
            
            logger.info(f"Deleted model {model_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete model {model_id}: {e}")
            return False
    
    def export_registry_report(self) -> Dict[str, Any]:
        """Export comprehensive registry report."""
        
        # Compute statistics
        total_models = len(self.models)
        total_versions = sum(len(entry.versions) for entry in self.models.values())
        total_downloads = sum(entry.metadata.download_count for entry in self.models.values())
        
        # Status distribution
        status_distribution = {}
        for entry in self.models.values():
            for version_info in entry.versions.values():
                status = version_info.status.value
                status_distribution[status] = status_distribution.get(status, 0) + 1
        
        # Category distribution
        category_distribution = {}
        for entry in self.models.values():
            for category in entry.metadata.categories:
                category_distribution[category] = category_distribution.get(category, 0) + 1
        
        # Author statistics
        author_stats = {}
        for entry in self.models.values():
            author = entry.metadata.author
            if author not in author_stats:
                author_stats[author] = {
                    "models": 0,
                    "total_downloads": 0,
                    "avg_rating": 0.0,
                }
            
            author_stats[author]["models"] += 1
            author_stats[author]["total_downloads"] += entry.metadata.download_count
            author_stats[author]["avg_rating"] += entry.metadata.rating
        
        # Compute average ratings
        for stats in author_stats.values():
            if stats["models"] > 0:
                stats["avg_rating"] /= stats["models"]
        
        # Top models
        top_models = sorted(
            self.models.values(),
            key=lambda x: x.metadata.download_count,
            reverse=True
        )[:10]
        
        return {
            "registry_info": self.registry_info,
            "statistics": {
                "total_models": total_models,
                "total_versions": total_versions,
                "total_downloads": total_downloads,
                "avg_downloads_per_model": total_downloads / max(total_models, 1),
            },
            "distributions": {
                "status": status_distribution,
                "categories": category_distribution,
            },
            "author_statistics": author_stats,
            "top_models": [
                {
                    "model_id": entry.metadata.model_id,
                    "name": entry.metadata.name,
                    "author": entry.metadata.author,
                    "downloads": entry.metadata.download_count,
                    "rating": entry.metadata.rating,
                }
                for entry in top_models
            ],
            "recent_activity": self._get_recent_activity(),
            "timestamp": time.time(),
        }
    
    def _get_recent_activity(self) -> List[Dict[str, Any]]:
        """Get recent registry activity."""
        
        activity = []
        current_time = time.time()
        
        for entry in self.models.values():
            # Recent registrations (last 30 days)
            if (current_time - entry.metadata.created_at) < (30 * 24 * 3600):
                activity.append({
                    "type": "registration",
                    "model_id": entry.metadata.model_id,
                    "timestamp": entry.metadata.created_at,
                })
            
            # Recent version updates
            for version, version_info in entry.versions.items():
                if (version_info.published_at and 
                    (current_time - version_info.published_at) < (30 * 24 * 3600)):
                    activity.append({
                        "type": "version_update",
                        "model_id": entry.metadata.model_id,
                        "version": version,
                        "timestamp": version_info.published_at,
                    })
        
        # Sort by timestamp (most recent first)
        activity.sort(key=lambda x: x["timestamp"], reverse=True)
        
        return activity[:20]  # Return last 20 activities
    
    def _serialize_model_entry(self, entry: ModelEntry) -> Dict[str, Any]:
        """Serialize model entry for storage."""
        
        return {
            "metadata": {
                k: (list(v) if isinstance(v, set) else v)
                for k, v in entry.metadata.__dict__.items()
            },
            "versions": {
                v: version_info.__dict__ 
                for v, version_info in entry.versions.items()
            },
            "latest_version": entry.latest_version,
            "download_urls": entry.download_urls,
            "verification_info": entry.verification_info,
            "access_control": entry.access_control,
        }
    
    def _deserialize_model_entry(self, data: Dict[str, Any]) -> ModelEntry:
        """Deserialize model entry from storage."""
        
        # Reconstruct metadata
        metadata_data = data["metadata"]
        if "tags" in metadata_data and isinstance(metadata_data["tags"], list):
            metadata_data["tags"] = set(metadata_data["tags"])
        
        metadata = ModelMetadata(**metadata_data)
        
        # Reconstruct versions
        versions = {}
        for version, version_data in data["versions"].items():
            version_data["status"] = ModelStatus(version_data["status"])
            versions[version] = VersionInfo(**version_data)
        
        return ModelEntry(
            metadata=metadata,
            versions=versions,
            latest_version=data.get("latest_version"),
            download_urls=data.get("download_urls", {}),
            verification_info=data.get("verification_info", {}),
            access_control=data.get("access_control", {}),
        )
    
    def get_registry_statistics(self) -> Dict[str, Any]:
        """Get real-time registry statistics."""
        
        return {
            "total_models": len(self.models),
            "total_versions": sum(len(entry.versions) for entry in self.models.values()),
            "memory_usage_mb": len(str(self.models)) / (1024 * 1024),  # Rough estimate
            "indices": {
                "tags": len(self.model_index),
                "categories": len(self.category_index),
                "authors": len(self.author_index),
            },
            "last_updated": self.registry_info["last_updated"],
        }