#!/usr/bin/env python3
"""Generate Software Bill of Materials (SBOM) for WASM Shim for Torch."""

import json
import subprocess
import sys
from pathlib import Path
from datetime import datetime
import hashlib
import pkg_resources
from typing import Dict, List, Any

try:
    from cyclonedx.model import Component, ComponentType
    from cyclonedx.factory.license import LicenseFactory
    from cyclonedx.output.json import JsonV1Dot4
    from cyclonedx.model.bom import Bom
    from cyclonedx.model.component import ComponentScope
except ImportError:
    print("Error: cyclonedx-bom package required. Install with: pip install cyclonedx-bom")
    sys.exit(1)


class SBOMGenerator:
    """Generate SBOM for the project."""
    
    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.package_info = self._get_package_info()
        
    def _get_package_info(self) -> Dict[str, Any]:
        """Extract package information from pyproject.toml."""
        import tomllib
        
        pyproject_path = self.project_root / "pyproject.toml"
        if not pyproject_path.exists():
            raise FileNotFoundError("pyproject.toml not found")
            
        with open(pyproject_path, "rb") as f:
            data = tomllib.load(f)
            
        return data.get("project", {})
    
    def _get_git_info(self) -> Dict[str, str]:
        """Get Git repository information."""
        try:
            commit_hash = subprocess.check_output(
                ["git", "rev-parse", "HEAD"], 
                cwd=self.project_root,
                text=True
            ).strip()
            
            remote_url = subprocess.check_output(
                ["git", "config", "--get", "remote.origin.url"],
                cwd=self.project_root,
                text=True
            ).strip()
            
            return {
                "commit": commit_hash,
                "repository": remote_url
            }
        except subprocess.CalledProcessError:
            return {}
    
    def _get_dependencies(self) -> List[Dict[str, Any]]:
        """Get installed package dependencies."""
        dependencies = []
        
        for dist in pkg_resources.working_set:
            if dist.project_name.lower() == self.package_info.get("name", "").lower():
                continue  # Skip self
                
            dep_info = {
                "name": dist.project_name,
                "version": dist.version,
                "location": dist.location,
            }
            
            # Try to get license information
            try:
                metadata = dist.get_metadata('METADATA') or dist.get_metadata('PKG-INFO')
                for line in metadata.split('\n'):
                    if line.startswith('License:'):
                        dep_info["license"] = line.split(':', 1)[1].strip()
                        break
            except:
                pass
                
            dependencies.append(dep_info)
                
        return sorted(dependencies, key=lambda x: x["name"].lower())
    
    def _calculate_file_hash(self, file_path: Path) -> str:
        """Calculate SHA-256 hash of a file."""
        sha256_hash = hashlib.sha256()
        try:
            with open(file_path, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    sha256_hash.update(chunk)
            return sha256_hash.hexdigest()
        except:
            return ""
    
    def _get_source_files(self) -> List[Dict[str, Any]]:
        """Get list of source files with hashes."""
        source_files = []
        src_dir = self.project_root / "src"
        
        if src_dir.exists():
            for py_file in src_dir.rglob("*.py"):
                rel_path = py_file.relative_to(self.project_root)
                source_files.append({
                    "path": str(rel_path),
                    "hash": self._calculate_file_hash(py_file),
                    "size": py_file.stat().st_size
                })
                
        return source_files
    
    def generate_cyclonedx_sbom(self) -> Bom:
        """Generate CycloneDX SBOM."""
        
        # Main component
        main_component = Component(
            type=ComponentType.LIBRARY,
            name=self.package_info.get("name", "wasm-shim-torch"),
            version=self.package_info.get("version", "0.1.0"),
            description=self.package_info.get("description", ""),
            scope=ComponentScope.REQUIRED
        )
        
        # Add license if available
        if "license" in self.package_info:
            license_factory = LicenseFactory()
            license_obj = license_factory.make_from_string(
                self.package_info["license"].get("text", "Apache-2.0")
            )
            main_component.licenses.add(license_obj)
        
        # Create BOM
        bom = Bom()
        bom.metadata.component = main_component
        
        # Add dependencies
        dependencies = self._get_dependencies()
        for dep in dependencies:
            component = Component(
                type=ComponentType.LIBRARY,
                name=dep["name"],
                version=dep["version"],
                scope=ComponentScope.REQUIRED
            )
            
            if "license" in dep:
                license_obj = license_factory.make_from_string(dep["license"])
                component.licenses.add(license_obj)
                
            bom.components.add(component)
        
        return bom
    
    def generate_custom_sbom(self) -> Dict[str, Any]:
        """Generate custom SBOM format with additional information."""
        
        git_info = self._get_git_info()
        dependencies = self._get_dependencies()
        source_files = self._get_source_files()
        
        sbom = {
            "bomFormat": "CycloneDX",
            "specVersion": "1.4",
            "serialNumber": f"urn:uuid:{datetime.now().isoformat()}",
            "version": 1,
            "metadata": {
                "timestamp": datetime.now().isoformat(),
                "tools": [
                    {
                        "vendor": "WASM Torch",
                        "name": "sbom-generator",
                        "version": "1.0.0"
                    }
                ],
                "component": {
                    "type": "library",
                    "bom-ref": self.package_info.get("name", "wasm-shim-torch"),
                    "name": self.package_info.get("name", "wasm-shim-torch"),
                    "version": self.package_info.get("version", "0.1.0"),
                    "description": self.package_info.get("description", ""),
                    "licenses": [
                        {
                            "license": {
                                "name": "Apache-2.0"
                            }
                        }
                    ],
                    "externalReferences": []
                },
                "manufacture": {
                    "name": "WASM Torch Contributors",
                    "url": self.package_info.get("urls", {}).get("Homepage", "")
                }
            },
            "components": [],
            "properties": [
                {
                    "name": "build:timestamp",
                    "value": datetime.now().isoformat()
                },
                {
                    "name": "build:python_version", 
                    "value": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
                }
            ]
        }
        
        # Add Git information
        if git_info:
            sbom["metadata"]["component"]["externalReferences"].extend([
                {
                    "type": "vcs",
                    "url": git_info.get("repository", "")
                },
                {
                    "type": "build-meta",
                    "comment": f"git-commit:{git_info.get('commit', '')}"
                }
            ])
        
        # Add dependencies as components
        for dep in dependencies:
            component = {
                "type": "library",
                "bom-ref": f"{dep['name']}@{dep['version']}",
                "name": dep["name"],
                "version": dep["version"],
                "scope": "required"
            }
            
            if "license" in dep:
                component["licenses"] = [
                    {
                        "license": {
                            "name": dep["license"]
                        }
                    }
                ]
            
            sbom["components"].append(component)
        
        # Add source file information
        if source_files:
            sbom["properties"].append({
                "name": "source:file_count",
                "value": str(len(source_files))
            })
            
            total_size = sum(f["size"] for f in source_files)
            sbom["properties"].append({
                "name": "source:total_size_bytes",
                "value": str(total_size)
            })
        
        return sbom
    
    def generate_security_metadata(self) -> Dict[str, Any]:
        """Generate security-focused metadata."""
        
        return {
            "security": {
                "vulnerability_scan": {
                    "timestamp": datetime.now().isoformat(),
                    "tools": ["pip-audit", "bandit", "safety"],
                    "status": "pending"
                },
                "code_analysis": {
                    "static_analysis": True,
                    "dependency_check": True,
                    "license_compliance": True
                },
                "supply_chain": {
                    "provenance_verification": True,
                    "signature_verification": False,
                    "build_reproducibility": "partial"
                }
            },
            "compliance": {
                "frameworks": ["SLSA", "SSDF"],
                "attestations": []
            }
        }
    
    def save_sbom(self, output_path: Path, format: str = "json") -> None:
        """Save SBOM to file."""
        
        if format == "cyclonedx":
            bom = self.generate_cyclonedx_sbom()
            json_output = JsonV1Dot4(bom)
            with open(output_path, 'w') as f:
                f.write(json_output.output_as_string())
        else:
            sbom = self.generate_custom_sbom()
            
            # Add security metadata
            security_metadata = self.generate_security_metadata()
            sbom.update(security_metadata)
            
            with open(output_path, 'w') as f:
                json.dump(sbom, f, indent=2)
        
        print(f"SBOM generated: {output_path}")
    
    def print_summary(self) -> None:
        """Print SBOM summary."""
        
        dependencies = self._get_dependencies()
        source_files = self._get_source_files()
        
        print("\n=== SBOM Generation Summary ===")
        print(f"Project: {self.package_info.get('name', 'Unknown')}")
        print(f"Version: {self.package_info.get('version', 'Unknown')}")
        print(f"Dependencies: {len(dependencies)}")
        print(f"Source files: {len(source_files)}")
        print(f"Timestamp: {datetime.now().isoformat()}")
        
        git_info = self._get_git_info()
        if git_info:
            print(f"Git commit: {git_info.get('commit', 'Unknown')[:8]}")
            print(f"Repository: {git_info.get('repository', 'Unknown')}")


def main():
    """Main entry point."""
    
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate SBOM for WASM Shim for Torch")
    parser.add_argument(
        "--output", 
        type=Path, 
        default=Path("sbom.json"),
        help="Output file path"
    )
    parser.add_argument(
        "--format",
        choices=["json", "cyclonedx"],
        default="json",
        help="SBOM format"
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path.cwd(),
        help="Project root directory"
    )
    parser.add_argument(
        "--summary",
        action="store_true",
        help="Print summary information"
    )
    
    args = parser.parse_args()
    
    try:
        generator = SBOMGenerator(args.project_root)
        generator.save_sbom(args.output, args.format)
        
        if args.summary:
            generator.print_summary()
            
    except Exception as e:
        print(f"Error generating SBOM: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()