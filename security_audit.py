#!/usr/bin/env python3
"""Advanced security audit script for WASM Torch."""

import os
import sys
import json
import hashlib
import subprocess
import logging
from pathlib import Path
from typing import Dict, List, Any, Set
import re
import ast
import argparse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SecurityAuditor:
    """Comprehensive security auditor for WASM Torch codebase."""
    
    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.findings: List[Dict[str, Any]] = []
        self.severity_levels = ['LOW', 'MEDIUM', 'HIGH', 'CRITICAL']
        
        # Security patterns to check
        self.security_patterns = {
            'hardcoded_secrets': [
                r'password\s*=\s*["\'][^"\']+["\']',
                r'api_key\s*=\s*["\'][^"\']+["\']',
                r'secret\s*=\s*["\'][^"\']+["\']',
                r'token\s*=\s*["\'][^"\']+["\']',
            ],
            'sql_injection': [
                r'execute\s*\(\s*["\'].*%.*["\']',
                r'query\s*\(\s*["\'].*\+.*["\']',
                r'cursor\.execute\s*\(\s*f["\']',
            ],
            'command_injection': [
                r'os\.system\s*\(',
                r'subprocess\.call\s*\([^)]*shell\s*=\s*True',
                r'subprocess\.run\s*\([^)]*shell\s*=\s*True',
                r'eval\s*\(',
                r'exec\s*\(',
            ],
            'path_traversal': [
                r'open\s*\(\s*[^,\)]*\.\.',
                r'Path\s*\([^)]*\.\.',
                r'\.\./',
                r'\.\.\\\\'
            ],
            'unsafe_deserialization': [
                r'pickle\.loads?\s*\(',
                r'marshal\.loads?\s*\(',
                r'yaml\.load\s*\(',
            ],
            'weak_crypto': [
                r'hashlib\.md5\s*\(',
                r'hashlib\.sha1\s*\(',
                r'random\.random\s*\(',
                r'random\.randint\s*\(',
            ]
        }
        
        # Safe patterns that might look suspicious but are OK
        self.safe_patterns = [
            r'# nosec',  # Bandit ignore comment
            r'# pragma: no cover',
            r'test_.*\.py',  # Test files
            r'example.*\.py',  # Example files
        ]
    
    def audit_codebase(self) -> Dict[str, Any]:
        """Perform comprehensive security audit."""
        logger.info("🔒 Starting Security Audit")
        
        # Scan Python files
        self._scan_python_files()
        
        # Check dependencies
        self._check_dependencies()
        
        # Check file permissions
        self._check_file_permissions()
        
        # Check configuration files
        self._check_configuration_security()
        
        # Check Docker/deployment security
        self._check_deployment_security()
        
        # Generate report
        return self._generate_security_report()
    
    def _scan_python_files(self) -> None:
        """Scan Python files for security vulnerabilities."""
        logger.info("Scanning Python files for security issues...")
        
        python_files = list(self.project_root.rglob("*.py"))
        
        for file_path in python_files:
            if self._is_safe_file(file_path):
                continue
            
            try:
                content = file_path.read_text(encoding='utf-8')
                self._analyze_file_content(file_path, content)
                self._analyze_ast(file_path, content)
                
            except Exception as e:
                logger.warning(f"Could not analyze {file_path}: {e}")
    
    def _is_safe_file(self, file_path: Path) -> bool:
        """Check if file should be skipped from security analysis."""
        file_str = str(file_path)
        
        # Skip test files, examples, and other safe files
        safe_indicators = [
            '/test/', '/tests/', '/examples/', '/docs/',
            'test_', '_test.py', 'conftest.py'
        ]
        
        return any(indicator in file_str.lower() for indicator in safe_indicators)
    
    def _analyze_file_content(self, file_path: Path, content: str) -> None:
        """Analyze file content for security patterns."""
        lines = content.split('\n')
        
        for line_num, line in enumerate(lines, 1):
            # Skip commented lines and safe patterns
            if any(re.search(pattern, line) for pattern in self.safe_patterns):
                continue
            
            # Check for security issues
            for category, patterns in self.security_patterns.items():
                for pattern in patterns:
                    if re.search(pattern, line, re.IGNORECASE):
                        severity = self._determine_severity(category, pattern, line)
                        
                        self.findings.append({
                            'type': 'pattern_match',
                            'category': category,
                            'severity': severity,
                            'file': str(file_path.relative_to(self.project_root)),
                            'line': line_num,
                            'pattern': pattern,
                            'code': line.strip(),
                            'description': f"Potential {category.replace('_', ' ')} vulnerability"
                        })
    
    def _analyze_ast(self, file_path: Path, content: str) -> None:
        """Analyze Python AST for deeper security issues."""
        try:
            tree = ast.parse(content)
            visitor = SecurityASTVisitor(file_path, self.project_root)
            visitor.visit(tree)
            self.findings.extend(visitor.findings)
            
        except SyntaxError as e:
            logger.warning(f"Syntax error in {file_path}: {e}")
        except Exception as e:
            logger.warning(f"AST analysis failed for {file_path}: {e}")
    
    def _determine_severity(self, category: str, pattern: str, line: str) -> str:
        """Determine severity of security finding."""
        # Critical patterns
        if category in ['command_injection', 'sql_injection']:
            return 'CRITICAL'
        
        # High severity patterns
        if category in ['hardcoded_secrets', 'unsafe_deserialization']:
            return 'HIGH'
        
        # Medium severity patterns
        if category in ['path_traversal', 'weak_crypto']:
            return 'MEDIUM'
        
        return 'LOW'
    
    def _check_dependencies(self) -> None:
        """Check for vulnerable dependencies."""
        logger.info("Checking dependencies for vulnerabilities...")
        
        # Check requirements files
        req_files = list(self.project_root.rglob("requirements*.txt"))
        req_files.extend(list(self.project_root.rglob("pyproject.toml")))
        
        for req_file in req_files:
            try:
                content = req_file.read_text()
                self._analyze_dependencies(req_file, content)
            except Exception as e:
                logger.warning(f"Could not read {req_file}: {e}")
        
        # Try to run safety check if available
        try:
            result = subprocess.run(
                ['safety', 'check', '--json'],
                cwd=self.project_root,
                capture_output=True,
                text=True,
                timeout=30
            )
            
            if result.returncode == 0:
                safety_data = json.loads(result.stdout)
                for vuln in safety_data:
                    self.findings.append({
                        'type': 'dependency_vulnerability',
                        'category': 'vulnerable_dependency',
                        'severity': 'HIGH',
                        'package': vuln.get('package_name', 'unknown'),
                        'version': vuln.get('installed_version', 'unknown'),
                        'vulnerability': vuln.get('vulnerability_id', 'unknown'),
                        'description': vuln.get('advisory', 'Known vulnerability')
                    })
        except Exception as e:
            logger.info(f"Safety check not available or failed: {e}")
    
    def _analyze_dependencies(self, file_path: Path, content: str) -> None:
        """Analyze dependency versions for known issues."""
        # Check for unpinned dependencies
        lines = content.split('\n')
        for line_num, line in enumerate(lines, 1):
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            
            # Check for unpinned versions
            if '==' not in line and '>=' not in line and line.count('=') == 0:
                self.findings.append({
                    'type': 'dependency_issue',
                    'category': 'unpinned_dependency',
                    'severity': 'MEDIUM',
                    'file': str(file_path.relative_to(self.project_root)),
                    'line': line_num,
                    'dependency': line,
                    'description': 'Unpinned dependency version could lead to supply chain attacks'
                })
    
    def _check_file_permissions(self) -> None:
        """Check for insecure file permissions."""
        logger.info("Checking file permissions...")
        
        for file_path in self.project_root.rglob("*"):
            if file_path.is_file():
                try:
                    stat = file_path.stat()
                    mode = oct(stat.st_mode)[-3:]
                    
                    # Check for world-writable files
                    if mode.endswith('6') or mode.endswith('7'):
                        self.findings.append({
                            'type': 'permission_issue',
                            'category': 'insecure_permissions',
                            'severity': 'MEDIUM',
                            'file': str(file_path.relative_to(self.project_root)),
                            'permissions': mode,
                            'description': 'File is world-writable'
                        })
                    
                    # Check for executable files that shouldn't be
                    if (file_path.suffix in ['.py', '.txt', '.md', '.json', '.yaml', '.yml'] and
                        mode.startswith('7')):
                        if not str(file_path).endswith('.py') or 'script' not in str(file_path).lower():
                            self.findings.append({
                                'type': 'permission_issue',
                                'category': 'unnecessary_executable',
                                'severity': 'LOW',
                                'file': str(file_path.relative_to(self.project_root)),
                                'permissions': mode,
                                'description': 'File has unnecessary execute permissions'
                            })
                            
                except Exception as e:
                    logger.warning(f"Could not check permissions for {file_path}: {e}")
    
    def _check_configuration_security(self) -> None:
        """Check configuration files for security issues."""
        logger.info("Checking configuration security...")
        
        config_files = []
        config_files.extend(list(self.project_root.rglob("*.yaml")))
        config_files.extend(list(self.project_root.rglob("*.yml")))
        config_files.extend(list(self.project_root.rglob("*.json")))
        config_files.extend(list(self.project_root.rglob("*.conf")))
        config_files.extend(list(self.project_root.rglob("*.ini")))
        
        for config_file in config_files:
            try:
                content = config_file.read_text()
                self._analyze_config_content(config_file, content)
            except Exception as e:
                logger.warning(f"Could not read config file {config_file}: {e}")
    
    def _analyze_config_content(self, file_path: Path, content: str) -> None:
        """Analyze configuration file content."""
        lines = content.split('\n')
        
        for line_num, line in enumerate(lines, 1):
            line_lower = line.lower()
            
            # Check for hardcoded secrets in configs
            secret_indicators = ['password', 'secret', 'key', 'token', 'credential']
            for indicator in secret_indicators:
                if indicator in line_lower and ('=' in line or ':' in line):
                    if not any(placeholder in line_lower for placeholder in 
                             ['placeholder', 'example', 'your_', 'xxx', '***']):
                        self.findings.append({
                            'type': 'config_issue',
                            'category': 'hardcoded_secret_config',
                            'severity': 'HIGH',
                            'file': str(file_path.relative_to(self.project_root)),
                            'line': line_num,
                            'description': f'Potential hardcoded secret in configuration: {indicator}'
                        })
            
            # Check for insecure defaults
            insecure_patterns = [
                ('debug.*true', 'Debug mode enabled'),
                ('ssl.*false', 'SSL disabled'),
                ('verify.*false', 'Certificate verification disabled'),
                ('0.0.0.0', 'Binding to all interfaces')
            ]
            
            for pattern, desc in insecure_patterns:
                if re.search(pattern, line_lower):
                    self.findings.append({
                        'type': 'config_issue',
                        'category': 'insecure_config',
                        'severity': 'MEDIUM',
                        'file': str(file_path.relative_to(self.project_root)),
                        'line': line_num,
                        'description': desc
                    })
    
    def _check_deployment_security(self) -> None:
        """Check deployment configuration security."""
        logger.info("Checking deployment security...")
        
        # Check Dockerfile
        dockerfile_paths = list(self.project_root.rglob("Dockerfile*"))
        for dockerfile in dockerfile_paths:
            try:
                content = dockerfile.read_text()
                self._analyze_dockerfile(dockerfile, content)
            except Exception as e:
                logger.warning(f"Could not read {dockerfile}: {e}")
        
        # Check docker-compose files
        compose_files = list(self.project_root.rglob("docker-compose*.yml"))
        compose_files.extend(list(self.project_root.rglob("docker-compose*.yaml")))
        
        for compose_file in compose_files:
            try:
                content = compose_file.read_text()
                self._analyze_docker_compose(compose_file, content)
            except Exception as e:
                logger.warning(f"Could not read {compose_file}: {e}")
    
    def _analyze_dockerfile(self, file_path: Path, content: str) -> None:
        """Analyze Dockerfile for security issues."""
        lines = content.split('\n')
        
        for line_num, line in enumerate(lines, 1):
            line_upper = line.upper().strip()
            
            # Check for running as root
            if line_upper.startswith('USER') and 'root' in line.lower():
                self.findings.append({
                    'type': 'dockerfile_issue',
                    'category': 'container_security',
                    'severity': 'HIGH',
                    'file': str(file_path.relative_to(self.project_root)),
                    'line': line_num,
                    'description': 'Container running as root user'
                })
            
            # Check for latest tag usage
            if 'FROM' in line_upper and ':latest' in line:
                self.findings.append({
                    'type': 'dockerfile_issue',
                    'category': 'container_security',
                    'severity': 'MEDIUM',
                    'file': str(file_path.relative_to(self.project_root)),
                    'line': line_num,
                    'description': 'Using :latest tag is not recommended for production'
                })
    
    def _analyze_docker_compose(self, file_path: Path, content: str) -> None:
        """Analyze docker-compose file for security issues."""
        # Check for privileged containers
        if 'privileged: true' in content:
            self.findings.append({
                'type': 'compose_issue',
                'category': 'container_security',
                'severity': 'CRITICAL',
                'file': str(file_path.relative_to(self.project_root)),
                'description': 'Privileged container mode detected'
            })
        
        # Check for host network mode
        if 'network_mode: host' in content:
            self.findings.append({
                'type': 'compose_issue',
                'category': 'container_security',
                'severity': 'HIGH',
                'file': str(file_path.relative_to(self.project_root)),
                'description': 'Host network mode bypasses container isolation'
            })
    
    def _generate_security_report(self) -> Dict[str, Any]:
        """Generate comprehensive security report."""
        # Count findings by severity
        severity_counts = {severity: 0 for severity in self.severity_levels}
        for finding in self.findings:
            severity_counts[finding.get('severity', 'LOW')] += 1
        
        # Calculate risk score
        risk_score = (
            severity_counts['CRITICAL'] * 10 +
            severity_counts['HIGH'] * 5 +
            severity_counts['MEDIUM'] * 2 +
            severity_counts['LOW'] * 1
        )
        
        # Determine overall security level
        if severity_counts['CRITICAL'] > 0:
            security_level = 'CRITICAL'
        elif severity_counts['HIGH'] > 0:
            security_level = 'HIGH'
        elif severity_counts['MEDIUM'] > 0:
            security_level = 'MEDIUM'
        else:
            security_level = 'LOW'
        
        report = {
            'summary': {
                'total_findings': len(self.findings),
                'severity_counts': severity_counts,
                'risk_score': risk_score,
                'security_level': security_level,
                'scan_timestamp': __import__('datetime').datetime.now().isoformat()
            },
            'findings': self.findings,
            'recommendations': self._generate_recommendations()
        }
        
        return report
    
    def _generate_recommendations(self) -> List[str]:
        """Generate security recommendations based on findings."""
        recommendations = []
        
        # Count findings by category
        category_counts = {}
        for finding in self.findings:
            category = finding.get('category', 'unknown')
            category_counts[category] = category_counts.get(category, 0) + 1
        
        # Generate category-specific recommendations
        if category_counts.get('hardcoded_secrets', 0) > 0:
            recommendations.append(
                "Use environment variables or secure secret management for sensitive data"
            )
        
        if category_counts.get('command_injection', 0) > 0:
            recommendations.append(
                "Avoid shell=True in subprocess calls and validate all user inputs"
            )
        
        if category_counts.get('path_traversal', 0) > 0:
            recommendations.append(
                "Implement proper path validation and sanitization"
            )
        
        if category_counts.get('vulnerable_dependency', 0) > 0:
            recommendations.append(
                "Update vulnerable dependencies to secure versions"
            )
        
        if category_counts.get('container_security', 0) > 0:
            recommendations.append(
                "Follow container security best practices (non-root user, specific tags)"
            )
        
        # General recommendations
        if len(self.findings) > 10:
            recommendations.append(
                "Consider implementing automated security scanning in CI/CD pipeline"
            )
        
        return recommendations


class SecurityASTVisitor(ast.NodeVisitor):
    """AST visitor for detecting security issues."""
    
    def __init__(self, file_path: Path, project_root: Path):
        self.file_path = file_path
        self.project_root = project_root
        self.findings: List[Dict[str, Any]] = []
    
    def visit_Call(self, node: ast.Call) -> None:
        """Check function calls for security issues."""
        # Get function name
        func_name = ''
        if isinstance(node.func, ast.Name):
            func_name = node.func.id
        elif isinstance(node.func, ast.Attribute):
            func_name = node.func.attr
        
        # Check for dangerous function calls
        dangerous_functions = {
            'eval': 'CRITICAL',
            'exec': 'CRITICAL',
            'compile': 'HIGH',
            '__import__': 'HIGH'
        }
        
        if func_name in dangerous_functions:
            self.findings.append({
                'type': 'ast_analysis',
                'category': 'dangerous_function',
                'severity': dangerous_functions[func_name],
                'file': str(self.file_path.relative_to(self.project_root)),
                'line': node.lineno,
                'function': func_name,
                'description': f'Dangerous function call: {func_name}'
            })
        
        self.generic_visit(node)
    
    def visit_Import(self, node: ast.Import) -> None:
        """Check imports for suspicious modules."""
        for alias in node.names:
            self._check_suspicious_import(alias.name, node.lineno)
        self.generic_visit(node)
    
    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        """Check from imports for suspicious modules."""
        if node.module:
            self._check_suspicious_import(node.module, node.lineno)
        self.generic_visit(node)
    
    def _check_suspicious_import(self, module_name: str, line_num: int) -> None:
        """Check if imported module is suspicious."""
        suspicious_modules = {
            'os': 'LOW',
            'subprocess': 'MEDIUM',
            'pickle': 'MEDIUM',
            'marshal': 'HIGH'
        }
        
        if module_name in suspicious_modules:
            self.findings.append({
                'type': 'ast_analysis',
                'category': 'suspicious_import',
                'severity': suspicious_modules[module_name],
                'file': str(self.file_path.relative_to(self.project_root)),
                'line': line_num,
                'module': module_name,
                'description': f'Import of potentially dangerous module: {module_name}'
            })


def main():
    """Main function for security audit."""
    parser = argparse.ArgumentParser(description='WASM Torch Security Auditor')
    parser.add_argument('--project-root', type=Path, default=Path('.'),
                       help='Project root directory to audit')
    parser.add_argument('--output', type=str, default='security_audit_report.json',
                       help='Output file for security report')
    parser.add_argument('--fail-on-high', action='store_true',
                       help='Exit with error code if HIGH or CRITICAL issues found')
    
    args = parser.parse_args()
    
    if not args.project_root.exists():
        logger.error(f"Project root does not exist: {args.project_root}")
        sys.exit(1)
    
    auditor = SecurityAuditor(args.project_root)
    report = auditor.audit_codebase()
    
    # Save report
    with open(args.output, 'w') as f:
        json.dump(report, f, indent=2)
    
    # Print summary
    summary = report['summary']
    logger.info("🔒 Security Audit Summary:")
    logger.info(f"   Total Findings: {summary['total_findings']}")
    logger.info(f"   Risk Score: {summary['risk_score']}")
    logger.info(f"   Security Level: {summary['security_level']}")
    
    for severity in ['CRITICAL', 'HIGH', 'MEDIUM', 'LOW']:
        count = summary['severity_counts'][severity]
        if count > 0:
            logger.info(f"   {severity}: {count}")
    
    if report['recommendations']:
        logger.info("🔍 Security Recommendations:")
        for rec in report['recommendations']:
            logger.info(f"   • {rec}")
    
    # Exit with error if requested and high-severity issues found
    if args.fail_on_high and (summary['severity_counts']['HIGH'] > 0 or 
                              summary['severity_counts']['CRITICAL'] > 0):
        logger.error("High or critical security issues found!")
        sys.exit(1)
    
    logger.info(f"Security audit complete. Report saved to: {args.output}")


if __name__ == "__main__":
    main()