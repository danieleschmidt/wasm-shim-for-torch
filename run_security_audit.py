#!/usr/bin/env python3
"""
Security Audit and Vulnerability Assessment for WASM-Torch
Comprehensive security scanning and hardening validation
"""

import os
import sys
import json
import time
import hashlib
import subprocess
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class SecurityAuditor:
    """Comprehensive security auditor for WASM-Torch."""
    
    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.security_report = {
            "scan_timestamp": time.time(),
            "total_files_scanned": 0,
            "vulnerabilities": [],
            "security_score": 0.0,
            "compliance_status": {},
            "recommendations": []
        }
        
        # Security patterns to detect
        self.vulnerability_patterns = {
            "sql_injection": [
                r"SELECT.*\+.*",
                r"INSERT.*\+.*",
                r"UPDATE.*\+.*",
                r"DELETE.*\+.*",
                r"\.execute\([^)]*\+",
            ],
            "code_injection": [
                r"eval\s*\(",
                r"exec\s*\(",
                r"compile\s*\(",
                r"__import__\s*\(",
                r"subprocess\.call.*shell=True",
                r"os\.system\s*\(",
            ],
            "path_traversal": [
                r"\.\./",
                r"\.\.\\",
                r"\.\.%2f",
                r"\.\.%5c",
                r"%2e%2e%2f",
                r"%2e%2e%5c",
            ],
            "hardcoded_secrets": [
                r"password\s*=\s*['\"][^'\"]{8,}['\"]",
                r"api_key\s*=\s*['\"][^'\"]{20,}['\"]",
                r"secret\s*=\s*['\"][^'\"]{16,}['\"]",
                r"token\s*=\s*['\"][^'\"]{20,}['\"]",
                r"private_key\s*=\s*['\"]",
            ],
            "insecure_random": [
                r"random\.random\(\)",
                r"random\.randint\(",
                r"random\.choice\(",
                r"time\.time\(\).*random",
            ],
            "dangerous_deserialization": [
                r"pickle\.loads\(",
                r"yaml\.load\(",
                r"eval\s*\(\s*input\s*\(",
                r"marshal\.loads\(",
            ]
        }
        
        # Security best practices checks
        self.security_checks = [
            "check_file_permissions",
            "check_dependency_vulnerabilities", 
            "check_configuration_security",
            "check_input_validation",
            "check_error_handling",
            "check_logging_security",
            "check_authentication_mechanisms",
            "check_encryption_usage"
        ]
    
    def run_comprehensive_audit(self) -> Dict[str, Any]:
        """Run comprehensive security audit."""
        
        logger.info("🔒 Starting comprehensive security audit")
        
        # Scan source code for vulnerabilities
        self._scan_source_code()
        
        # Check file permissions and access controls
        self._check_file_security()
        
        # Validate configuration security
        self._check_configuration_security()
        
        # Check for dependency vulnerabilities (simulated)
        self._check_dependency_security()
        
        # Validate input handling
        self._check_input_validation()
        
        # Check error handling and information disclosure
        self._check_error_handling()
        
        # Validate logging and monitoring
        self._check_logging_security()
        
        # Check encryption and cryptographic practices
        self._check_cryptographic_security()
        
        # Calculate security score
        self._calculate_security_score()
        
        # Generate recommendations
        self._generate_security_recommendations()
        
        logger.info(f"🔍 Security audit complete. Score: {self.security_report['security_score']:.1f}/100")
        
        return self.security_report
    
    def _scan_source_code(self):
        """Scan source code for vulnerability patterns."""
        
        logger.info("📁 Scanning source code for vulnerabilities")
        
        python_files = list(self.project_root.rglob("*.py"))
        self.security_report["total_files_scanned"] = len(python_files)
        
        for file_path in python_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                self._check_file_for_vulnerabilities(file_path, content)
                
            except Exception as e:
                logger.warning(f"Could not scan {file_path}: {e}")
    
    def _check_file_for_vulnerabilities(self, file_path: Path, content: str):
        """Check a single file for vulnerability patterns."""
        
        import re
        
        lines = content.split('\n')
        
        for vuln_type, patterns in self.vulnerability_patterns.items():
            for pattern in patterns:
                try:
                    matches = re.finditer(pattern, content, re.IGNORECASE | re.MULTILINE)
                    
                    for match in matches:
                        # Find line number
                        line_num = content[:match.start()].count('\n') + 1
                        
                        vulnerability = {
                            "type": vuln_type,
                            "file": str(file_path.relative_to(self.project_root)),
                            "line": line_num,
                            "pattern": pattern,
                            "matched_text": match.group(),
                            "severity": self._get_vulnerability_severity(vuln_type),
                            "description": self._get_vulnerability_description(vuln_type)
                        }
                        
                        self.security_report["vulnerabilities"].append(vulnerability)
                        
                except re.error as e:
                    logger.warning(f"Regex error for pattern {pattern}: {e}")
    
    def _get_vulnerability_severity(self, vuln_type: str) -> str:
        """Get severity level for vulnerability type."""
        
        severity_map = {
            "sql_injection": "high",
            "code_injection": "critical",
            "path_traversal": "high", 
            "hardcoded_secrets": "high",
            "insecure_random": "medium",
            "dangerous_deserialization": "high"
        }
        
        return severity_map.get(vuln_type, "medium")
    
    def _get_vulnerability_description(self, vuln_type: str) -> str:
        """Get description for vulnerability type."""
        
        descriptions = {
            "sql_injection": "Potential SQL injection vulnerability through string concatenation",
            "code_injection": "Code injection vulnerability through dynamic execution",
            "path_traversal": "Path traversal vulnerability allowing directory navigation",
            "hardcoded_secrets": "Hardcoded credentials or secrets in source code",
            "insecure_random": "Use of insecure random number generation",
            "dangerous_deserialization": "Dangerous deserialization that could lead to code execution"
        }
        
        return descriptions.get(vuln_type, "Security vulnerability detected")
    
    def _check_file_security(self):
        """Check file permissions and access controls."""
        
        logger.info("🔐 Checking file permissions and access controls")
        
        security_issues = []
        
        # Check for world-writable files
        for file_path in self.project_root.rglob("*"):
            if file_path.is_file():
                try:
                    stat_info = file_path.stat()
                    mode = oct(stat_info.st_mode)[-3:]
                    
                    # Check for world-writable (mode ending in 2, 3, 6, 7)
                    if mode[-1] in ['2', '3', '6', '7']:
                        security_issues.append({
                            "type": "file_permissions",
                            "file": str(file_path.relative_to(self.project_root)),
                            "issue": "World-writable file",
                            "mode": mode,
                            "severity": "medium"
                        })
                    
                    # Check for executable Python files
                    if file_path.suffix == '.py' and mode[0] in ['7', '5', '1']:
                        if not file_path.name.startswith('run_') and file_path.name != '__main__.py':
                            security_issues.append({
                                "type": "file_permissions", 
                                "file": str(file_path.relative_to(self.project_root)),
                                "issue": "Executable Python file (potential security risk)",
                                "mode": mode,
                                "severity": "low"
                            })
                            
                except OSError:
                    pass  # Skip files we can't stat
        
        # Check for sensitive files in version control
        sensitive_patterns = [
            "*.key", "*.pem", "*.p12", "*.pfx",
            "*.env", ".env.*", 
            "*password*", "*secret*", "*token*"
        ]
        
        for pattern in sensitive_patterns:
            for file_path in self.project_root.rglob(pattern):
                if file_path.is_file():
                    security_issues.append({
                        "type": "sensitive_files",
                        "file": str(file_path.relative_to(self.project_root)),
                        "issue": "Potentially sensitive file in repository",
                        "pattern": pattern,
                        "severity": "high"
                    })
        
        self.security_report["file_security_issues"] = security_issues
    
    def _check_configuration_security(self):
        """Check configuration files for security issues."""
        
        logger.info("⚙️ Checking configuration security")
        
        config_issues = []
        
        # Check common configuration files
        config_files = [
            "*.yml", "*.yaml", "*.json", "*.toml", "*.ini", "*.cfg"
        ]
        
        for pattern in config_files:
            for config_file in self.project_root.rglob(pattern):
                if config_file.is_file():
                    try:
                        with open(config_file, 'r') as f:
                            content = f.read().lower()
                        
                        # Check for sensitive data in config
                        sensitive_keywords = [
                            "password", "secret", "key", "token", "credential",
                            "auth", "api_key", "private", "cert"
                        ]
                        
                        for keyword in sensitive_keywords:
                            if keyword in content:
                                config_issues.append({
                                    "type": "configuration_security",
                                    "file": str(config_file.relative_to(self.project_root)),
                                    "issue": f"Configuration contains sensitive keyword: {keyword}",
                                    "severity": "medium"
                                })
                        
                        # Check for default credentials
                        default_creds = [
                            "admin:admin", "root:root", "admin:password",
                            "test:test", "guest:guest"
                        ]
                        
                        for cred in default_creds:
                            if cred in content:
                                config_issues.append({
                                    "type": "configuration_security",
                                    "file": str(config_file.relative_to(self.project_root)),
                                    "issue": f"Default credentials found: {cred}",
                                    "severity": "high"
                                })
                                
                    except Exception:
                        pass  # Skip files we can't read
        
        self.security_report["configuration_issues"] = config_issues
    
    def _check_dependency_security(self):
        """Check for known vulnerable dependencies."""
        
        logger.info("📦 Checking dependency security")
        
        dependency_issues = []
        
        # Check requirements.txt for known vulnerable packages
        requirements_file = self.project_root / "requirements.txt"
        if requirements_file.exists():
            try:
                with open(requirements_file, 'r') as f:
                    requirements = f.read()
                
                # Simulate vulnerability database check
                known_vulnerable = [
                    ("pillow", "8.0.0", "CVE-2021-25287", "high"),
                    ("pyyaml", "5.1", "CVE-2020-1747", "medium"),
                    ("jinja2", "2.10", "CVE-2020-28493", "medium"),
                    ("urllib3", "1.25.8", "CVE-2021-33503", "medium")
                ]
                
                for package, version, cve, severity in known_vulnerable:
                    if package in requirements.lower():
                        dependency_issues.append({
                            "type": "vulnerable_dependency",
                            "package": package,
                            "version": version,
                            "cve": cve,
                            "severity": severity,
                            "description": f"Package {package} has known vulnerability {cve}"
                        })
                        
            except Exception as e:
                logger.warning(f"Could not check requirements.txt: {e}")
        
        # Check pyproject.toml for similar issues
        pyproject_file = self.project_root / "pyproject.toml"
        if pyproject_file.exists():
            try:
                with open(pyproject_file, 'r') as f:
                    content = f.read()
                
                # Check for development dependencies in production
                if 'pytest' in content and 'tool.pytest' not in content:
                    dependency_issues.append({
                        "type": "dependency_security",
                        "issue": "Development dependencies may be included in production",
                        "severity": "low",
                        "description": "Development tools should not be included in production builds"
                    })
                    
            except Exception as e:
                logger.warning(f"Could not check pyproject.toml: {e}")
        
        self.security_report["dependency_issues"] = dependency_issues
    
    def _check_input_validation(self):
        """Check input validation mechanisms."""
        
        logger.info("🔍 Checking input validation")
        
        validation_issues = []
        
        # Look for input handling patterns
        python_files = list(self.project_root.rglob("*.py"))
        
        for file_path in python_files:
            try:
                with open(file_path, 'r') as f:
                    content = f.read()
                
                lines = content.split('\n')
                
                # Check for direct user input usage without validation
                import re
                
                risky_patterns = [
                    (r"input\s*\([^)]*\)", "Direct input() usage without validation"),
                    (r"sys\.argv\[[^]]+\]", "Command line argument usage without validation"),
                    (r"request\.(GET|POST|args|form)\[", "Web request parameter usage"),
                    (r"json\.loads\([^)]*\)", "JSON parsing without exception handling"),
                    (r"int\([^)]*\)", "Integer conversion without validation"),
                    (r"float\([^)]*\)", "Float conversion without validation")
                ]
                
                for pattern, description in risky_patterns:
                    matches = re.finditer(pattern, content, re.IGNORECASE)
                    
                    for match in matches:
                        line_num = content[:match.start()].count('\n') + 1
                        
                        # Check if there's nearby validation (simple heuristic)
                        line_content = lines[line_num - 1] if line_num <= len(lines) else ""
                        nearby_lines = lines[max(0, line_num-3):min(len(lines), line_num+3)]
                        nearby_content = '\n'.join(nearby_lines)
                        
                        # Look for validation keywords
                        validation_keywords = [
                            "validate", "check", "verify", "isinstance", "try:",
                            "except", "if", "assert", "raise"
                        ]
                        
                        has_validation = any(keyword in nearby_content.lower() 
                                           for keyword in validation_keywords)
                        
                        if not has_validation:
                            validation_issues.append({
                                "type": "input_validation",
                                "file": str(file_path.relative_to(self.project_root)),
                                "line": line_num,
                                "issue": description,
                                "matched_text": match.group(),
                                "severity": "medium"
                            })
                            
            except Exception:
                pass  # Skip files we can't read
        
        self.security_report["input_validation_issues"] = validation_issues
    
    def _check_error_handling(self):
        """Check error handling and information disclosure."""
        
        logger.info("⚠️ Checking error handling")
        
        error_issues = []
        
        python_files = list(self.project_root.rglob("*.py"))
        
        for file_path in python_files:
            try:
                with open(file_path, 'r') as f:
                    content = f.read()
                
                import re
                
                # Check for information disclosure in error messages
                disclosure_patterns = [
                    (r"print\s*\(\s*[^)]*exception[^)]*\)", "Exception details printed to output"),
                    (r"print\s*\(\s*[^)]*error[^)]*\)", "Error details printed to output"),
                    (r"print\s*\(\s*[^)]*traceback[^)]*\)", "Traceback printed to output"),
                    (r"traceback\.print_exc\s*\(\s*\)", "Full traceback disclosure"),
                    (r"str\s*\(\s*e\s*\)", "Exception converted to string (potential disclosure)")
                ]
                
                for pattern, description in disclosure_patterns:
                    matches = re.finditer(pattern, content, re.IGNORECASE)
                    
                    for match in matches:
                        line_num = content[:match.start()].count('\n') + 1
                        
                        error_issues.append({
                            "type": "error_handling",
                            "file": str(file_path.relative_to(self.project_root)),
                            "line": line_num,
                            "issue": description,
                            "matched_text": match.group(),
                            "severity": "low"
                        })
                
                # Check for bare except clauses
                bare_except_matches = re.finditer(r"except\s*:", content)
                for match in bare_except_matches:
                    line_num = content[:match.start()].count('\n') + 1
                    error_issues.append({
                        "type": "error_handling",
                        "file": str(file_path.relative_to(self.project_root)),
                        "line": line_num,
                        "issue": "Bare except clause catches all exceptions",
                        "severity": "medium"
                    })
                    
            except Exception:
                pass
        
        self.security_report["error_handling_issues"] = error_issues
    
    def _check_logging_security(self):
        """Check logging configuration and security."""
        
        logger.info("📝 Checking logging security")
        
        logging_issues = []
        
        python_files = list(self.project_root.rglob("*.py"))
        
        for file_path in python_files:
            try:
                with open(file_path, 'r') as f:
                    content = f.read()
                
                import re
                
                # Check for logging of sensitive data
                sensitive_logging = [
                    (r"log.*password", "Potential password logging"),
                    (r"log.*secret", "Potential secret logging"),
                    (r"log.*token", "Potential token logging"),
                    (r"log.*key", "Potential key logging"),
                    (r"print.*password", "Password printed to output"),
                    (r"print.*secret", "Secret printed to output")
                ]
                
                for pattern, description in sensitive_logging:
                    matches = re.finditer(pattern, content, re.IGNORECASE)
                    
                    for match in matches:
                        line_num = content[:match.start()].count('\n') + 1
                        
                        logging_issues.append({
                            "type": "logging_security",
                            "file": str(file_path.relative_to(self.project_root)),
                            "line": line_num,
                            "issue": description,
                            "matched_text": match.group(),
                            "severity": "high"
                        })
                        
            except Exception:
                pass
        
        self.security_report["logging_issues"] = logging_issues
    
    def _check_cryptographic_security(self):
        """Check cryptographic implementations and practices."""
        
        logger.info("🔐 Checking cryptographic security")
        
        crypto_issues = []
        
        python_files = list(self.project_root.rglob("*.py"))
        
        for file_path in python_files:
            try:
                with open(file_path, 'r') as f:
                    content = f.read()
                
                import re
                
                # Check for weak cryptographic practices
                weak_crypto = [
                    (r"md5\s*\(", "MD5 is cryptographically broken"),
                    (r"sha1\s*\(", "SHA1 is cryptographically weak"),
                    (r"DES\s*\(", "DES encryption is weak"),
                    (r"RC4\s*\(", "RC4 cipher is broken"),
                    (r"random\.random\(\).*crypt", "Insecure random for cryptography"),
                    (r"time\.time\(\).*salt", "Predictable salt generation")
                ]
                
                for pattern, description in weak_crypto:
                    matches = re.finditer(pattern, content, re.IGNORECASE)
                    
                    for match in matches:
                        line_num = content[:match.start()].count('\n') + 1
                        
                        crypto_issues.append({
                            "type": "cryptographic_security",
                            "file": str(file_path.relative_to(self.project_root)),
                            "line": line_num,
                            "issue": description,
                            "matched_text": match.group(),
                            "severity": "high"
                        })
                
                # Check for hardcoded cryptographic keys
                key_patterns = [
                    (r"['\"][A-Za-z0-9+/]{32,}={0,2}['\"]", "Potential hardcoded base64 key"),
                    (r"['\"][0-9a-fA-F]{32,}['\"]", "Potential hardcoded hex key")
                ]
                
                for pattern, description in key_patterns:
                    matches = re.finditer(pattern, content)
                    
                    for match in matches:
                        # Skip if it looks like a test or example
                        if any(word in match.group().lower() for word in ['test', 'example', 'demo', 'sample']):
                            continue
                            
                        line_num = content[:match.start()].count('\n') + 1
                        
                        crypto_issues.append({
                            "type": "cryptographic_security",
                            "file": str(file_path.relative_to(self.project_root)),
                            "line": line_num,
                            "issue": description,
                            "matched_text": match.group()[:20] + "...",  # Truncate for security
                            "severity": "critical"
                        })
                        
            except Exception:
                pass
        
        self.security_report["cryptographic_issues"] = crypto_issues
    
    def _calculate_security_score(self):
        """Calculate overall security score."""
        
        # Base score
        score = 100.0
        
        # Deduct points for vulnerabilities
        severity_penalties = {
            "critical": 20,
            "high": 10,
            "medium": 5,
            "low": 2
        }
        
        all_issues = []
        all_issues.extend(self.security_report.get("vulnerabilities", []))
        all_issues.extend(self.security_report.get("file_security_issues", []))
        all_issues.extend(self.security_report.get("configuration_issues", []))
        all_issues.extend(self.security_report.get("dependency_issues", []))
        all_issues.extend(self.security_report.get("input_validation_issues", []))
        all_issues.extend(self.security_report.get("error_handling_issues", []))
        all_issues.extend(self.security_report.get("logging_issues", []))
        all_issues.extend(self.security_report.get("cryptographic_issues", []))
        
        for issue in all_issues:
            severity = issue.get("severity", "medium")
            penalty = severity_penalties.get(severity, 5)
            score -= penalty
        
        # Ensure score doesn't go below 0
        score = max(0.0, score)
        
        self.security_report["security_score"] = score
        self.security_report["total_issues"] = len(all_issues)
        
        # Security grade
        if score >= 90:
            grade = "A"
        elif score >= 80:
            grade = "B" 
        elif score >= 70:
            grade = "C"
        elif score >= 60:
            grade = "D"
        else:
            grade = "F"
        
        self.security_report["security_grade"] = grade
    
    def _generate_security_recommendations(self):
        """Generate security recommendations based on findings."""
        
        recommendations = []
        
        # General recommendations
        recommendations.extend([
            "Implement comprehensive input validation for all user inputs",
            "Use parameterized queries to prevent SQL injection",
            "Implement proper error handling without information disclosure",
            "Use strong cryptographic algorithms (AES-256, SHA-256 or better)",
            "Implement proper authentication and authorization mechanisms",
            "Use secure random number generation for cryptographic purposes",
            "Regularly update dependencies to patch known vulnerabilities",
            "Implement comprehensive logging without sensitive data exposure",
            "Use environment variables for configuration secrets",
            "Implement rate limiting and DoS protection"
        ])
        
        # Specific recommendations based on findings
        issue_types = set()
        all_issues = []
        all_issues.extend(self.security_report.get("vulnerabilities", []))
        all_issues.extend(self.security_report.get("file_security_issues", []))
        all_issues.extend(self.security_report.get("configuration_issues", []))
        all_issues.extend(self.security_report.get("dependency_issues", []))
        all_issues.extend(self.security_report.get("input_validation_issues", []))
        all_issues.extend(self.security_report.get("error_handling_issues", []))
        all_issues.extend(self.security_report.get("logging_issues", []))
        all_issues.extend(self.security_report.get("cryptographic_issues", []))
        
        for issue in all_issues:
            issue_types.add(issue.get("type", "unknown"))
        
        if "code_injection" in issue_types:
            recommendations.append("CRITICAL: Remove all dynamic code execution (eval, exec)")
        
        if "hardcoded_secrets" in issue_types:
            recommendations.append("HIGH: Move all secrets to environment variables or secure vaults")
        
        if "vulnerable_dependency" in issue_types:
            recommendations.append("HIGH: Update vulnerable dependencies immediately")
        
        if "cryptographic_security" in issue_types:
            recommendations.append("HIGH: Review and strengthen cryptographic implementations")
        
        if "input_validation" in issue_types:
            recommendations.append("MEDIUM: Implement comprehensive input validation")
        
        self.security_report["recommendations"] = recommendations


def main():
    """Main security audit execution."""
    
    print("🔒 WASM-Torch Security Audit & Vulnerability Assessment")
    print("=" * 70)
    
    project_root = Path(__file__).parent
    auditor = SecurityAuditor(project_root)
    
    try:
        # Run comprehensive security audit
        report = auditor.run_comprehensive_audit()
        
        # Print summary
        print("\n" + "=" * 70)
        print("🛡️ SECURITY AUDIT SUMMARY")
        print("=" * 70)
        print(f"Security Score: {report['security_score']:.1f}/100 (Grade: {report['security_grade']})")
        print(f"Total Files Scanned: {report['total_files_scanned']}")
        print(f"Total Issues Found: {report['total_issues']}")
        
        # Print issue breakdown
        issue_categories = [
            ("Code Vulnerabilities", "vulnerabilities"),
            ("File Security", "file_security_issues"),
            ("Configuration", "configuration_issues"),
            ("Dependencies", "dependency_issues"),
            ("Input Validation", "input_validation_issues"),
            ("Error Handling", "error_handling_issues"),
            ("Logging Security", "logging_issues"),
            ("Cryptographic", "cryptographic_issues")
        ]
        
        print("\n📊 ISSUE BREAKDOWN:")
        for category_name, category_key in issue_categories:
            issues = report.get(category_key, [])
            if issues:
                print(f"  {category_name}: {len(issues)} issues")
                
                # Show severity breakdown
                severity_counts = {}
                for issue in issues:
                    severity = issue.get("severity", "unknown")
                    severity_counts[severity] = severity_counts.get(severity, 0) + 1
                
                severity_str = ", ".join([f"{count} {sev}" for sev, count in severity_counts.items()])
                print(f"    ({severity_str})")
        
        # Print top recommendations
        print("\n🎯 TOP SECURITY RECOMMENDATIONS:")
        for i, rec in enumerate(report["recommendations"][:5], 1):
            print(f"  {i}. {rec}")
        
        if len(report["recommendations"]) > 5:
            print(f"  ... and {len(report['recommendations']) - 5} more recommendations")
        
        # Print critical issues
        critical_issues = []
        for category_key in [key for _, key in issue_categories]:
            for issue in report.get(category_key, []):
                if issue.get("severity") == "critical":
                    critical_issues.append(issue)
        
        if critical_issues:
            print("\n🚨 CRITICAL ISSUES REQUIRING IMMEDIATE ATTENTION:")
            for issue in critical_issues[:3]:  # Show top 3
                print(f"  ❌ {issue.get('issue', issue.get('description', 'Unknown issue'))}")
                if 'file' in issue:
                    print(f"     File: {issue['file']} (Line {issue.get('line', 'N/A')})")
        
        # Save detailed report
        report_file = project_root / "security_audit_report.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        print(f"\n📄 Detailed security report saved to {report_file}")
        
        # Return exit code based on security score
        if report["security_score"] >= 80:
            print("\n✅ SECURITY AUDIT PASSED - Good security posture")
            exit_code = 0
        elif report["security_score"] >= 60:
            print("\n⚠️ SECURITY AUDIT WARNING - Moderate security risks detected")
            exit_code = 1
        else:
            print("\n❌ SECURITY AUDIT FAILED - Significant security risks detected")
            exit_code = 2
        
        return exit_code
        
    except Exception as e:
        print(f"\n💥 Security audit failed with exception: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)