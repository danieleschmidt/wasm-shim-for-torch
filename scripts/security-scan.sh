#!/bin/bash
# Comprehensive security scanning script for WASM Shim for Torch
# Runs multiple security tools and generates consolidated report

set -euo pipefail

# Configuration
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPORTS_DIR="${PROJECT_ROOT}/security-reports"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
CONSOLIDATED_REPORT="${REPORTS_DIR}/security_report_${TIMESTAMP}.json"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Create reports directory
mkdir -p "${REPORTS_DIR}"

echo -e "${BLUE}Starting comprehensive security scan...${NC}"
echo "Project: WASM Shim for Torch"
echo "Timestamp: $(date)"
echo "Reports directory: ${REPORTS_DIR}"
echo

# Function to log messages
log() {
    echo -e "${BLUE}[$(date '+%H:%M:%S')]${NC} $1"
}

# Function to handle errors
error() {
    echo -e "${RED}[ERROR]${NC} $1" >&2
}

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Initialize consolidated report
init_report() {
    cat > "${CONSOLIDATED_REPORT}" << EOF
{
  "scan_info": {
    "timestamp": "$(date --iso-8601=seconds)",
    "project": "WASM Shim for Torch",
    "version": "0.1.0",
    "scanner_version": "1.0.0"
  },
  "summary": {
    "total_issues": 0,
    "critical": 0,
    "high": 0,
    "medium": 0,
    "low": 0,
    "info": 0
  },
  "scans": {}
}
EOF
}

# Function to update summary in consolidated report
update_summary() {
    local tool="$1"
    local critical="${2:-0}"
    local high="${3:-0}"
    local medium="${4:-0}"
    local low="${5:-0}"
    local info="${6:-0}"
    
    # Update individual tool summary
    python3 << EOF
import json

with open("${CONSOLIDATED_REPORT}", 'r') as f:
    report = json.load(f)

report['scans']['${tool}'] = {
    'critical': ${critical},
    'high': ${high},
    'medium': ${medium},
    'low': ${low},
    'info': ${info},
    'total': ${critical} + ${high} + ${medium} + ${low} + ${info}
}

# Update overall summary
report['summary']['critical'] += ${critical}
report['summary']['high'] += ${high}
report['summary']['medium'] += ${medium}
report['summary']['low'] += ${low}
report['summary']['info'] += ${info}
report['summary']['total_issues'] = (
    report['summary']['critical'] + 
    report['summary']['high'] + 
    report['summary']['medium'] + 
    report['summary']['low'] + 
    report['summary']['info']
)

with open("${CONSOLIDATED_REPORT}", 'w') as f:
    json.dump(report, f, indent=2)
EOF
}

# 1. Bandit - Python code security analysis
run_bandit() {
    log "Running Bandit (Python security analysis)..."
    
    if ! command_exists bandit; then
        error "Bandit not installed. Install with: pip install bandit[toml]"
        return 1
    fi
    
    local bandit_report="${REPORTS_DIR}/bandit_${TIMESTAMP}.json"
    
    cd "${PROJECT_ROOT}"
    if bandit -r src/ -f json -o "${bandit_report}" 2>/dev/null; then
        log "Bandit scan completed successfully"
        
        # Parse results
        if [[ -f "${bandit_report}" ]]; then
            local high_issues=$(python3 -c "
import json
with open('${bandit_report}', 'r') as f:
    data = json.load(f)
print(len([r for r in data.get('results', []) if r['issue_severity'] == 'HIGH']))
")
            local medium_issues=$(python3 -c "
import json
with open('${bandit_report}', 'r') as f:
    data = json.load(f)
print(len([r for r in data.get('results', []) if r['issue_severity'] == 'MEDIUM']))
")
            local low_issues=$(python3 -c "
import json
with open('${bandit_report}', 'r') as f:
    data = json.load(f)
print(len([r for r in data.get('results', []) if r['issue_severity'] == 'LOW']))
")
            
            update_summary "bandit" 0 "${high_issues}" "${medium_issues}" "${low_issues}" 0
        fi
    else
        error "Bandit scan failed"
        update_summary "bandit" 0 0 0 0 0
    fi
}

# 2. Safety - Python dependency vulnerability check
run_safety() {
    log "Running Safety (dependency vulnerability check)..."
    
    if ! command_exists safety; then
        error "Safety not installed. Install with: pip install safety"
        return 1
    fi
    
    local safety_report="${REPORTS_DIR}/safety_${TIMESTAMP}.json"
    
    cd "${PROJECT_ROOT}"
    if safety check --json --output "${safety_report}" 2>/dev/null || true; then
        log "Safety scan completed"
        
        # Parse results
        if [[ -f "${safety_report}" ]]; then
            local vulnerabilities=$(python3 -c "
import json
try:
    with open('${safety_report}', 'r') as f:
        data = json.load(f)
    print(len(data))
except:
    print(0)
")
            update_summary "safety" 0 "${vulnerabilities}" 0 0 0
        fi
    else
        log "Safety scan completed (no vulnerabilities or errors)"
        update_summary "safety" 0 0 0 0 0
    fi
}

# 3. pip-audit - Python package vulnerability scanner
run_pip_audit() {
    log "Running pip-audit (package vulnerability scanner)..."
    
    if ! command_exists pip-audit; then
        error "pip-audit not installed. Install with: pip install pip-audit"
        return 1
    fi
    
    local audit_report="${REPORTS_DIR}/pip_audit_${TIMESTAMP}.json"
    
    cd "${PROJECT_ROOT}"
    if pip-audit --format=json --output="${audit_report}" 2>/dev/null || true; then
        log "pip-audit scan completed"
        
        # Parse results
        if [[ -f "${audit_report}" ]]; then
            local vulnerabilities=$(python3 -c "
import json
try:
    with open('${audit_report}', 'r') as f:
        data = json.load(f)
    print(len(data.get('vulnerabilities', [])))
except:
    print(0)
")
            update_summary "pip_audit" 0 "${vulnerabilities}" 0 0 0
        fi
    else
        log "pip-audit scan completed (no vulnerabilities)"
        update_summary "pip_audit" 0 0 0 0 0
    fi
}

# 4. Semgrep - Static analysis for security issues
run_semgrep() {
    log "Running Semgrep (static analysis)..."
    
    if ! command_exists semgrep; then
        log "Semgrep not installed. Skipping..."
        return 0
    fi
    
    local semgrep_report="${REPORTS_DIR}/semgrep_${TIMESTAMP}.json"
    
    cd "${PROJECT_ROOT}"
    if semgrep --config=auto --json --output="${semgrep_report}" src/ 2>/dev/null || true; then
        log "Semgrep scan completed"
        
        # Parse results
        if [[ -f "${semgrep_report}" ]]; then
            local critical_issues=$(python3 -c "
import json
try:
    with open('${semgrep_report}', 'r') as f:
        data = json.load(f)
    print(len([r for r in data.get('results', []) if r.get('extra', {}).get('severity') == 'ERROR']))
except:
    print(0)
")
            local medium_issues=$(python3 -c "
import json
try:
    with open('${semgrep_report}', 'r') as f:
        data = json.load(f)
    print(len([r for r in data.get('results', []) if r.get('extra', {}).get('severity') == 'WARNING']))
except:
    print(0)
")
            local info_issues=$(python3 -c "
import json
try:
    with open('${semgrep_report}', 'r') as f:
        data = json.load(f)
    print(len([r for r in data.get('results', []) if r.get('extra', {}).get('severity') == 'INFO']))
except:
    print(0)
")
            
            update_summary "semgrep" "${critical_issues}" 0 "${medium_issues}" 0 "${info_issues}"
        fi
    else
        log "Semgrep scan completed with no issues"
        update_summary "semgrep" 0 0 0 0 0
    fi
}

# 5. Generate SBOM
generate_sbom() {
    log "Generating Software Bill of Materials (SBOM)..."
    
    local sbom_file="${REPORTS_DIR}/sbom_${TIMESTAMP}.json"
    
    if [[ -x "${PROJECT_ROOT}/scripts/generate-sbom.py" ]]; then
        cd "${PROJECT_ROOT}"
        if python3 scripts/generate-sbom.py --output "${sbom_file}" --summary; then
            log "SBOM generated successfully"
        else
            error "SBOM generation failed"
        fi
    else
        error "SBOM generator not found or not executable"
    fi
}

# 6. License compliance check
check_licenses() {
    log "Checking license compliance..."
    
    local license_report="${REPORTS_DIR}/licenses_${TIMESTAMP}.txt"
    
    cd "${PROJECT_ROOT}"
    
    # Check pip-licenses if available
    if command_exists pip-licenses; then
        pip-licenses --format=json > "${license_report}.json" 2>/dev/null || true
        pip-licenses --format=plain > "${license_report}" 2>/dev/null || true
        log "License information collected"
    else
        log "pip-licenses not available. Install with: pip install pip-licenses"
    fi
}

# 7. Secret scanning
scan_secrets() {
    log "Scanning for secrets and sensitive information..."
    
    local secrets_report="${REPORTS_DIR}/secrets_${TIMESTAMP}.txt"
    
    cd "${PROJECT_ROOT}"
    
    # Simple regex-based secret detection
    {
        echo "=== Secret Scanning Report ==="
        echo "Timestamp: $(date)"
        echo
        
        # Check for common secret patterns
        echo "Checking for potential secrets..."
        
        # API keys, tokens, passwords
        if grep -r -i -n \
            -E "(api[_-]?key|secret|token|password|pwd)\s*[:=]\s*['\"]?[a-zA-Z0-9]{8,}" \
            src/ tests/ --include="*.py" 2>/dev/null || true; then
            echo "Found potential secrets (review manually)"
        else
            echo "No obvious secrets found"
        fi
        
        echo
        echo "Checking for hardcoded credentials..."
        
        # Hardcoded credentials
        if grep -r -i -n \
            -E "(password|passwd|pwd)\s*[:=]\s*['\"][^'\"]{3,}['\"]" \
            src/ tests/ --include="*.py" 2>/dev/null || true; then
            echo "Found potential hardcoded credentials"
        else
            echo "No hardcoded credentials found"
        fi
        
    } > "${secrets_report}"
    
    log "Secret scanning completed"
}

# Generate final report
generate_final_report() {
    log "Generating final security report..."
    
    # Add metadata to consolidated report
    python3 << EOF
import json
import os

with open("${CONSOLIDATED_REPORT}", 'r') as f:
    report = json.load(f)

# Add file paths
report['files'] = {
    'bandit': 'bandit_${TIMESTAMP}.json',
    'safety': 'safety_${TIMESTAMP}.json',
    'pip_audit': 'pip_audit_${TIMESTAMP}.json',
    'semgrep': 'semgrep_${TIMESTAMP}.json',
    'sbom': 'sbom_${TIMESTAMP}.json',
    'licenses': 'licenses_${TIMESTAMP}.txt',
    'secrets': 'secrets_${TIMESTAMP}.txt'
}

# Add recommendations
recommendations = []
if report['summary']['critical'] > 0:
    recommendations.append('Address critical security issues immediately')
if report['summary']['high'] > 0:
    recommendations.append('Review and fix high-severity issues')
if report['summary']['medium'] > 0:
    recommendations.append('Plan to address medium-severity issues')

report['recommendations'] = recommendations

# Determine overall security posture
total_critical_high = report['summary']['critical'] + report['summary']['high']
if total_critical_high == 0:
    security_posture = 'GOOD'
elif total_critical_high < 5:
    security_posture = 'MODERATE'
else:
    security_posture = 'NEEDS_ATTENTION'

report['security_posture'] = security_posture

with open("${CONSOLIDATED_REPORT}", 'w') as f:
    json.dump(report, f, indent=2)
EOF

    log "Final report generated: ${CONSOLIDATED_REPORT}"
}

# Print summary
print_summary() {
    echo
    echo -e "${BLUE}=== Security Scan Summary ===${NC}"
    
    python3 << EOF
import json
with open("${CONSOLIDATED_REPORT}", 'r') as f:
    report = json.load(f)

summary = report['summary']
posture = report['security_posture']

print(f"Overall Security Posture: {posture}")
print(f"Total Issues Found: {summary['total_issues']}")
print(f"  Critical: {summary['critical']}")
print(f"  High: {summary['high']}")
print(f"  Medium: {summary['medium']}")
print(f"  Low: {summary['low']}")
print(f"  Info: {summary['info']}")
print()

if report.get('recommendations'):
    print("Recommendations:")
    for rec in report['recommendations']:
        print(f"  • {rec}")
    print()

print(f"Detailed reports available in: ${REPORTS_DIR}")
print(f"Consolidated report: ${CONSOLIDATED_REPORT}")
EOF
}

# Main execution
main() {
    log "Initializing security scan..."
    init_report
    
    # Run all security scans
    run_bandit
    run_safety
    run_pip_audit
    run_semgrep
    generate_sbom
    check_licenses
    scan_secrets
    
    # Generate final report and summary
    generate_final_report
    print_summary
    
    echo
    log "Security scan completed successfully!"
    
    # Exit with appropriate code based on findings
    python3 << EOF
import json
import sys
with open("${CONSOLIDATED_REPORT}", 'r') as f:
    report = json.load(f)
    
total_critical_high = report['summary']['critical'] + report['summary']['high']
if total_critical_high > 0:
    sys.exit(1)  # Fail if critical or high issues found
else:
    sys.exit(0)  # Success
EOF
}

# Handle command line arguments
case "${1:-}" in
    --help|-h)
        echo "Usage: $0 [options]"
        echo "Options:"
        echo "  --help, -h    Show this help message"
        echo "  --quick       Run only basic scans"
        echo "  --full        Run comprehensive scan (default)"
        exit 0
        ;;
    --quick)
        log "Running quick security scan..."
        init_report
        run_bandit
        run_safety
        generate_final_report
        print_summary
        ;;
    --full|"")
        main
        ;;
    *)
        error "Unknown option: $1"
        exit 1
        ;;
esac