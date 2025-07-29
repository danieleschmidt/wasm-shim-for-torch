# Security Policy

## Supported Versions

| Version | Supported          |
| ------- | ------------------ |
| 0.1.x   | :white_check_mark: |

## Reporting a Vulnerability

We take security vulnerabilities seriously. If you discover a security vulnerability in WASM Shim for Torch, please report it responsibly:

### 🔒 Private Disclosure

**DO NOT** create a public GitHub issue for security vulnerabilities.

Instead, please:

1. **Email**: Send details to `security@yourdomain.com`
2. **Subject Line**: `[SECURITY] WASM Shim for Torch - [Brief Description]`
3. **Include**:
   - Description of the vulnerability
   - Steps to reproduce
   - Potential impact
   - Suggested fix (if any)
   - Your contact information

### 📅 Response Timeline

- **Acknowledgment**: Within 24 hours
- **Initial Assessment**: Within 48 hours  
- **Status Update**: Within 7 days
- **Fix Timeline**: Varies by severity (critical: 72 hours, high: 1 week, medium: 2 weeks)

### 🏆 Recognition

Security researchers who responsibly disclose vulnerabilities will be:
- Acknowledged in our security advisories (unless you prefer to remain anonymous)
- Listed in our CONTRIBUTORS.md file
- Invited to test the fix before public release

### 🔐 Security Measures

Our security practices include:

**Code Security**:
- Static analysis with CodeQL and Bandit
- Dependency vulnerability scanning with Safety
- Regular security audits
- Secure coding guidelines

**Browser Security**:
- WASM sandbox execution only
- No arbitrary code execution
- Memory safety through WASM constraints
- Cross-origin isolation requirements

**Infrastructure Security**:
- Automated dependency updates via Dependabot
- Container security scanning
- SBOM (Software Bill of Materials) generation
- Supply chain security verification

### 📋 Security Checklist

When reporting vulnerabilities, please consider:

- [ ] Can this be exploited in a browser environment?
- [ ] Does this affect WASM sandboxing?
- [ ] Could this lead to memory corruption?
- [ ] Are user credentials or data at risk?
- [ ] Is there potential for remote code execution?

### 🚨 Emergency Contact

For critical vulnerabilities that could cause immediate harm:
- **Email**: `security-urgent@yourdomain.com`
- **Expected Response**: Within 2 hours during business hours

Thank you for helping keep WASM Shim for Torch secure! 🛡️