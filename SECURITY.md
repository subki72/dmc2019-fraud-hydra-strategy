# Security Policy

## Supported Versions

| Version | Supported          |
| ------- | ------------------ |
| 1.0.x   | :white_check_mark: |
| < 1.0   | :x:                |

## Reporting a Vulnerability

We take the security of **dmc2019-fraud-hydra-strategy** seriously. If you discover a security vulnerability, please follow responsible disclosure practices:

1. **Do NOT open a public GitHub issue** for undisclosed vulnerabilities.
2. Email the repository maintainer via GitHub Security Advisories (`Security -> Advisories -> Report a vulnerability`) with:
   - Affected file paths and line numbers.
   - Steps to reproduce or minimal Proof-of-Concept (PoC).
   - Impact assessment and suggested remediation (if available).
3. You will receive an acknowledgement within **48 hours**, and a remediation timeline within **5 business days**.

## Security Controls in Place

- **Model Artifact Integrity**: SHA-256 checksum verification (`ModelBundle.load(..., verify_checksum=True)`) and path confinement protection against directory traversal.
- **Secret Hygiene**: `.env` files are strictly excluded via `.gitignore`; zero credentials are committed to version control.
- **Automated Scanning**: Dependabot weekly alerts, Semgrep SAST static analysis, and Trivy container vulnerability scans run on every pull request and push to `main`.
- **Container Hardening**: Multi-stage Docker build executing under an unprivileged non-root user (`appuser`).
