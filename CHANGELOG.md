# Changelog

All notable changes to **dmc2019-fraud-hydra-strategy** will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2026-11-22

### Added
- Multi-stage non-root production `Dockerfile` (`appuser`), `.dockerignore`, and `docker-compose.yml`.
- Single-command developer workflow `Makefile` (`install`, `test`, `lint`, `format`, `typecheck`, `run`, `docker-build`, `clean`).
- Security governance & automation: `.github/dependabot.yml`, `SECURITY.md`, Trivy container scanning, and Semgrep SAST step in CI.
- Operational runbook at `docs/runbook.md` covering batch scoring, checksum verification failures, and rollback procedures.
- Developer governance artifacts: `CONTRIBUTING.md`, `CHANGELOG.md`, and `.pre-commit-config.yaml`.

### Security
- Added explicit `.env` and `.env.*` exclusion patterns in `.gitignore` to guarantee zero credential leaks.
