# Contributing to dmc2019-fraud-hydra-strategy

Thank you for your interest in contributing! Please follow these engineering standards when submitting pull requests.

## Local Development Setup

```bash
git clone https://github.com/subki72/dmc2019-fraud-hydra-strategy.git
cd dmc2019-fraud-hydra-strategy
make install
```

## Quality Gates

Before opening a pull request, ensure all quality checks pass locally:

```bash
# Run unit & integration tests (coverage >= 80%)
make test

# Run Ruff linting & format checks
make lint

# Run static type checking
make typecheck
```

## Pre-Commit Hooks

Install pre-commit hooks to catch formatting issues and secret leaks automatically before commit:

```bash
pip install pre-commit
pre-commit install
```

## Commit Message Convention

We follow [Conventional Commits](https://www.conventionalcommits.org/):
- `feat(scope): ...` for new features
- `fix(scope): ...` for bug fixes
- `ci(scope): ...` for CI/CD pipeline changes
- `docs(scope): ...` for documentation updates
- `chore(scope): ...` for maintenance and governance
