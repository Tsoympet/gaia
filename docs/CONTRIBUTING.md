# Contributing to G.A.I.A

Thank you for your interest in contributing to G.A.I.A! We welcome contributions to improve the project.

## How to Contribute

1. **Fork the Repository**:
   - Click "Fork" on the GitHub repository page.

2. **Clone Your Fork**:
   ```bash
   git clone https://github.com/Tsoympet/gaia.git
   cd gaia

3.Create a Branch:

git checkout -b feature/your-feature-name

4.Make Changes:

- Follow the coding style in existing files.
- Add tests in tests/ for new features.
- Update documentation in docs/ if needed.

5.Commit Changes:

git commit -m "Add your feature description"

6.Push to Your Fork:

git push origin feature/your-feature-name

7.Open a Pull Request:
Go to the original repository and create a pull request.
Describe your changes and link any related issues.

Code Style

-Use PEP 8 for Python code.
-Keep JavaScript (extension) consistent with existing style (e.g., camelCase).
-Add docstrings for new functions and classes.

Issues

Report bugs or suggest features by opening a GitHub issue.
Use clear titles and provide detailed descriptions.

Testing

Add unit tests in tests/ using pytest.
Run tests with:

pytest tests/

By contributing, you agree that your contributions will be licensed under the MIT License.

#### 5. `CHANGELOG.md`
Tracks project versions and changes.

```markdown
# Changelog

## [2.4.0] - 2025-05-30
### Added
- Unrestricted knowledge acquisition via `KnowledgeAcquisitionModule`.
- General task execution with `TaskOrchestrator`.
- Enhanced browser extension for universal data scraping.
- Reinforcement learning for self-directed learning.
- Ethical safeguards with configurable strictness.

### Changed
- Updated `gaia.py` to integrate new modules.
- Refactored faction logic for broader task support.
- Improved GUI with task and knowledge tabs.

## [2.3.0] - 2025-04-15
### Added
- Faction-driven learning with five factions.
- Browser extension for internet access.
- Federated learning for knowledge sharing.

### Changed
- Enhanced `ValueAlignmentEngine` for ethical oversight.
- Updated GUI with faction monitoring.
