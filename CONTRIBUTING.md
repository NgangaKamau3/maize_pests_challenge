# Contributing to Fall Armyworm Detection System

Thank you for considering contributing to the Fall Armyworm Detection System! This document provides guidelines and instructions for contributing to this project.

## Code of Conduct

By participating in this project, you agree to abide by our Code of Conduct. Please be respectful and considerate of others.

## How Can I Contribute?

### Reporting Bugs

If you find a bug, please create an issue using the bug report template. Include:

- A clear, descriptive title
- Steps to reproduce the issue
- Expected behavior vs. actual behavior
- Screenshots if applicable
- Environment details (OS, Python version, etc.)

### Suggesting Enhancements

For feature requests, create an issue using the feature request template. Include:

- A clear description of the feature
- Rationale for why this feature would be beneficial
- Any implementation ideas you have

### Pull Requests

1. Fork the repository
2. Create a new branch (`git checkout -b feature/your-feature-name`)
3. Make your changes
4. Run tests to ensure your changes don't break existing functionality
5. Commit your changes (`git commit -m 'Add some feature'`)
6. Push to the branch (`git push origin feature/your-feature-name`)
7. Open a Pull Request

#### Pull Request Guidelines

- Follow the coding style of the project
- Include tests for new features
- Update documentation as needed
- Keep PRs focused on a single change
- Link any relevant issues

## Development Setup

1. Clone the repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Set up the dataset:
   - Download the dataset from the source provided in the README
   - Extract to `PestDataset/Combined_pestDataset/`
   - **Note**: The dataset is not included in the repository due to size constraints

## Testing

Before submitting a PR, please run the tests to ensure your changes don't break existing functionality:

```bash
python -m unittest discover tests
```

## Code Style

- Follow PEP 8 guidelines
- Use meaningful variable and function names
- Add docstrings to functions and classes
- Keep functions focused on a single responsibility

## Documentation

If you're adding or changing features, please update the documentation accordingly:

- Update README.md if necessary
- Add or update docstrings
- Comment complex code sections

## Questions?

If you have any questions about contributing, please open an issue with the "question" label.

Thank you for your contributions!