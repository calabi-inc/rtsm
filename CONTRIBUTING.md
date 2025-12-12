# Contributing to RTSM

Thanks for your interest in contributing to RTSM! This document provides guidelines for contributing to the project.

## Getting Started

1. Fork the repository
2. Clone your fork:
   ```bash
   git clone https://github.com/YOUR_USERNAME/rtsm.git
   cd rtsm
   ```
3. Install in development mode:
   ```bash
   pip install -e ".[dev]"
   ```
4. Create a branch for your changes:
   ```bash
   git checkout -b feat/your-feature-name
   ```

## Development Setup

### Prerequisites

- Python 3.10+
- CUDA-capable GPU
- RGB-D camera (optional, for end-to-end testing)

### Running Tests

```bash
pytest tests/
```

### Code Style

- Use [black](https://github.com/psf/black) for formatting
- Use [ruff](https://github.com/astral-sh/ruff) for linting
- Type hints are encouraged

```bash
black rtsm/
ruff check rtsm/
```

## Making Changes

### Commit Messages

Use bracket prefixes:

```
[feat] add new segmentation adapter
[fix] handle null depth values
[docs] update installation guide
[refactor] simplify association logic
[test] add tests for proximity index
[chore] update dependencies
```

### Pull Request Process

1. Ensure your code passes tests and linting
2. Update documentation if needed
3. Keep PRs focused — one feature or fix per PR
4. Write a clear PR description explaining:
   - What the change does
   - Why it's needed
   - How to test it

### PR Title Format

```
[feat] add YOLO-World adapter
[fix] memory leak in working memory store
[docs] add API examples
```

## What to Contribute

### Good First Issues

Look for issues labeled `good first issue` or `help wanted`.

### Ideas Welcome

- New segmentation model adapters (YOLO-World, SAM2, etc.)
- SLAM integrations (ORB-SLAM3, etc.)
- Communication protocols (ROS 2, MQTT, Kafka)
- Performance optimizations
- Documentation improvements
- Bug fixes

### Before Starting Major Work

For significant changes, please open an issue first to discuss the approach. This helps avoid duplicate work and ensures alignment with project direction.

## Project Structure

```
rtsm/
├── core/           # Pipeline, association, data models
├── models/         # FastSAM, CLIP adapters
├── stores/         # Working memory, proximity index, vector stores
├── io/             # ZeroMQ ingestion, frame buffering
├── api/            # REST API server
├── visualization/  # WebSocket server, 3D demo
└── utils/          # Helpers, transforms
```

## Questions?

- Open an issue for bugs or feature requests
- Reach out to [Calabi](https://github.com/calabi-inc) for collaboration inquiries

## License

By contributing, you agree that your contributions will be licensed under the Apache-2.0 License.
