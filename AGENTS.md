# Repository Guidelines

## Project Structure & Module Organization
Source code lives under `frog_rl/`. Core areas are `algorithms/` for training logic, `modules/` for model components, `networks/` for reusable layers, `storage/` for rollout buffers, `runners/` for training entry points, `env/` for environment wrappers, and `utils/` for shared helpers. Keep new model code close to the existing family it matches, for example `frog_rl/modules/transformer.py` for standalone encoders.

## Build, Test, and Development Commands
- `python3 -m pip install -e .` installs the package in editable mode for local development.
- `python3 -m py_compile frog_rl/<path>.py` is a quick syntax check when you change one or two files.
- `python3 -m pyright frog_rl` runs static type checking if `pyright` is available.

There is no dedicated test suite in this repository yet, so prefer small smoke checks after edits.

## Coding Style & Naming Conventions
Use Python 3.9+ with 4-space indentation and ASCII by default. Follow the existing RSL-RL-inspired style: explicit `nn.Module` classes, short docstrings, and descriptive method names such as `act`, `evaluate`, `update_normalization`, and `reset`. Use snake_case for functions and variables, PascalCase for classes, and keep config keys aligned with existing names like `amp_cfg`, `obs_groups`, and `num_layers`.

## Testing Guidelines
When adding or changing behavior, run a focused import or compile check on the touched module. For sequence models or buffers, include a tiny tensor-based smoke test that exercises the main forward path and expected output shape. If you add new public APIs, update the relevant doc file or README example alongside the code.

## Commit & Pull Request Guidelines
Recent commits use short, lower-case summaries such as `add amp`, `clean amp`, and `update amp wasabi`. Keep commit messages similarly concise and task-focused. Pull requests should describe the behavior change, list affected files or modules, and note any configuration changes. Include reproduction steps for bug fixes and sample configs when a feature changes training behavior.

## Agent-Specific Notes
Do not overwrite unrelated user changes. Prefer local repository patterns over inventing new abstractions, and keep edits narrowly scoped to the requested feature.
