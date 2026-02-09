# AGENTS.md

This repository contains JAX/Flax (nnx) research code for parametric flows in Wasserstein space.
Use this document as the default operating guide for agentic coding changes.

## Repository Notes
- Language: Python
- ML stack: JAX, Flax (nnx), jaxtyping
- Notebooks and research scripts are present; tests are lightweight and may be long-running.
- No Cursor rules or Copilot rules were found in `.cursor/rules/`, `.cursorrules`, or `.github/copilot-instructions.md`.

## Build / Lint / Test Commands
### Environment
- No `pyproject.toml`, `setup.py`, or `requirements.txt` was found.
- Install dependencies manually as needed (e.g., `jax`, `flax`, `jaxtyping`, `matplotlib`).
- Prefer running inside a virtual environment.

### Build
- There is no build step. Run Python modules directly.

### Lint
- No linter configuration was found.
- If you add a linter later, document it here and follow its configuration.

### Tests
- Tests live in `tests/` and are plain Python scripts.
- Prefer running tests with `pytest` if available.

#### Run the full test suite
- `python -m pytest`

#### Run a single test file
- `python -m pytest tests/anderson_acceleration_test.py`

#### Run a single test by keyword
- `python -m pytest -k "anderson"`

#### Run as a standalone script (if pytest unavailable)
- `python tests/anderson_acceleration_test.py`

### Typical Research Scripts
- Notebooks and scripts may be long-running; do not execute automatically.
- When in doubt, ask before running heavy training or plotting.

## Code Style Guidelines
### Formatting
- Match the existing style: standard 4-space indentation.
- Keep line lengths readable; wrap long function calls across lines.
- Use blank lines to separate logical sections (imports, constants, classes).

### Imports
- Use standard library imports first, then third-party, then local modules.
- Keep imports explicit; avoid wildcard imports.
- Prefer importing JAX NumPy as `jnp`.
- Example order:
  - `import os`
  - `from pathlib import Path`
  - `import jax`
  - `import jax.numpy as jnp`
  - `from flax import nnx`
  - `from functionals.functional import Potential`

### Naming Conventions
- Modules, functions, and variables: `snake_case`.
- Classes: `PascalCase`.
- Constants: `UPPER_SNAKE_CASE` when truly constant.
- Use descriptive names for parameters and return values.

### Types and Shapes
- Use type hints where the code already does (e.g., `Array`, `ArrayLike`).
- Prefer `jaxtyping` for array typing and shape-aware annotations.
- Keep types simple in research code; do not over-annotate exploratory scripts.

### JAX / Flax Conventions
- Keep JAX functions pure when possible; avoid side effects inside `jit`-ed code.
- Prefer `jax.numpy` (`jnp`) operations instead of `numpy`.
- Use `nnx.Module` and `nnx.Linear` consistently with current patterns.
- Maintain existing initialization defaults (`xavier_uniform`, small bias std).

### Error Handling
- Raise `ValueError` for invalid user-provided parameters (e.g., unknown activation).
- Avoid swallowing exceptions; prefer explicit failures with informative messages.
- When adding new branches, include an `else` clause that raises an error.

### Logging and Printing
- Avoid adding noisy `print` statements to core modules.
- For debugging, prefer structured logging if added later; otherwise keep prints local.

### Data and File IO
- Avoid writing large artifacts by default in library code.
- When saving outputs, use explicit paths and clearly named files.
- Do not hardcode user-specific paths.

### Performance Considerations
- Prefer `jax.jit` for hot functions when it already exists.
- Minimize Python loops over arrays in performance-critical paths.
- Keep batch sizes and sample sizes configurable.

### Testing Style
- Keep tests deterministic where possible (seed RNGs explicitly).
- If a test is long-running, mark it or document runtime expectations.
- Prefer small sample sizes in tests to reduce runtime.

### Notebooks
- Avoid refactoring notebooks unless requested.
- If updating notebooks, keep outputs cleared unless explicitly asked.

### Documentation
- Update README only if user requests or if new commands are introduced.
- Keep inline comments minimal; explain non-obvious math only when requested.

## Contribution Checklist for Agents
- Identify relevant modules before editing.
- Keep changes minimal and aligned with existing patterns.
- Update or add tests only when explicitly requested.
- Mention any long-running commands before execution.
- Ask for clarification if requirements are ambiguous.

## Pointers to Key Modules
- Architectures: `architectures/architectures.py`
- Functionals: `functionals/*.py`
- Flows: `flows/*.py`
- Geometry helpers: `geometry/*.py`
- Parametric model: `parametric_model/parametric_model.py`
- Tests: `tests/`

## Notes on Running Experiments
- Some scripts write `.pkl` files and figures into the repo root.
- Prefer writing outputs to a dedicated results directory if adding new scripts.
- Confirm with the user before running GPU-heavy code.

## When Adding New Code
- Follow existing patterns for activation functions and model components.
- Keep new helpers in the closest relevant module rather than creating new files.
- If a new file is required, mirror the existing directory structure.
- Provide clear defaults and surface key parameters.

## If You Need Help
- Read `README.md` for the high-level project summary.
- Search for related functionality in `functionals/` and `flows/` before adding new code.
