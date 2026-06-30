# Symm rep learn

This is a code repository accompaning the paper "Representation Learning for Equivariant Inference with Guarantees". Read the [README](README.md).

# Jupyter Notebooks

Notebooks in this repository are designed tutorial like examples of the paper core experiments and contributions. Hence they should be educative, with self contained problem formulations and following the paper notation found in `.vscode/encp.pdf`. Pointing to the specific equations and sections in the paper is encouraged. Follow these guidelines when writing notebooks:

- You are the "educator" you dont write in the notebook to the user or to yourself, you write in the notebook to the "reader" so only explanation and exposition messages.

## Commit message guidelines

Generate commit messages using this exact format:

v<major>.<minor>.<patch>-<type>(<optional scope>): <short imperative summary>

- <codebase change 1>
- <codebase change ...>
- <Docs/Test change 1>
- <Version bump >

## Rules

- Always include the version in the header.
- Use lowercase for `<type>` and `<scope>`.
- Write the summary in imperative mood.
- Keep the first line concise and specific with important codebase changes.
- Add 2 to 5 bullets only when they add useful detail.
- Mention tests, docs, refactors, and version-file updates when relevant.
- Prefer these types: `feat`, `fix`, `perf`, `refactor`, `docs`, `test`, `build`, `ci`, `chore`.

## Version selection rules

- Version is exposed in `pyproject.toml`
- **MAJOR**: incompatible API changes, breaking public behavior changes, removed functionality, or changes requiring users to modify existing code or workflows.
- **MINOR**: backward-compatible new features, new public API additions, or new capabilities that do not break existing usage.
- **PATCH**: backward-compatible bug fixes, performance improvements, internal refactors without public behavior changes, documentation-only changes, test-only changes, or maintenance work.

## Pre-1.0 policy

- For versions `0.y.z`, still include the full version in the title.
- For pre-1.0 releases, use this policy:
  - breaking changes or new features: increment **MINOR** and reset **PATCH** to `0`
  - bug fixes, refactors, perf improvements, docs, tests, build, ci, and chore changes: increment **PATCH**
- Do not increment **MAJOR** above `0` unless the diff clearly indicates the project is moving to a stable `1.0.0` release.

## Type guidance

- `feat` = new user-facing or API-facing functionality
- `fix` = bug fix
- `perf` = performance improvement without intended feature change
- `refactor` = code restructuring without intended behavior change
- `docs` = documentation only
- `test` = tests only
- `build` = build-system or dependency packaging changes
- `ci` = CI/CD configuration changes
- `chore` = maintenance tasks not affecting product behavior

## Examples

- `v0.4.7-fix(stats): correct covariance computation and improve projection efficiency`
- `v0.4.8-perf(linearizer): reduce allocations in projection pipeline`
- `v1.0.0-feat(api): finalize stable public interface`
- `v2.0.0-fix(core): align implementation with the new breaking API contract`
