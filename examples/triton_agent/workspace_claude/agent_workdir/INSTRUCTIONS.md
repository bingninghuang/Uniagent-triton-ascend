# Current Task

This file is overwritten by the host for every rollout.

The generated task will include:

- operator name
- target Ascend architecture
- PyTorch reference file path
- required implementation path
- compact validation commands

The agent should implement `ModelNew` in the required implementation file and
follow `CLAUDE.md` plus the local `triton-op-verifier` skill. Runtime workspaces
use this file and `CLAUDE.md` as the only required top-level markdown entry
points.
