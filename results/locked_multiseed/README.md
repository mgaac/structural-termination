# Locked five-seed evidence bundle

This directory publishes the compact evidence for the
`validation-locked-v1` experiment:

- `protocol_manifest.json` freezes split sizes, generator seeds, dataset
  hashes, overlap checks, model seeds, supervision, rollout mode, and selection
  rules;
- `aggregate.json` records means and sample standard deviations across the five
  model seeds;
- `seeds/seed_*/locked_results.json` contains each seed's validation selections
  and ID/OOD test metrics.

The generated datasets, checkpoints, resolved machine-local configurations,
training logs, and aborted pilot runs remain local and are excluded from Git.
They contain workstation-specific paths and are not needed to audit the
published aggregate. `make locked-prepare` deterministically regenerates the
datasets and verifies zero exact graph overlap; `make locked` trains or resumes
the full experiment.

## Provenance boundary

The five runs were produced from commit `b8f8a8d` with a recorded dirty tracked
diff hash of `5bd792d7`. Untracked source files were not covered by that hash, so
the exact original training tree cannot be reconstructed from Git alone. The
published source exactly regenerated seed 11's complete `locked_results.json`
from its retained checkpoint during an independent audit, but this does not
retroactively establish bit-for-bit training provenance for all five runs.

Accordingly, these records support the reported evaluation results and the
bounded claims in the root README. They are not presented as a clean-source
training reproduction. A future clean rerun is required to close that gap.
