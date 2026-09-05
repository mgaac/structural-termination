# Locked no-supervision ablation

This directory contains the latest matched five-seed ablation in which the
termination-distance loss is disabled. It uses the same datasets, architecture,
optimizer, seeds, training budget, autoregressive evaluation, and
validation-only selection protocol as `../locked_multiseed/`.

- `protocol_manifest.json` records the frozen protocol and dataset hashes.
- `aggregate.json` summarizes the five no-supervision seeds.
- `seeds/seed_*/locked_results.json` contains the per-seed evidence.
- `comparison_to_supervised.json` contains the paired comparison used by the
  root README.

Machine-local datasets, checkpoints, configurations, and logs are excluded
from Git. Run `make locked-no-supervision` to train or resume this treatment,
then `make locked-compare` to regenerate the paired comparison.
