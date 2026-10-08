# Third-party notices

This release adapts the Apache-2.0 LLaMA-Factory training framework at upstream commit `97b32d31`. The upstream license
text is retained in [`LICENSE`](LICENSE). CIG-SFT adds the token-credit objective, erased-input processing, training
arguments, recipes, prepared data, and tests described in this repository.

The prepared math and code mixtures are rebuilt by `example/scripts/build_dataset.py` from their public dataset sources;
the manifests record the source identifiers, row counts, and hashes used for each snapshot. The external source licenses
apply to those datasets in addition to the Apache-2.0 license for this code release.

The paper-specific benchmark evaluation drivers are maintained separately and are not included in this training release.

The upstream framework's top-level demo-data directory is intentionally omitted. It contains unrelated sample records
and media; the CIG-SFT examples used by the recipes remain under `example/data/`.
