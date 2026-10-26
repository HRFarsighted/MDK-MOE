# Reproducibility status

This repository is a documentation and packaging snapshot of the supplied research code. Existing `.py` and `.sh` files were copied without edits. `SOURCE_MANIFEST.json` lists their sizes and SHA-256 hashes. Cache files, raw benchmark data, manuscript build files, and editor settings were excluded.

## Confirmed static integration issues

| Issue | Evidence | Effect |
| --- | --- | --- |
| Inconsistent model identifier | Experiment registry key is `MDK_MoE`; shell sweep uses `MDK-MoE` | Once imports are restored, the original sweep's model name cannot resolve in the registry |
| Inconsistent forward calls | `Model.forward(self, x)` accepts one input; train/validation/test/prediction use multi-input calls for names without `Linear` | The registered `MDK_MoE` follows incompatible call branches |
| Log directory missing | Shell sweep creates `logs/LongForecasting`, but redirects into `logs/LongForecasting/9.30/exchange/` | Redirection fails before Python starts unless that nested directory is created |

These issues were identified by source inspection. No original source was edited to resolve them. The supplied `exp/exp_basic.py` is now included and defines the imported `Exp_Basic` class. The missing-base-file issue is resolved at the source level; runtime import and training remain unverified. Reconcile the remaining entry-point integration before claiming reproduction.

## Available and missing artifacts

Available: model implementation, dataset loaders, entry point, experiment base (`exp/exp_basic.py`), experiment implementation, one Exchange sweep, and manuscript Table 9 configuration transcription.

Not supplied: full per-horizon/per-dataset run scripts, verified environment versions, reference training logs, trained checkpoints, and the complete set of 13 benchmark datasets. The supplied Exchange CSV remains in the original local source folder and is excluded from redistribution here.

## Configuration provenance

`paper_configs.csv` transcribes the 13 dataset rows of Appendix C.4, Table 9 in the supplied manuscript. The caption text specifies forecast horizon 720 for long-term datasets and 12 for PEMS. The table does not specify every CLI argument. In particular, it does not provide `covar_loss`, `temperature`, scheduler, or all data-loader settings. Do not treat it as a complete executable configuration.

The original Exchange script enumerates 2 wavelet kernel sizes × 2 learning rates × 4 batch sizes × 4 history lengths × 7 latent dimensions = **448 runs**, each with one repetition. This differs from a minimal reproduction command.

## Environment and evaluation

`requirements.txt` is derived from imports, not from the authors' original environment. Python, PyTorch, CUDA, NumPy, pandas, and other versions remain to be verified. For example, the PEMS loader uses `DataFrame.fillna(method=...)`; compatibility must be checked against the chosen pandas release.

The entry point seeds Python, NumPy, and PyTorch with 2024 once before its repetition loop. This does not establish deterministic GPU execution or a multi-seed uncertainty estimate.

Training uses AdamW and `0.5 * SmoothL1 + 0.5 * MSE + covar_loss * loss_mi + loss_moe * loss_moe_term`. Validation uses MSE. See the source for actual metric scaling and output behavior; do not label the returned test metrics as original-unit metrics without checking the loader and inversion path.

## Validation performed for this package

- SHA-256 and byte equality for every included original Python/shell file.
- Python AST parsing without importing or executing the experiment.
- Local Markdown link, figure, configuration-row, and archive checks.
- No runtime smoke test or end-to-end experiment; PyTorch was unavailable in the packaging environment.

Before a reproduction release, resolve the remaining runtime integration, recover the tested environment, validate one minimal train/test run, and then verify all reported experiment configurations against logs and checkpoints.
