# MDK-MOE

## Multi-view Decomposed Kalman Mixture of Experts for Non-stationary Time Series Forecasting

Research implementation accompanying **MDK-MoE: Multi-view Decomposed Kalman Mixture of Experts Framework for Non-stationary Time Series Forecasting**.

**Rui Hou, Yao Liu, Ruilin Jiang, Mengyao Lu, Jingbo Wang, Lanyi Zhang, and Qiao Liu**  
University of Electronic Science and Technology of China

[Method](#method) · [Setup](#setup) · [Datasets](#datasets) · [Usage](#usage) · [Reproducibility](docs/REPRODUCIBILITY.md) · [Citation](#citation)

> **Release status:** This package preserves the supplied Python and shell sources byte for byte. The supplied `exp/exp_basic.py` is included. The training entry point still has unresolved model-identifier and model-call integration issues. End-to-end training and the paper's results have not been verified from this snapshot. See [reproducibility notes](docs/REPRODUCIBILITY.md) before running experiments.

## Method

MDK-MoE integrates complementary decomposition views to model non-stationary time series:

1. **Embedding:** reversible instance normalization, linear embedding, and optional positional encoding.
2. **Multi-view Decomposition (MVD):** FFT, moving average, and Morlet-wavelet-based filtering produce six low/high-frequency representations.
3. **Maximizing Mutual Information (MMI):** a cross-view contrastive objective encourages consistency between complementary representations.
4. **Kalman MoE:** learned expert weights form a fused observation, and a learnable Kalman gain corrects a predicted latent state.
5. **Projection:** a residual connection, dropout, MLP prediction, and denormalization produce the forecast.

![MDK-MoE framework from the accompanying manuscript](assets/framework.png)

The manuscript evaluates long-term forecasting on nine benchmarks and short-term forecasting on four PEMS benchmarks. These are manuscript-reported experiments; this repository snapshot does not contain a verified reproduction of all 13 datasets.

## Repository structure

```text
MDK-MOE/
├── run_MDK_MoE.py              # Original experiment entry point
├── models/MDK_MoE.py           # Decomposition, contrastive objective, Kalman MoE, prediction
├── exp/                      # Original experiment base and training/evaluation implementation
├── data_provider/             # Data loaders and dispatch
├── layers/Embed.py            # Embedding components
├── RevIN/RevIN.py              # Reversible instance normalization
├── utils/                     # Metrics, time features, training utilities
├── scripts/exchange_rate_96.sh # Original Exchange hyperparameter sweep
├── dataset/README.md          # Data preparation; benchmark files excluded
├── assets/framework.png       # Original manuscript framework figure
├── docs/                      # Reproducibility, method mapping, attribution
├── paper_configs.csv          # Transcription of manuscript Appendix C.4, Table 9
├── references.bib             # Provisional manuscript citation
├── requirements.txt           # Import-derived dependencies; versions unverified
└── requirements-optional.txt  # Optional FLOP profiling dependency
```

`exp/exp_basic.py` is included as supplied, without reconstruction or modification.

## Setup

Use an isolated environment. The original training environment and CUDA/PyTorch versions have not been supplied, so the dependency list is **not a validated environment lock**.

```bash
git clone https://github.com/HRFarsighted/MDK-MOE.git
cd MDK-MOE
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

On Windows PowerShell, activate with `.\.venv\Scripts\Activate.ps1`. Match the PyTorch build to the intended CUDA environment before experiments. The original shell sweep assumes Bash; Linux or WSL is appropriate for that script. Runtime compatibility is pending validation.

Optional profiling dependencies:

```bash
python -m pip install -r requirements-optional.txt
```

## Datasets

Obtain benchmarks separately and place them under `dataset/`. The supplied source folder contains an Exchange CSV; benchmark data are not redistributed in this prepared package. The manuscript points to [Time-Series-Library](https://github.com/thuml/Time-Series-Library) for data access. Consult the original benchmark sources for their terms and provenance.

| Benchmark | CLI `--data` | Variables | Forecast horizon(s) in manuscript |
| --- | --- | ---: | --- |
| ETTh1 / ETTh2 / ETTm1 / ETTm2 | Matching dataset name | 7 | 96, 192, 336, 720 |
| ECL | `custom` | 321 | 96, 192, 336, 720 |
| Exchange | `custom` or `exchange` | 8 | 96, 192, 336, 720 |
| Weather | `weather` or `custom` | 21 | 96, 192, 336, 720 |
| Traffic | `traffic` or `custom` | 862 | 96, 192, 336, 720 |
| Solar | `Solar` | 137 | 96, 192, 336, 720 |
| PEMS03 / PEMS04 / PEMS07 / PEMS08 | Matching dataset name | Read from benchmark | 12 |

See [data preparation](dataset/README.md) for expected file formats and split behavior. Set `--enc_in` to the number of input variables.

## Usage

The existing experiment registry uses **`MDK_MoE`**, with an underscore. The repository name is `MDK-MOE`, and the paper refers to the method as `MDK-MoE`.

### Model interface

The model class is `models.MDK_MoE.Model`. Its interface is:

```python
# x: [batch_size, seq_len, enc_in]
loss_moe, loss_mi, forecast = model(x)
# forecast: [batch_size, pred_len, enc_in]
```

Required configuration attributes are `seq_len`, `pred_len`, `d_model`, `enc_in`, `level`, `threshold_ratio`, `moving_avg`, `dropout`, `pos`, `temperature`, and `gating_topk`.

### Experiment command template

**This command documents the existing CLI. It is not runnable end to end until the model-call integration is resolved.** It is a configuration example, not a verified reproduction command.

```bash
python -u run_MDK_MoE.py \
  --is_training 1 \
  --model_id Exchange_96_96 \
  --model MDK_MoE \
  --data custom \
  --root_path ./dataset/ \
  --data_path exchange_rate.csv \
  --features M \
  --seq_len 96 \
  --pred_len 96 \
  --enc_in 8 \
  --d_model 32 \
  --moving_avg 25 \
  --level 15 \
  --threshold_ratio 0.1 \
  --gating_topk 3 \
  --learning_rate 0.0001 \
  --batch_size 64 \
  --covar_loss 0.02 \
  --loss_moe 1 \
  --lradj cosine \
  --itr 1 \
  --num_workers 0
```

This illustrative command combines the Exchange input/dimension settings from Table 9 with `pred_len=96`, `covar_loss=0.02`, and `lradj=cosine` from the original Exchange sweep. Table 9's long-term configurations are explicitly described in the manuscript for horizon **720**; do not treat this example as the Table 9 reproduction.

The original `scripts/exchange_rate_96.sh` is a **448-configuration sweep**, not a single experiment. It also uses `MDK-MoE` instead of the registered `MDK_MoE`, and redirects logs to an uncreated nested directory. It is preserved as supplied; inspect the [known issues](docs/REPRODUCIBILITY.md) before use.

### Important parameters

| Argument | Meaning in the supplied implementation |
| --- | --- |
| `--seq_len`, `--pred_len` | Historical input length and forecast horizon |
| `--d_model` | Per-view latent feature dimension |
| `--enc_in` | Number of input variables |
| `--threshold_ratio` | FFT low-frequency mask ratio; see implementation for symmetric-bin selection |
| `--moving_avg` | Moving-average kernel size |
| `--level` | Morlet-wavelet convolution kernel size |
| `--temperature` | Contrastive objective temperature |
| `--gating_topk` | Expert rank threshold used in the gating transformation |
| `--covar_loss` | Weight of the returned cross-view contrastive loss |
| `--loss_moe` | Weight of expert load-balancing loss |

The supplied gating computes weights for all six views; `gating_topk` does not remove the other views. Parameter meanings above follow the code. Differences from manuscript terminology are recorded in [method mapping](docs/METHOD_MAPPING.md).

## Reproducibility

The original entry point fixes the initial seed to 2024 and uses AdamW. Training combines half-weighted Smooth L1 and MSE with the two auxiliary losses. Validation selects checkpoints using MSE. Test metrics are computed using `utils/metrics.py`.

Only static parsing and source-integrity checks were performed when preparing this package. No GPU training, runtime smoke test, checkpoint validation, or benchmark reproduction is claimed. Full commands for all datasets, the original environment versions, and verified checkpoints are not present in the supplied source snapshot.

See [reproducibility notes](docs/REPRODUCIBILITY.md), [paper configurations](paper_configs.csv), and [code-to-paper mapping](docs/METHOD_MAPPING.md).

## Citation

The following is a provisional manuscript citation based on the supplied LaTeX source. Replace it with the final proceedings citation and persistent identifier when available.

```bibtex
@unpublished{hou2026mdkmoe,
  title = {{MDK-MoE}: Multi-view Decomposed Kalman Mixture of Experts Framework for Non-stationary Time Series Forecasting},
  author = {Hou, Rui and Liu, Yao and Jiang, Ruilin and Lu, Mengyao and Wang, Jingbo and Zhang, Lanyi and Liu, Qiao},
  year = {2026},
  booktitle = {Advances in Neural Information Processing Systems}
}
```

## Attribution and license

See [attribution notes](docs/ATTRIBUTION.md). No repository-wide license has been selected in this prepared snapshot. Third-party code and dataset terms must be reconciled before an open-source release. A public repository alone does not supply an open-source license.
