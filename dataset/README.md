# Data preparation

Benchmark files are not distributed in this package. The original source folder contains `exchange_rate.csv`, with columns `date,0,1,2,3,4,5,6,OT`. You can copy your authorized local copy into this directory for local use. Other datasets must be obtained separately.

The manuscript references [Time-Series-Library](https://github.com/thuml/Time-Series-Library) for benchmark access. Verify the specific release, original source terms, and preprocessing before comparing results.

| Loader | Format | Split in supplied code |
| --- | --- | --- |
| ETT hour / minute | CSV with `date` and numeric variables | Fixed 12/4/4 nominal-month train/validation/test boundaries; minute data multiply the hourly counts by 4 |
| Custom (Exchange, Weather, ECL, Traffic) | CSV with `date` and numeric variables | Chronological 70% / 10% / 20% when `train_only=False` |
| Solar | Comma-separated numeric rows, no header or date column | Chronological 70% / 10% / 20% |
| PEMS | NPZ with `data` shaped `[time, nodes, features]` | Chronological 60% / 20% / 20%; uses `data[:, :, 0]` |

Standardization is fitted on training data in the loaders. ETT and custom CSV validation/test windows overlap earlier history by `seq_len`; PEMS splits are sliced before window construction. Exact behavior is defined by `data_provider/data_loader.py`.

Set `--root_path`, `--data_path`, `--data`, `--enc_in`, and `--freq` for the dataset. For univariate/target-specific modes, also inspect `--features` and `--target`. The documentation command uses the available Exchange file format.

Raw files are ignored by Git. Do not assume a benchmark file is covered by a future repository code license.
