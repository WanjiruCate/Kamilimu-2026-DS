# Case study 3: which power source is on?

GridWatch sensors in six Sierra Leone hospitals report voltage and frequency about every two minutes. They do not report whether the load is on the grid, on solar, or on a generator. nLine's current system fills that in with hand-written rules. This folder is the KamiLimu pipeline that reads the **entire** extract, rebuilds those rules as a baseline, fits several time-series models, explains the chosen model with SHAP, and serves it from an API a web page or a phone can call.

The teaching notebook is [`../case_study_3_detecting_generator_usage.ipynb`](../case_study_3_detecting_generator_usage.ipynb). It reads the artifacts already on disk, so a class can discuss the results without refitting. This file is the long version: what the data is, what each model is doing, what the numbers mean, and how to run the same job on Kaggle when a laptop cannot hold the file.

The numbers below are from the run saved in `artifacts/` on 15 October 2026 (`model` version 1.0.0, trained 2026-10-15T09:08:45Z, scikit-learn 1.9.0). A later `python -m pipeline.train` rewrites them.

## The decision

Each two-minute reading is one of three classes:

| Class | What it means in the case |
| --- | --- |
| `grid` | The hospital is taking utility power |
| `solar` | The hospital is on its solar plant |
| `generator` | The hospital is on a diesel generator |

`unknown` rows (12,940 of them) are kept in the ledger and left out of fitting and scoring. They are often a dead sensor: voltage around 2 V, or a frequency nowhere near 50 Hz.

The case also asks for **how long** each room spent on each source. The pipeline answers that on the February–March 2024 holdout by turning consecutive readings into durations. See [Usage hours](#usage-hours).

## Why earlier teams stopped near 2.1 million rows

The extract has a header plus **5,125,181** data rows. Two ordinary tools throw most of them away, and both failures look like "the file has about 2.1 million rows":

- Excel's worksheet limit is 1,048,576 rows. Two sheets are 2,097,152 rows.
- `pd.read_csv` on the 789 MB file builds one big object-dtype frame. On an 8 GB laptop that allocation dies, and whatever is left in memory is again a couple of million rows.

The loader does not do that. `pipeline/load.py` counts newlines first, then reads the csv in chunks of 100,000 rows with explicit dtypes (`float32` voltage and frequency, strings for the ids), and writes one parquet file per sensor. After the load it asserts the pandas row count equals 5,125,181. In this extract that check passed with **0 malformed lines** and **0 duplicate timestamps**.

Do not "clean" the file down to the rows that look convenient. The point of the ledger is that every record is accounted for: modelled, or named as unknown, or counted as a collapsed duplicate. Here the duplicate count is zero.

## What is in this folder

```
case_study_3/
  README.md                  this file
  bootstrap.py               sets paths before pipeline is imported (Kaggle)
  kaggle_power_source.ipynb  run-all notebook for a Kaggle CPU session
  requirements.txt           training stack, including shap
  pipeline/                  load, features, models, SHAP, training
  service/                   FastAPI, web console, Java client, Docker
  artifacts/                 model.joblib, metrics, figures, SHAP
  work/                      local cache (gitignored): parquet and feature npz
  tests/                     small checks that do not need the 789 MB file
```

`work/` can be deleted. The next train will re-read the csv, which is the slow part. Do not delete `artifacts/` unless you mean to retrain: the API and the teaching notebook both read it.

## Run it on a laptop

From `Deploying ML Solutions/case_study_3`, with the csv at the path the course repo already uses:

```bash
python -m pip install -r requirements.txt
python -m pipeline.train
```

The first run reads 789 MB once, partitions 35 sensors, builds features, fits every model, writes figures, and then runs SHAP. On the machine that produced these artifacts the modelling finished in about four minutes **after** the csv was already on disk. The memory risk is the single-frame read, which this code does not do. If the laptop still swaps or the kernel dies, use Kaggle. Do not drop sensors to make it fit.

A second run reuses `work/` and only refits. Set nothing to force a feature rebuild unless the feature code changed (`FEATURE_VERSION` in `pipeline/config.py`, or `run(force_rebuild=True)`).

SHAP alone, against a model that is already saved:

```bash
python -m pipeline.shap_analysis
```

That needs `work/features/tabular_train.npz` and `tabular_test.npz` from a completed feature build. It does not retrain.

Then the API, from `service/`:

```bash
python -m uvicorn api:app --host 127.0.0.1 --port 8010
```

Open http://127.0.0.1:8010/ for the console and http://127.0.0.1:8010/docs for the interactive schema.

## Run it on Kaggle

Kaggle's CPU notebook has more RAM than an 8 GB laptop. A GPU is not used. The boosting model, the logistic model, and the small NumPy RNN/LSTM are all CPU models. Internet has to be on for the first cell, because the kernel needs `sktime`, `numba`, and `shap`.

### What to upload

1. Zip the `case_study_3` folder. **Leave `work/` out.** That directory is a cache of the same data and can be larger than the csv. `artifacts/` can stay in the zip or stay out; training rewrites it.
2. On kaggle.com, **Datasets → New Dataset**, upload the zip, and create it. Kaggle unpacks datasets under `/kaggle/input/<name>/`.
3. Create a **second** dataset from `labeled_power_source_data.csv` (789 MB). Keep that file name if you can. The finder also accepts the largest csv under `/kaggle/input` when the name differs.

### What to click

1. **Create → New Notebook**.
2. Settings (the right-hand panel): **Internet On**, **Accelerator None**. Persistence of `/kaggle/working` is optional; download `artifacts/` at the end either way.
3. **Add Data** and attach both datasets.
4. **File → Import Notebook** and choose `kaggle_power_source.ipynb`, or paste its cells. The notebook lives next to this README.
5. **Run All**.

The first code cell searches `/kaggle/input` for `bootstrap.py` and for the csv, then sets:

| Variable | Value on Kaggle | Why |
| --- | --- | --- |
| `POWER_SOURCE_WORK` | `/kaggle/working/work` | input datasets are read-only |
| `ARTIFACT_DIR` | `/kaggle/working/artifacts` | models and figures have to be writable |
| `POWER_SOURCE_CSV` | the csv that was found | `pipeline.config` reads this at import time |

`bootstrap.configure()` must run **before** `import pipeline`. The directories are bound when `pipeline.config` is first imported. Setting the variables afterwards does nothing.

When the run finishes, download `/kaggle/working/artifacts` and copy it over `case_study_3/artifacts` on the laptop. The API and the teaching notebook will then show the Kaggle model without another fit.

Expect the same order of magnitude as the local run (a few minutes of modelling, plus about a minute for SHAP) once the csv is attached. The session is a CPU session on purpose.

## The data

File: `Kamilimu_Case_Study_Dataset/Kamilimu_Case_Study_Dataset/Sierra Leone/labeled_power_source_data.csv`.

| | |
| --- | --- |
| Rows | 5,125,181 plus a header |
| Span | 2023-04-25 18:20:06 to 2024-03-31 23:58:06 |
| Sensors | 35, column `respondent_id` |
| Sites | princess_christian (1,483,567), ola_during (1,163,483), masanga (863,654), kambia (733,453), kabala (511,654), bonthe (369,370) |
| Cadence | median gap 120 seconds; about 0.23% of gaps are longer than 10 minutes |
| Labels | grid 2,403,874 · solar 2,160,602 · generator 547,765 · unknown 12,940 |

Columns used: `respondent_id`, `time`, `site_id`, `room`, `voltage`, `frequency`, `available_power_sources`, `active_power_source`. The file's first column is an unnamed index. In this extract it runs `0 … 5125180` with no resets, so it is not a sensor id and it is not used as a feature.

`available_power_sources` is the installation survey (what the room *can* run). `active_power_source` is the rule system's guess at what *is* running. There is no independent meter label in the extract. Beating the rules is "reproducing the rule system more smoothly", not proof against a fuel log. The notebook says this, and the model card should keep saying it.

Survey strings in the file include `Solar &amp; Generator` with the HTML ampersand. `source_flags` decodes that before it looks for the words grid, solar, and generator.

### Clocks

Sierra Leone is on UTC. The csv stores naive wall-clock times. Feature code converts them with `datetime64[ns]` nanoseconds, which keeps the hour that was written down. Do not call `.timestamp()` on a naive value and then `fromtimestamp` on a Kenya laptop: that pair shifts the hour-of-day features by three hours (UTC+3). The API treats ISO timestamps the same way, as the clock time in the file.

## Labels and the rule baseline

The written system, reconstructed in `pipeline/rules.py`:

1. If the survey says the room has only one source, that source is the label. Frequency is not allowed to overrule a solar-only room.
2. Otherwise, call **grid** when the room has grid equipment, at least two sensors reported in that two-minute bucket, and the frequency is within **0.11 Hz** of the site median.
3. Otherwise, call **solar** when the room has solar, the frequency is in **49.8–50.2 Hz**, and the mean absolute deviation in the window is **≤ 0.1**.
4. Otherwise call **generator**.

Then two hospital-specific sentences, applied only on multi-source rows:

- **Princess Christian** (`princess_christian`): do not call generator while the site still looks like it is on the grid.
- **Ola During** (`ola_during`): while the grid looks off, a frequency in the solar band is solar.

The note never defines "grid off" as a number. This pipeline treats the site median as off-grid when at least two sensors reported and the median sits more than **0.5 Hz** from 50 Hz. That 0.5 is an assumption, written down here so it is not an invisible constant.

Two versions of the rules are scored:

- `rules_centered` uses a window of three points before and three after, which is what the note describes. It peeks about six minutes into the future, so it is **reported and not deployed**.
- `rules_trailing` uses only the past (seven readings, and it refuses to cross a gap longer than 10 minutes). This is the live baseline.

The site median is the median frequency of every sensor at that site in the same 120-second bucket. The extract has **no grid-only room** to use as a reference meter, which is what the note would have preferred.

On the holdout the centred rules reach macro F1 **0.8636** and generator recall **0.975**, with generator precision only about **0.57**. They fire on generator often, and they are often wrong when they do. That is the gap the model is allowed to close.

## Features

Feature engineering here is a procedure, not a bag of columns. Name the decision (the source at one stamp, for one sensor). Write the questions an engineer would ask, in hertz and volts. Turn each question into a formula that uses only that stamp and the recent past, and that the API can recompute from a phone's JSON. Check the formula on a flat 50 Hz line, where the trailing spread must be zero. Only then run it on the extract. The teaching notebook walks those steps on ten hand-made rows.

Twenty-four columns, in `FEATURE_COLUMNS`. Anything the API serves is causal: at time *t* it may use readings at *t* and earlier, plus the site median of the current bucket (other sensors in the same two minutes, not the future).

| Group | Columns | Role |
| --- | --- | --- |
| Raw | `voltage`, `frequency` | The sensor |
| Level | `abs_freq_dev_50`, `abs_volt_dev_230` | Distance from 50 Hz and 230 V |
| Change | `freq_diff_1`, `volt_diff_1` | One-step difference, but only inside a short gap |
| Trailing window of 7 | mean, std, MAD, min, max of frequency; std of voltage | The "how noisy is this stretch" evidence the solar rule wants |
| Site | `freq_dev_site_median`, `n_sensors_in_bucket` | The grid rule's reference, built without a future sample |
| Clock | `hour_sin`, `hour_cos`, `dow_sin`, `dow_cos` | Time of day and day of week, encoded so Monday sits next to Sunday |
| Survey | `has_grid`, `has_solar`, `has_generator`, `n_sources` | What is installed, not what is on |
| Quality | `gap_minutes`, `low_voltage` | A long silence, or voltage under 50 V |

"MAD" here is the mean absolute deviation from the window mean, not a median absolute deviation. The centred MAD used by `rules_centered` is stored beside the feature matrix and is **not** one of the 24 model inputs.

Class ids are alphabetical, and every model is checked for that order: `generator=0`, `grid=1`, `solar=2`.

### Holdout and the samples the trees actually see

- **Holdout:** `time >= 2024-02-01` UTC. 947,085 known-label rows. No `fit` call sees them. Full-holdout metrics are computed on all of those rows.
- **Trees** are fit on a stratified reservoir of the pre-holdout rows: 40,000 per class for gradient boosting (120,000 rows), 8,000 per class for the random forest (24,000). Scoring still uses every holdout row.
- **Linear models** see every pre-holdout row, twice, through `partial_fit`.
- **Windows** for the sequence models are length 16 (about half an hour at a 2-minute cadence). 1,500 train and 500 test windows per class. Their F1 is **not** comparable to the full-holdout F1. The sample is balanced and small.

## Models

| Name | What is fit | Eligible to deploy |
| --- | --- | --- |
| `rules_trailing` | Causal rules above | yes |
| `rules_centered` | Same rules with a future window | no |
| `sgd` | StandardScaler on every pre-holdout row, values clipped to ±8, then `SGDClassifier` (`log_loss`, L2 `alpha=1e-4`, `average=True`, two epochs, class-balanced sample weights) | yes |
| `sgd_pca` | Same logistic head on 10 incremental principal components | yes |
| `hgb` | `HistGradientBoostingClassifier`, learning rate 0.08, 120 iterations, depth 8, `min_samples_leaf` 30, L2 0.1 | yes |
| `rf` | 40 trees, depth 12, `min_samples_leaf` 5 | yes |
| `hybrid_hgb` | Survey override for single-source rooms, boosting for the rest, then the PCMH and Ola sentences | yes |
| `hybrid_sgd` | The same hybrid around the logistic model | yes |
| `online` | The logistic model, still allowed to `partial_fit` after deployment | served, not in the bake-off |

The winner is the eligible model with the best **macro F1 on multi-source holdout rows**, then overall macro F1. Single-source rooms are mostly the survey, so a model can look strong there without having learned frequency. Multi-source F1 is the number that matches the case.

**Deployed default: `hybrid_hgb`.** Holdout macro F1 0.9385, multi-source macro F1 0.9364, generator recall 0.9856, accuracy 0.947.

### Full holdout (947,085 rows)

| Model | Macro F1 | Multi-source macro F1 | Generator recall | Accuracy |
| --- | --- | --- | --- | --- |
| hybrid_hgb | 0.9385 | 0.9364 | 0.9856 | 0.9470 |
| hgb | 0.9355 | 0.9348 | 0.9856 | 0.9468 |
| rf | 0.9314 | 0.9339 | 0.9878 | 0.9455 |
| rules_centered | 0.8636 | 0.8611 | 0.9750 | 0.9354 |
| rules_trailing | 0.8636 | 0.8611 | 0.9737 | 0.9354 |
| hybrid_sgd | 0.8599 | 0.8571 | 0.6018 | 0.9332 |
| sgd | 0.8553 | 0.8545 | 0.6018 | 0.9327 |
| sgd_pca | 0.7117 | 0.7096 | 0.4566 | 0.9109 |

Boosting is the model that actually recovers generator. The logistic model is close on macro F1 and weak on generator recall (0.60). It is kept because it can update online, not because it won.

### PCA

PCA is the dimensionality reduction step. Each row of 24 scaled features is rewritten as fewer scores. The first score is the row's position along the direction where the rows spread out the most. The next score is at right angles to that, so it describes something the first score left out. Scaling comes first: a frequency near 50 and a 0/1 flag are put on the same footing, or the large numbers would choose the axes alone.

A loading says how hard an original column pulls on a score. The scree plot says what share of the spread each score keeps. Ten components, fit with `partial_fit` on every pre-holdout row, explain **86.87%** of the variance of the scaled features. The scree plot is `artifacts/figures/pca_scree.png`. The other 13% is the part `sgd_pca` agrees to drop. Frequency, distance from 50 Hz, the trailing mean, and the site-median deviation are largely one signal, which is why that logistic-on-PCA model loses generator. The trees skip PCA and cut on Hertz directly. Thresholds in Hertz are also easier to audit than a component score. The notebook shows the same idea first on two columns that are copies of each other: one score keeps essentially all of their spread.

### Temporal cross-validation

Three expanding calendar cuts, scored on the pre-holdout teaching sample only. The February holdout is not one of these folds.

| Train before | Test until | Boosting macro F1 | Logistic macro F1 |
| --- | --- | --- | --- |
| 2023-08-01 | 2023-10-01 | 0.9649 | 0.9491 |
| 2023-10-01 | 2023-12-01 | 0.9537 | 0.9283 |
| 2023-12-01 | 2024-02-01 | 0.9482 | 0.9161 |

Both models get a little worse as the cut moves forward. Neither collapses. Random k-fold is not used: it would train on Tuesday afternoon and test on Tuesday morning.

### Permutation importance

Shuffling a column on 2,000 holdout rows (4 repeats) and watching macro F1. Top drops for the boosting model: `has_grid` 0.382, `abs_freq_dev_50` 0.166, then `n_sensors_in_bucket` 0.012. This is a global "does the column matter" test. It is not SHAP. A column can matter in opposite directions on different windows, and a shuffle cannot show that.

## SHAP

SHAP is computed for the **boosting model**, which is the part of the hybrid that reads the sensors. The rules are deterministic; explaining them with SHAP would only restate the thresholds.

`shap.TreeExplainer` cannot emit class probabilities for a multiclass `HistGradientBoostingClassifier` (it raises `NotImplementedError` unless `model_output="raw"`). Raw margins are the wrong unit for a classroom, because a mentee cannot read them as "this feature raised the probability of generator by 0.2". The analysis is **permutation SHAP** on `predict_proba`. A positive value raised that class's probability. A negative value lowered it.

- 120 holdout rows per class (360 rows), drawn from `work/features/tabular_test.npz`.
- Background: 34 training rows per class (102 rows), from `tabular_train.npz`. The masker is told to use every one of those rows. shap's default cap is 100; staying near that cap avoids a silent subsample.
- One call of `python -m pipeline.shap_analysis` takes about a minute on a laptop CPU.

Figures, all under `artifacts/figures/`:

| File | What to look at |
| --- | --- |
| `shap_mean_abs.png` | Average absolute effect on each class probability |
| `shap_beeswarm_generator.png` | Every explained window, for the generator probability. Colour is the feature value |
| `shap_dependence.png` | `has_grid` and frequency deviation against the generator effect; site deviation against the solar effect |
| `shap_waterfall.png` | One real window of each class, top features, with the feature's own value on the axis |

`artifacts/shap_summary.json` is the full ranking and one local example per class. `artifacts/shap_values.npz` is the matrix `(360, 24, 3)` if you want to redraw a plot. `metrics.json` carries a short `shap` object, and `GET /model-info` returns it so the web console can draw the generator ranking without another compute step.

Reading the checked-in run, in words:

- **Generator probability** moves first with distance from 50 Hz (`abs_freq_dev_50`, mean |SHAP| 0.204) and with whether grid equipment is installed (`has_grid`, 0.181). On the beeswarm, a large deviation (red) pushes generator probability up, and `has_grid = 1` (red) pushes it down. That is the rule story, measured on the model rather than assumed.
- **Grid probability** is dominated by `has_grid` (mean |SHAP| 0.398). Frequency deviation is a distant second (0.028). The survey is doing most of the grid decision, which is why multi-source F1 is the selection metric.
- **Solar probability** depends on being close to 50 Hz (mean |SHAP| 0.220) and on whether grid equipment is installed (0.217).
- The trailing min, max, and spread of frequency are real but smaller. Voltage barely moves any class. `low_voltage` has mean |SHAP| 0 in this sample: the unknown / dead-sensor rows were already excluded from training.
- One generator window in the waterfall has `abs_freq_dev_50 = 2.62` and that feature adds 0.21 to the generator probability, from a base of 0.34. One grid window has `has_grid = 1` adding 0.44, from a base of 0.32, while three installed sources pull the other way. One solar window is 0.09 Hz from 50 and that closeness adds 0.23. Direction depends on the window. That is the thing permutation importance cannot say.

The teaching notebook has a section that displays these four figures. If the json is missing it tells you the command to run.

## Sequence models and the forecast

These are a second view of the same problem, on purpose. They are not in the deployment bake-off, because their test set is 1,500 balanced windows rather than 947,085 rows.

| Model | Macro F1 on the 1,500 windows | Generator recall |
| --- | --- | --- |
| Gradient boosting on the same window's feature row | 0.970 | 0.994 |
| sktime summary classifier | 0.863 | |
| sktime time-series forest (frequency channel only) | 0.856 | |
| LSTM, hidden size 32, 8 epochs | 0.824 | 0.690 |
| RNN, same budget | 0.750 | 0.844 |
| sktime ROCKET, 1,000 kernels | 0.493 | 0.476 |

`TimeSeriesForestClassifier` refused a multivariate array, so it sees frequency only. Voltage is channel 0 and frequency is channel 1 in the window tensor `(n, T, 2)`. ROCKET on a length-16 series with 1,000 kernels does not beat the hand-built features. That is a useful negative result: the trailing mean and the site median already extracted what a short window can say.

sktime's deep classifiers (`LSTMFCNClassifier`, and the torch variants) need TensorFlow or PyTorch. Those are the right next step on a GPU machine and the wrong dependency for the 8 GB laptops and for the Docker image. `pipeline/sequence_models.py` is a many-to-one RNN and LSTM in NumPy (Adam, gradient clip 5, forget-gate bias 1). Both pass a finite-difference gradient check. The API serves those weights.

### Forecasting, which is a different question

Classification says "what is on now". The sktime forecast says "what will the hourly mean frequency be". It is fit for one long sensor, respondent `285A769C` (Ola During, HDU, grid with solar and generator backup): 8,168 hourly means from 2023-04-26 16:00 to 2024-03-31 23:00.

Last 14 days (336 hours), from `pipeline/forecasting.py` and sktime 1.2:

| Forecaster | MAE (Hz) |
| --- | --- |
| Seasonal naive, period 24 hours | 0.2235 |
| Ridge reduction, window 48 hours | 0.1870 |

Skill against the naive forecast is 0.163. An expanding window (8 folds, `sktime.split.ExpandingWindowSplitter`) gives naive MAE 0.1426 and ridge 0.1149. The reduction model is better than "same hour yesterday". It is not a power-source classifier, and the notebook keeps those scores in a separate section so they are not compared with macro F1.

sktime 1.3 will change the default of `remember_data`. The teaching notebook filters that warning. It is not a failed fit.

## Usage hours

On the holdout, each gap is given to the earlier reading and capped at 30 minutes, so a sensor that goes silent is not counted as a day of runtime. Generator hours: **383.1** from the labels, **436.0** from `hybrid_hgb`. The model over-counts generator relative to the rules. Given the rules' generator precision of about 0.57, some of that gap is the model catching events the rules miss, and some of it is the model being wrong. The hours are not a fuel invoice until someone checks them against a log.

By site, label hours versus model hours: bonthe 20 / 22, kabala 161 / 192, kambia 90 / 96, masanga 111 / 119, ola_during 0 / 4, princess_christian 0 / 2. The last two are the hospitals whose smoothing rules suppress generator. The per-room table is `artifacts/usage_durations.csv` (31 rooms have a holdout interval; not every one of the 35 sensors does).

## joblib, in one sitting

A fitted model is a Python object in memory. Close the laptop and it is gone. [joblib](https://joblib.readthedocs.io/) writes that object to a file and reads it back:

```python
import joblib
joblib.dump(bundle, "model.joblib")
bundle = joblib.load("model.joblib")
```

JSON can store the metrics and the figures' numbers. It cannot store a fitted `HistGradientBoostingClassifier` and hand you back a classifier. `pickle` can, and joblib is built on pickle. The extra joblib gives you is large NumPy arrays, written as one block instead of one number at a time. A boosting model is mostly arrays, which is why scikit-learn's own model files use joblib.

`artifacts/model.joblib` is a dictionary, not a single estimator. The keys a mentee should be able to name are `hgb`, `rf`, `sgd`, `sgd_pca`, `scaler`, `pca`, `classes`, `feature_names`, and `sklearn_version`. The API calls `joblib.load` once at startup and serves that dictionary. `online_model_init.joblib` is the logistic model as training left it. `online_model.joblib` is the copy `POST /online/update` is allowed to dump again. The RNN and LSTM weights stay in `.npz` files, because those models are plain NumPy arrays.

Load this file with the same scikit-learn that wrote it (1.9.0). A `.joblib` file is a pickle, so load files this training run produced.

The teaching notebook has a short joblib 101, with a toy `dump` / `load` and then a look inside the real file, just before the deployment section.

## The API

`service/api.py`. The image and the local process both listen on port **8010**, so this service does not collide with the house-price app in the parent `app/` folder. Leave that app alone.

| Method | Path | Role |
| --- | --- | --- |
| GET | `/` | Web console |
| GET | `/health` | Version, row count, holdout macro F1 |
| GET | `/model-info` | Comparison, permutation importance, SHAP summary, forecast |
| GET | `/examples` | Three real holdout windows |
| GET | `/usage` | Per-room holdout hours |
| POST | `/predict` | One window |
| POST | `/predict/batch` | Up to 50 windows |
| POST | `/online/update` | One labelled window into the logistic model |
| POST | `/online/reset` | Restore the logistic model from `online_model_init.joblib` |

`model` on a predict call is `best` (the hybrid), `rules`, `hgb`, `online`, `lstm`, or `rnn`. `adjusted_by_rules` is true when the hybrid label differs from the boosting model's own argmax, which is how you see a PCMH or Ola sentence fire.

A request is the survey, the site, and a short history. Times are the clock times from the file:

```json
{
  "site_id": "bonthe",
  "room": "Maternity ward - Solar & Generator",
  "available_power_sources": "Solar & Generator",
  "site_median_frequency": 49.4,
  "n_sensors_in_bucket": 3,
  "model": "best",
  "history": [
    {"time": "2024-03-21T14:06:00", "voltage": 220.0, "frequency": 47.4},
    {"time": "2024-03-21T14:08:00", "voltage": 221.0, "frequency": 47.5}
  ]
}
```

The service recomputes the 24 features from that history. It does not trust the client to send a feature vector. If the history is shorter than 16 steps, the LSTM and RNN pad by repeating the first reading and set `warning`.

`service/mobile/PowerSourceClient.java` is a `java.net.http` client for `POST /predict`. An Android emulator reaches the laptop at `http://10.0.2.2:8010`. The same JSON is what a Kotlin, React Native, or Flutter app would post. The page at `/` is only a client.

The console's SHAP section reads `mean_abs_shap.generator` from `/model-info`. Restart uvicorn after a new SHAP run: `load_metrics` is cached for the life of the process.

## Docker

From `case_study_3`, once Docker Desktop is running:

```bash
docker build -f service/Dockerfile -t kamilimu-power-source .
docker run --rm -p 8010:8010 kamilimu-power-source
```

The image is `python:3.12-slim`, runs as `appuser`, copies `pipeline/`, `service/`, and `artifacts/`, and runs the service tests during the build. `shap` is **not** in `service/requirements.txt`. The explanations are computed at training time and shipped as JSON and PNG. That keeps numba and the SHAP wheel out of the runtime image.

On the machine that built the artifacts, the Docker CLI was installed (27.3.1) and the engine was not running (`dockerDesktopLinuxEngine` returned an internal error on ping). The image was not built here. The local pytest run is the check that did happen. Start Docker Desktop and rerun the two commands above before you treat the container as verified.

## Online learning

`POST /online/update` with a `label` of `generator`, `grid`, or `solar` calls `partial_fit` on the logistic model and writes `artifacts/online_model.joblib`. `POST /online/reset` copies the init file back over it.

Two things to expect in class:

- The deployed decision is still `hybrid_hgb` until you ask for `model: "online"`. Feedback does not silently change the boosting model. Boosting here is a batch fit.
- `average=True` on an SGD that has already seen millions of rows means one classroom label moves the coefficients by something like `1e-7`. The update is real. `np.allclose` with its default tolerance will tell you nothing changed. Compare the max absolute difference, or reset and send a batch, if you are demonstrating it.

The service tests copy artifacts to `work/test_artifacts` and point `ARTIFACT_DIR` there before they import the app, so a test update does not touch the real `online_model.joblib`.

## Caches and a clean rebuild

| Path | Safe to delete? |
| --- | --- |
| `work/raw/` | Yes. The next train re-reads the csv. |
| `work/features/` | Yes, same consequence. SHAP needs these npz files, so rebuild features before re-running SHAP. |
| `artifacts/model.joblib` | Only if you are about to train again. The API will not start without it. |
| `artifacts/figures/shap_*.png` | Regenerated by `python -m pipeline.shap_analysis`. |

`work/` is gitignored. Artifacts are the thing you would hand to a deployment. They contain no row-level sensor export except the small example windows and the SHAP sample.

## Tests

From `case_study_3`:

```bash
python -m pytest -q tests service/tests
```

`tests/test_core.py` checks the survey parser, a flat 50 Hz trailing MAD of zero, the single-source solar rule, the 0.11 Hz grid rule, the RNN and LSTM gradients, and the SHAP ranking helper. `service/tests` skips if `model.joblib` is missing. joblib on NumPy 2.5 prints a `DeprecationWarning` about setting an array shape. The tests still pass; that warning is not a failure.

## What this pipeline will not claim

- The labels are the rule system. A higher F1 means closer agreement with those rules, plus the hybrid's own smoothing. It is not an accuracy against a current clamp or a fuel log.
- Generator hours are capped intervals on one holdout season. They are a demonstration, not a bill.
- The 0.5 Hz "grid off" cutoff is ours. Change it in `GRID_OFF_DEV_HZ` and the PCMH and Ola sentences move.
- There is no grid-only reference room in this extract. The site median is the stand-in.
- ROCKET and the small LSTM are teaching models on 1,500 windows. They are not the production score.
- One online label will not visibly retune a model fit on four million rows. Reset between demos.

## Environment that produced the artifacts

Python 3.13, pandas 2.2.3, NumPy 2.5, scikit-learn 1.9.0, sktime 1.2.0, numba 0.68, shap 0.53.0, FastAPI, pyarrow. Torch and TensorFlow are intentionally not installed. The service pins the same scikit-learn, because `model.joblib` is a pickle.
