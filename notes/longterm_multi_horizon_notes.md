# Long-Term Multi-Horizon Forecasting Notes

## Step 1: Leakage-Safe Data Builder

Script:

- `scripts/longterm_multi_horizon.py`

Command run:

```bash
python3 scripts/longterm_multi_horizon.py
```

What the code does:

- Loads the raw PEMS08 data.
- Builds a long-term speed forecasting dataset.
- Uses the current traffic state and older speed history as input.
- Predicts future speed at four horizons: +1h, +3h, +6h, +12h.
- Splits the data chronologically into train, validation, and test periods.
- Runs assertions to check that no measured input feature uses future information.

Input features currently include:

- current flow
- current occupancy
- current speed
- speed 3h ago
- speed 6h ago
- speed 12h ago
- speed 1d ago
- speed 2d ago
- speed 7d ago
- current time-of-day and day-of-week encodings
- target time-of-day and day-of-week encodings for each forecast horizon

Important leakage rule:

- At time `t`, measured features may only come from time `t` or earlier.
- Future calendar information is allowed because the target time is known.
- Future traffic measurements are not allowed.

Observed results:

- Raw PEMS08 shape: `T=17856`, `N=170`, `C=3`
- Raw date range: `2016-07-01 00:00:00` to `2016-08-31 23:55:00`
- Built input tensor: `X = (15696, 170, 29)`
- Built target tensor: `y = (15696, 170, 4)`
- Train split: `9417` graph snapshots
- Validation split: `3139` graph snapshots
- Test split: `3140` graph snapshots
- Leakage checks passed.

Observation:

- PEMS08 has enough usable data for the planned long-term multi-horizon task.
- The data builder gives us a clean foundation before adding baselines or neural models.

## Step 2: Leakage-Safe Baselines

Script:

- `scripts/longterm_multi_horizon.py`

Command run:

```bash
python3 scripts/longterm_multi_horizon.py --mode baselines
```

What changed in the code:

- Added a baseline mode to the same script.
- The script still builds the exact same leakage-safe dataset first.
- Then it evaluates simple forecasting methods on the test split.
- It saves a JSON report to `reports/pems08_longterm_multi_horizon_baselines.json`.

Baselines implemented:

- `mean_train_global`: predicts the average train speed for each horizon.
- `current_speed`: predicts that future speed will equal speed at current time `t`.
- `speed_lag_*`: predicts future speed using one historical speed lag.
- `ridge_alpha_1`: ridge regression trained on all current-time features and known time encodings.

Important detail:

- These baselines use only measured features from time `t` or earlier.
- Target-time calendar features are allowed because the future timestamp is known.
- No future traffic measurements are used.

Results on the PEMS08 test split:

| Model | Avg MAE | +1h MAE | +3h MAE | +6h MAE | +12h MAE |
|---|---:|---:|---:|---:|---:|
| mean_train_global | 3.9930 | 3.9841 | 3.9821 | 4.0071 | 3.9988 |
| current_speed | 3.9366 | 2.1171 | 3.4536 | 4.5177 | 5.6579 |
| speed_lag_36_3h | 4.7420 | 3.8996 | 4.5038 | 5.2530 | 5.3116 |
| speed_lag_72_6h | 5.0594 | 4.7280 | 5.2055 | 5.6245 | 4.6798 |
| speed_lag_144_12h | 4.5073 | 5.5002 | 5.2106 | 4.6607 | 2.6577 |
| speed_lag_288_1d | 4.4192 | 3.1543 | 3.9761 | 4.8124 | 5.7340 |
| speed_lag_576_2d | 4.6388 | 3.6279 | 4.2218 | 4.9406 | 5.7647 |
| speed_lag_2016_7d | 4.1295 | 2.6089 | 3.6450 | 4.5965 | 5.6675 |
| ridge_alpha_1 | 2.8699 | 2.1364 | 3.1328 | 3.4462 | 2.7640 |

Observations:

- Current speed is very strong for +1h, but gets worse as the horizon grows.
- The 12h historical lag is naturally strong for +12h because it lines up with the target horizon.
- The ridge regression baseline is the best overall baseline.
- Ridge gets +6h MAE `3.4462` under this current split and feature setup.
- This gives the future neural model a meaningful baseline to beat.

## Step 3: Shared Multi-Horizon GRU

Script:

- `scripts/longterm_multi_horizon.py`

Command run:

```bash
LD_LIBRARY_PATH=/usr/local/lib/python3.13/site-packages/nvidia/cudnn/lib:/usr/local/lib/python3.13/site-packages/nvidia/cublas/lib:/usr/local/lib/python3.13/site-packages/nvidia/cuda_runtime/lib:/usr/local/lib/python3.13/site-packages/nvidia/cuda_nvrtc/lib:/usr/local/lib/python3.13/site-packages/nvidia/cuda_cupti/lib:/usr/local/lib/python3.13/site-packages/nvidia/cufft/lib:/usr/local/lib/python3.13/site-packages/nvidia/curand/lib:/usr/local/lib/python3.13/site-packages/nvidia/cusolver/lib:/usr/local/lib/python3.13/site-packages/nvidia/cusparse/lib:/usr/local/lib/python3.13/site-packages/nvidia/nccl/lib:/usr/local/lib/python3.13/site-packages/nvidia/nvjitlink/lib:/usr/local/lib/python3.13/site-packages/nvidia/nvtx/lib python3 scripts/longterm_multi_horizon.py --mode train_nn --epochs 6 --patience 3 --batch-size 256 --hidden-dim 32 --num-layers 1 --dropout 0.0
```

What changed in the code:

- Added `train_nn` mode.
- Added a small shared GRU neural network.
- The model sees a short time window of the already-built features.
- The same GRU weights are shared across all sensors.
- The model predicts all four horizons at once: +1h, +3h, +6h, +12h.
- Added train-only normalization for inputs and targets.
- Added validation MAE tracking and early-stopping support.
- Added `--device auto|cuda|cpu`, so training can use GPU when available or CPU when needed.

Model used in this run:

- architecture: shared per-sensor GRU
- input window: 12 steps = 1 hour
- hidden size: 32
- GRU layers: 1
- dropout: 0.0
- parameters: 6,180
- device used in this run: CPU
- epochs: 6

Training progress:

| Epoch | Train Loss | Validation MAE | Validation RMSE |
|---:|---:|---:|---:|
| 1 | 0.2242 | 2.9460 | 5.8673 |
| 2 | 0.1881 | 2.7747 | 5.6399 |
| 3 | 0.1767 | 2.6640 | 5.5130 |
| 4 | 0.1704 | 2.5965 | 5.4419 |
| 5 | 0.1666 | 2.5495 | 5.3963 |
| 6 | 0.1637 | 2.5183 | 5.3505 |

Test results:

| Model | Avg MAE | +1h MAE | +3h MAE | +6h MAE | +12h MAE |
|---|---:|---:|---:|---:|---:|
| shared_sensor_gru | 2.4884 | 2.0219 | 2.6415 | 2.8784 | 2.4118 |

Saved outputs:

- `reports/pems08_longterm_multi_horizon_gru.json`
- `checkpoints/pems08_longterm_multi_horizon_gru.pt`

Observations:

- The GRU improves over the ridge baseline on every horizon in this run.
- The largest improvement is at +6h, where MAE improves from ridge `3.4462` to GRU `2.8784`.
- This is a strong sign that the sequence model is learning useful temporal structure beyond the handcrafted lag regression baseline.

### GPU Run With More Epochs

Command run:

```bash
python3 scripts/longterm_multi_horizon.py --mode train_nn --device auto --epochs 12 --patience 4 --batch-size 256 --hidden-dim 32 --num-layers 1 --dropout 0.0
```

Environment:

- CUDA available: yes
- GPU: NVIDIA GeForce GTX 1050 Ti with Max-Q Design
- Device used by script: `cuda`

Training progress:

| Epoch | Train Loss | Validation MAE | Validation RMSE |
|---:|---:|---:|---:|
| 1 | 0.2242 | 2.9460 | 5.8673 |
| 2 | 0.1881 | 2.7747 | 5.6399 |
| 3 | 0.1767 | 2.6640 | 5.5130 |
| 4 | 0.1704 | 2.5965 | 5.4419 |
| 5 | 0.1666 | 2.5495 | 5.3963 |
| 6 | 0.1637 | 2.5183 | 5.3505 |
| 7 | 0.1614 | 2.4871 | 5.3221 |
| 8 | 0.1594 | 2.4599 | 5.2949 |
| 9 | 0.1577 | 2.4412 | 5.2685 |
| 10 | 0.1562 | 2.4271 | 5.2446 |
| 11 | 0.1549 | 2.4057 | 5.2291 |
| 12 | 0.1538 | 2.3994 | 5.2119 |

Test results:

| Model | Avg MAE | +1h MAE | +3h MAE | +6h MAE | +12h MAE |
|---|---:|---:|---:|---:|---:|
| shared_sensor_gru | 2.3821 | 1.9295 | 2.4916 | 2.7548 | 2.3527 |

Observations:

- GPU training worked successfully.
- The longer 12-epoch run improved over the earlier 6-epoch CPU run.
- The +6h MAE improved from `2.8784` to `2.7548`.
- Compared with the best ridge baseline, +6h MAE improved from `3.4462` to `2.7548`.

### Larger GRU Run

Command run:

```bash
python3 scripts/longterm_multi_horizon.py --mode train_nn --device auto --epochs 20 --patience 5 --batch-size 128 --hidden-dim 64 --num-layers 1 --dropout 0.0
```

Model used in this run:

- architecture: shared per-sensor GRU
- input window: 12 steps = 1 hour
- hidden size: 64
- GRU layers: 1
- dropout: 0.0
- parameters: 18,500
- device used by script: CUDA/GPU
- epochs: 20

Training progress:

| Epoch | Train Loss | Validation MAE | Validation RMSE |
|---:|---:|---:|---:|
| 1 | 0.1992 | 2.7020 | 5.5236 |
| 2 | 0.1680 | 2.5324 | 5.3433 |
| 3 | 0.1597 | 2.4489 | 5.2467 |
| 4 | 0.1549 | 2.3962 | 5.2047 |
| 5 | 0.1519 | 2.3666 | 5.1740 |
| 6 | 0.1499 | 2.3511 | 5.1564 |
| 7 | 0.1482 | 2.3287 | 5.1387 |
| 8 | 0.1467 | 2.3131 | 5.1234 |
| 9 | 0.1455 | 2.3080 | 5.0989 |
| 10 | 0.1442 | 2.2933 | 5.1049 |
| 11 | 0.1433 | 2.2763 | 5.0936 |
| 12 | 0.1422 | 2.2675 | 5.0787 |
| 13 | 0.1414 | 2.2511 | 5.0883 |
| 14 | 0.1407 | 2.2490 | 5.0687 |
| 15 | 0.1399 | 2.2492 | 5.0648 |
| 16 | 0.1393 | 2.2434 | 5.0580 |
| 17 | 0.1387 | 2.2375 | 5.0692 |
| 18 | 0.1384 | 2.2699 | 5.0489 |
| 19 | 0.1378 | 2.2441 | 5.0523 |
| 20 | 0.1373 | 2.2166 | 5.0580 |

Test results:

| Model | Avg MAE | +1h MAE | +3h MAE | +6h MAE | +12h MAE |
|---|---:|---:|---:|---:|---:|
| shared_sensor_gru_h64 | 2.2103 | 1.8161 | 2.2622 | 2.5748 | 2.1880 |

Observations:

- Increasing the hidden size from 32 to 64 improved all horizons.
- The +6h MAE improved from `2.7548` to `2.5748`.
- Validation MAE was still improving near the end, so a slightly longer run may improve further.
- This is the best neural result so far.

### Longer Hidden-64 GRU Run

Command run:

```bash
python3 scripts/longterm_multi_horizon.py --mode train_nn --device auto --run-name gru_h64_e40 --epochs 40 --patience 8 --batch-size 128 --hidden-dim 64 --num-layers 1 --dropout 0.0
```

Model used in this run:

- architecture: shared per-sensor GRU
- input window: 12 steps = 1 hour
- hidden size: 64
- GRU layers: 1
- dropout: 0.0
- parameters: 18,500
- device used by script: CUDA/GPU
- epochs: 40

Training progress:

| Epoch | Train Loss | Validation MAE | Validation RMSE |
|---:|---:|---:|---:|
| 1 | 0.1992 | 2.7020 | 5.5236 |
| 2 | 0.1680 | 2.5324 | 5.3433 |
| 3 | 0.1597 | 2.4489 | 5.2467 |
| 4 | 0.1549 | 2.3962 | 5.2047 |
| 5 | 0.1519 | 2.3666 | 5.1740 |
| 6 | 0.1499 | 2.3511 | 5.1564 |
| 7 | 0.1482 | 2.3287 | 5.1387 |
| 8 | 0.1467 | 2.3131 | 5.1234 |
| 9 | 0.1455 | 2.3080 | 5.0989 |
| 10 | 0.1442 | 2.2933 | 5.1049 |
| 11 | 0.1433 | 2.2763 | 5.0936 |
| 12 | 0.1422 | 2.2675 | 5.0787 |
| 13 | 0.1414 | 2.2511 | 5.0883 |
| 14 | 0.1407 | 2.2490 | 5.0687 |
| 15 | 0.1399 | 2.2492 | 5.0648 |
| 16 | 0.1393 | 2.2434 | 5.0580 |
| 17 | 0.1387 | 2.2375 | 5.0692 |
| 18 | 0.1384 | 2.2699 | 5.0489 |
| 19 | 0.1378 | 2.2441 | 5.0523 |
| 20 | 0.1373 | 2.2166 | 5.0580 |
| 21 | 0.1368 | 2.2304 | 5.0425 |
| 22 | 0.1365 | 2.2112 | 5.0480 |
| 23 | 0.1359 | 2.2224 | 5.0370 |
| 24 | 0.1356 | 2.2122 | 5.0383 |
| 25 | 0.1353 | 2.2096 | 5.0470 |
| 26 | 0.1349 | 2.2182 | 5.0365 |
| 27 | 0.1346 | 2.2174 | 5.0232 |
| 28 | 0.1343 | 2.2014 | 5.0285 |
| 29 | 0.1340 | 2.2250 | 5.0175 |
| 30 | 0.1337 | 2.1980 | 5.0226 |
| 31 | 0.1332 | 2.2000 | 5.0188 |
| 32 | 0.1330 | 2.1890 | 5.0137 |
| 33 | 0.1326 | 2.1967 | 5.0280 |
| 34 | 0.1325 | 2.2069 | 5.0174 |
| 35 | 0.1321 | 2.1907 | 5.0137 |
| 36 | 0.1315 | 2.1864 | 5.0015 |
| 37 | 0.1314 | 2.1817 | 5.0125 |
| 38 | 0.1313 | 2.1777 | 5.0072 |
| 39 | 0.1311 | 2.1935 | 4.9968 |
| 40 | 0.1310 | 2.1878 | 4.9936 |

Test results:

| Model | Avg MAE | +1h MAE | +3h MAE | +6h MAE | +12h MAE |
|---|---:|---:|---:|---:|---:|
| shared_sensor_gru_h64_e40 | 2.1719 | 1.7829 | 2.2302 | 2.5249 | 2.1497 |

Saved outputs:

- `reports/pems08_longterm_multi_horizon_gru_h64_e40.json`
- `checkpoints/pems08_longterm_multi_horizon_gru_h64_e40.pt`

Observations:

- Training longer improved the hidden-64 GRU again.
- The +6h MAE improved from `2.5748` to `2.5249`.
- Validation MAE flattened after roughly epoch 30, so future improvements may come more from architecture/window changes than simply more epochs.
- This is the best result so far.

### Longer Window Attempt: GPU Out Of Memory

Command attempted:

```bash
python3 scripts/longterm_multi_horizon.py --mode train_nn --device auto --run-name gru_w24_h64_e40 --window 24 --epochs 40 --patience 8 --batch-size 128 --hidden-dim 64 --num-layers 1 --dropout 0.0
```

Result:

- The run failed with a CUDA out-of-memory error.
- GPU memory is limited to about 4GB on the NVIDIA GeForce GTX 1050 Ti.
- A 2-hour window (`--window 24`) with batch size 128 is too large for this GPU.

Observation:

- The longer window experiment is still useful, but it needs a smaller batch size such as 32 or 16.

### Longer Window With Smaller Batch

Command run:

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python3 scripts/longterm_multi_horizon.py --mode train_nn --device auto --run-name gru_w24_h64_b32_e40 --window 24 --epochs 40 --patience 8 --batch-size 32 --hidden-dim 64 --num-layers 1 --dropout 0.0
```

Model used in this run:

- architecture: shared per-sensor GRU
- input window: 24 steps = 2 hours
- hidden size: 64
- GRU layers: 1
- dropout: 0.0
- parameters: 18,500
- device used by script: CUDA/GPU
- batch size: 32
- early stopped at epoch 39

Training progress:

| Epoch | Train Loss | Validation MAE | Validation RMSE |
|---:|---:|---:|---:|
| 1 | 0.1737 | 2.4507 | 5.2523 |
| 2 | 0.1525 | 2.3485 | 5.1444 |
| 3 | 0.1471 | 2.3209 | 5.1088 |
| 4 | 0.1435 | 2.2571 | 5.0721 |
| 5 | 0.1411 | 2.2620 | 5.0440 |
| 6 | 0.1391 | 2.2599 | 5.0373 |
| 7 | 0.1374 | 2.2338 | 5.0248 |
| 8 | 0.1358 | 2.2104 | 5.0233 |
| 9 | 0.1346 | 2.2021 | 5.0223 |
| 10 | 0.1335 | 2.2112 | 4.9869 |
| 11 | 0.1326 | 2.1934 | 4.9807 |
| 12 | 0.1315 | 2.1817 | 4.9731 |
| 13 | 0.1309 | 2.1748 | 4.9785 |
| 14 | 0.1296 | 2.1662 | 4.9631 |
| 15 | 0.1288 | 2.1884 | 4.9561 |
| 16 | 0.1280 | 2.1764 | 4.9620 |
| 17 | 0.1274 | 2.1726 | 4.9479 |
| 18 | 0.1256 | 2.1620 | 4.9232 |
| 19 | 0.1252 | 2.1629 | 4.9371 |
| 20 | 0.1249 | 2.1589 | 4.9423 |
| 21 | 0.1244 | 2.1545 | 4.9354 |
| 22 | 0.1241 | 2.1489 | 4.9330 |
| 23 | 0.1237 | 2.1539 | 4.9367 |
| 24 | 0.1234 | 2.1549 | 4.9323 |
| 25 | 0.1231 | 2.1476 | 4.9378 |
| 26 | 0.1228 | 2.1640 | 4.9296 |
| 27 | 0.1225 | 2.1834 | 4.9292 |
| 28 | 0.1221 | 2.1589 | 4.9184 |
| 29 | 0.1212 | 2.1451 | 4.9207 |
| 30 | 0.1210 | 2.1407 | 4.9232 |
| 31 | 0.1209 | 2.1360 | 4.9306 |
| 32 | 0.1208 | 2.1415 | 4.9268 |
| 33 | 0.1206 | 2.1404 | 4.9120 |
| 34 | 0.1204 | 2.1397 | 4.9192 |
| 35 | 0.1200 | 2.1385 | 4.9186 |
| 36 | 0.1199 | 2.1393 | 4.9252 |
| 37 | 0.1198 | 2.1366 | 4.9214 |
| 38 | 0.1196 | 2.1377 | 4.9196 |
| 39 | 0.1195 | 2.1360 | 4.9183 |

Test results:

| Model | Avg MAE | +1h MAE | +3h MAE | +6h MAE | +12h MAE |
|---|---:|---:|---:|---:|---:|
| shared_sensor_gru_w24_h64 | 2.1447 | 1.7765 | 2.2108 | 2.4658 | 2.1259 |

Saved outputs:

- `reports/pems08_longterm_multi_horizon_gru_w24_h64_b32_e40.json`
- `checkpoints/pems08_longterm_multi_horizon_gru_w24_h64_b32_e40.pt`

Observations:

- The 2-hour input window improved over the 1-hour window.
- Average MAE improved from `2.1719` to `2.1447`.
- +6h MAE improved from `2.5249` to `2.4658`.
- This is the best result so far.

### Two-Layer GRU With Dropout

Command run:

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python3 scripts/longterm_multi_horizon.py --mode train_nn --device auto --run-name gru_w24_h64_l2_d02_b32_e40 --window 24 --epochs 40 --patience 8 --batch-size 32 --hidden-dim 64 --num-layers 2 --dropout 0.2
```

Model used in this run:

- architecture: shared per-sensor GRU
- input window: 24 steps = 2 hours
- hidden size: 64
- GRU layers: 2
- dropout: 0.2
- parameters: 43,460
- device used by script: CUDA/GPU
- batch size: 32
- epochs completed: 40

Training summary:

- Validation MAE improved from `2.4707` at epoch 1 to `2.0834` at epoch 40.
- Validation RMSE improved from `5.2531` at epoch 1 to `4.8757` at epoch 40.
- Training did not early stop, so the final epoch was the selected checkpoint.

Test results:

| Model | Avg MAE | +1h MAE | +3h MAE | +6h MAE | +12h MAE |
|---|---:|---:|---:|---:|---:|
| shared_sensor_gru_w24_h64_l2_d02 | 2.1015 | 1.7566 | 2.1402 | 2.3974 | 2.1120 |

Saved outputs:

- `reports/pems08_longterm_multi_horizon_gru_w24_h64_l2_d02_b32_e40.json`
- `checkpoints/pems08_longterm_multi_horizon_gru_w24_h64_l2_d02_b32_e40.pt`

Observations:

- Adding a second GRU layer with dropout improved the previous best result.
- Average MAE improved from `2.1447` to `2.1015`.
- +6h MAE improved from `2.4658` to `2.3974`.
- This is the best neural result so far.

## Step 5: Result Summary Table And MAE Figure

Script:

- `scripts/summarize_longterm_results.py`

Command run:

```bash
python3 scripts/summarize_longterm_results.py
```

What the code does:

- Reads the baseline report from `reports/pems08_longterm_multi_horizon_baselines.json`.
- Reads all GRU neural reports matching `reports/pems08_longterm_multi_horizon_gru*.json`.
- Collects MAE/RMSE/MAPE/R2 values from each report.
- Prints a compact comparison table sorted by average MAE.
- Saves a combined JSON summary.
- Saves a report-ready MAE-vs-horizon figure.

Generated outputs:

- `reports/pems08_longterm_multi_horizon_summary.json`
- `reports/figures/pems08_longterm_multi_horizon_mae.png`

Best baseline:

| Model | Avg MAE | +1h MAE | +3h MAE | +6h MAE | +12h MAE |
|---|---:|---:|---:|---:|---:|
| ridge_alpha_1 | 2.8699 | 2.1364 | 3.1328 | 3.4462 | 2.7640 |

Best neural model:

| Model | Avg MAE | +1h MAE | +3h MAE | +6h MAE | +12h MAE |
|---|---:|---:|---:|---:|---:|
| GRU w24 h64 L2 d0.2 | 2.1015 | 1.7566 | 2.1402 | 2.3974 | 2.1120 |

Important comparison:

- Average MAE improved over the best baseline by `0.7683` mph.
- Relative average MAE improvement over the best baseline: about `26.8%`.
- +6h MAE improved over the best baseline by `1.0489` mph.