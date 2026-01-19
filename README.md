# WrongWayDriving_DA-RTFM_I3D
````md
# WrongWayDriving — DA-RTFM (I3D)

This repository contains the code and scripts used for **wrong-way driving detection / safety alerting** using **DA-RTFM** with **I3D** features (RGB + Flow).

---

## Repository

Clone the project:

```bash
git clone https://github.com/abdulsamad1999/WrongWayDriving_DA-RTFM_I3D
cd WrongWayDriving_DA-RTFM_I3D
````

---

## Prerequisites

Install Python dependencies:

```bash
pip install -r requirements.txt
```

> Notes:
>
> * The training/testing commands below assume the required dataset lists exist under `list/`.
> * Visualization options use a Visdom-like setup via `--viz-server` and `--viz-port`.

---

## Download Required Assets

### 1) Download Features

Download features from:

* [https://pern-my.sharepoint.com/:f:/g/personal/abdul_1617851_talmeez_pk/IgB2B-WyR7GOR6KiBcGZNrg7ATWpS0GypHvgxDQqCeYVjog?e=d6999A](https://pern-my.sharepoint.com/:f:/g/personal/abdul_1617851_talmeez_pk/IgB2B-WyR7GOR6KiBcGZNrg7ATWpS0GypHvgxDQqCeYVjog?e=d6999A)

Place the downloaded **features folder adjacent to** the cloned `DA-RTFM` folder (i.e., next to the repository root directory).

Example layout:

```
<parent-directory>/
├─ WrongWayDriving_DA-RTFM_I3D/     # cloned repo
└─ <features-folder>/              # downloaded features (adjacent)
```

---

### 2) Download Saved Weights and Results

Download saved model weights and results from:

* [https://pern-my.sharepoint.com/:f:/g/personal/abdul_1617851_talmeez_pk/IgA7NJRZONVCRqMrqrToUWPZAagsVlrRQ0vikEPscdDKHJU?e=DbfodL](https://pern-my.sharepoint.com/:f:/g/personal/abdul_1617851_talmeez_pk/IgA7NJRZONVCRqMrqrToUWPZAagsVlrRQ0vikEPscdDKHJU?e=DbfodL)

From the downloaded content, place the following directories **inside** the cloned repository root:

* `run_ablation`
* `runs_max_safety`

Expected layout:

```
WrongWayDriving_DA-RTFM_I3D/
├─ run_ablation/
├─ runs_max_safety/
├─ list/
├─ main.py
├─ run_ablation.py
└─ ...
```

---

## Usage

---

## Flow Feature Extraction

Extract flow features using (only if you are using a custom dataset):

```bash
python extract_flow_features.py \
  --list list/mytest.list \
  --video-root ..\dataset \
  --suffix _flow \
  --num-segments 32 \
  --theta-ref-file theta_ref.json
```

### Training (Max Safety Run)

```bash
python main.py \
  --dataset wrongway-dataset \
  --rgb-list list/mytrain.list \
  --test-rgb-list list/mytest.list \
  --feature-size 1024 \
  --num-segments 32 \
  --batch-size 16 \
  --max-epoch 100 \
  --lr "[0.0001]*15000" \
  --split-by-label \
  --use-flow --flow-suffix _flow --flow-dim 4 \
  --lambda-temp 1e-4 --temp-on-all \
  --lambda-dir 1e-2 --dir-margin 0.0 \
  --seed 123 \
  --output-dir runs_max_safety \
  --viz-env wrongway-max-safety \
  --viz-server http://localhost --viz-port 8097
```

---

### Ablation

```bash
python run_ablation.py \
  --dataset wrongway-dataset \
  --rgb-list list/mytrain.list \
  --test-rgb-list list/mytest.list \
  --split-by-label \
  --seed 123 \
  --viz-env wrongway-ablation \
  --viz-server http://localhost --viz-port 8097
```

---

### Testing (Safety Alerting)

```bash
python test_safety_alerting.py \
  --dataset wrongway-dataset \
  --rgb-list list/mytrain.list \
  --test-rgb-list list/mytest.list \
  --feature-size 1024 \
  --num-segments 32 \
  --batch-size 16 \
  --split-by-label \
  --use-flow --flow-suffix _flow --flow-dim 4 \
  --precision-floor 0.85 \
  --alert-rule auto \
  --m-grid 3,4,5,6,8 \
  --k-grid-mode strict \
  --model-path runs_max_safety/ckpt/rtfm_nosparse_best.pkl
```

> If you are on Linux/macOS, you may want to replace `..\dataset` with a POSIX path (e.g., `../dataset`).

---

## Troubleshooting

* **Missing runs/weights:** Ensure `runs_max_safety/` and `run_ablation/` are placed inside the repo root.
* **Missing features:** Ensure the downloaded features directory is adjacent to the cloned repo directory.
* **Visualization not working:** Confirm the service at `http://localhost:8097` is running and reachable.

---


