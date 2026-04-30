# PRISM Dataset

**PRISM** (*Pressure and Inertial Sensing for Human Motion and Interaction*) is a multimodal motion-capture dataset combining **SMPL body**, **8-IMU body sensors**, **insole pressure**, **SLAM head trajectory + environment point cloud**, and **scene objects**. It covers everyday locomotion, balance, sports, and stepping/sitting interactions in light and dark conditions.

PRISM was collected and released as part of **GRIP: Ground Reaction Inertial Poser** (CVPR 2026), which fuses 4 wrist/foot IMUs with insole pressure (vertical GRF, CoP, contact) to reconstruct physically plausible full-body motion via a two-stage *kinematics → physics-controlled dynamics* architecture.

<p align="center">
  <a href="https://ryosukehori.github.io/grip-project/">
    <img alt="Project Page" src="https://img.shields.io/badge/Project_Page-GRIP-1f7a8c?style=for-the-badge&logo=github&logoColor=white">
  </a>
  <a href="https://arxiv.org/abs/2603.16233">
    <img alt="arXiv" src="https://img.shields.io/badge/arXiv-2603.16233-b31b1b?style=for-the-badge&logo=arxiv&logoColor=white">
  </a>
  <a href="https://ryosukehori.github.io/grip-project/">
    <img alt="CVPR 2026" src="https://img.shields.io/badge/CVPR-2026-1f4e79?style=for-the-badge">
  </a>
</p>

<p align="center">
  <img src="image/PRISM.png" alt="PRISM teaser" width="720">
</p>

---

## At a glance

| | |
|---|---|
| Sampling rate | 100 fps |
| Sequence length | 10 s (1000 frames) |
| Total sequences | **1,275** (train 1,021 / test 254) |
| Total duration | ≈ 3.5 hours |
| Subjects | 6 (`subj001` – `subj006`) |
| Takes | 149 train / 127 test |
| | |

**Trials** include `Walking`, `Jogging`, `Switch Direction`, `Sidestep`, `Lunge Walk`, `Forward Jump`, `Squatting`, `Stretching`, `Balancing`, `Object Carrying`, `Stepping Boxes/Stair`, `Sitting Boxes`, `Soccer`, `Tennis`, `Baseball`, `Golf`. The `Dark` suffix denotes low-light recordings; `Stepping*` / `Sitting*` trials include accompanying object meshes.

---

## Layout

```
PRISM/
├── data/PRISM/<subj>/<take>.pkl   # one continuous take per file
├── json/
│   ├── dataset_split.json         # sequence-level train / test split
│   └── dataset_statistics.json    # per-sequence chunk windows + flags
├── image/PRISM.png                # teaser
├── data_viewer.py                 # aitviewer-based visualization
├── requirements.txt               # Python dependencies (NumPy 2.1.x)
└── README.md
```

**Files are stored as full-length takes**, not pre-chunked — `data/PRISM/<subj>/<take>.pkl` contains the entire recording. Train / test partitioning happens at the chunk level: the JSON files under `json/` describe how each take is sliced into 1000-frame (10 s) chunks (`seq000` → frames 0–1000, `seq001` → 1000–2000, …) and assign each chunk to either split. Load the take, then slice with `start_frame:end_frame` from `json/dataset_statistics.json` (see [Quick start](#quick-start)).

---

## File format

Each `.pkl` is a Python `dict`. `F` is the take's frame count (e.g. 13,160).

### `info` — metadata

```
subj_info : {subj_id, gender, age, height, weight, shoe_length, arm_length}
data_info : {take_id, trial_name, insole_size, fps,
             synth_imu_frames: {L_Wrist: [...], R_Wrist: [...]}}
```

`synth_imu_frames` lists frame indices where the wrist IMU is **synthesized from SMPL** rather than measured by the Apple Watch. During recording the Watch occasionally dropped or returned corrupted samples; those frames are reconstructed from SMPL kinematics and their indices are preserved here so you can mask, weight, or exclude them as needed. Test-split chunks are guaranteed not to contain any synthesized wrist-IMU frames (see [Train / test split](#train--test-split)).

### `smpl_params` — SMPL body

| field | shape | notes |
|---|---|---|
| `poses` | `[F, 72]` float32 | axis-angle; first 3 dims = global orientation |
| `betas` | `[F, 10]` float32 | |
| `trans` | `[F, 3]` float64 | add `root_offset` for world coords |
| `gender` | `'male'` / `'female'` | |
| `root_offset` | `[3]` float64 | |

### `imu` — measured / synthesized IMU (8 parts)

Parts: `L_Foot`, `R_Foot`, `L_Wrist`, `R_Wrist`, `Head`, `Pelvis`, `L_Knee`, `R_Knee`. Field names encode `<quantity>_<frame>_<processing>`.

| field | shape | notes |
|---|---|---|
| `acc_world_raw`  | `[F, 3]` | world frame, unfiltered |
| `acc_world_filt` | `[F, 3]` | world frame, low-pass filtered |
| `ori_world`      | `[F, 3, 3]` | world-frame rotation matrix |
| `acc_local_raw`  | `[F, 3]` | feet only — insole IMU, sensor frame, raw |
| `gyr_local_raw`  | `[F, 3]` | feet only — insole IMU, sensor frame, raw |

### `imu_gt` — SMPL-derived ground truth (8 parts)

| field | shape |
|---|---|
| `pos_world`, `vel_world`, `acc_world` | `[F, 3]` |
| `ori_world` | `[F, 3, 3]` |

### `insole` — pressure / GRF

Per foot (`L_Foot`, `R_Foot`):

| field | shape | notes |
|---|---|---|
| `contacts` | `[F, 2]` bool | `[front, back]` |
| `force` | `[F, 1]` | total |
| `forces` | `[F, 16]` | 16-cell pressures (insole-local) |
| `CoP` | `[F, 2]` | insole-local |
| `force_world`, `CoP_world` | `[F, 3]` | world frame |

`combined.{force_world, CoP_world}` provide both-feet sums.

### `slam` — Project Aria SLAM

| field | shape | notes |
|---|---|---|
| `points` | `[N, 3]` | environment point cloud |
| `head_traj` | `[F, 3]` | head trajectory (world frame) |

### `objects` — optional scene meshes

Present only for `Stepping*` / `Sitting*` trials; `None` otherwise. Each entry: `{vertices [V,3], faces [Fc,3], centroid [3], extents [3]}`.

### Coordinate conventions

- World frame is **Z-up**.
- World translation = `smpl_params['trans'] + smpl_params['root_offset']`.
- IMU mounting positions (approximate SMPL vertex/joint indices) are defined as `IMU_VERTEX_IDX` / `IMU_JOINT_IDX` in `data_viewer.py`.

---

## Train / test split

`json/dataset_split.json`:

```json
{
  "train": ["subj001_take002_seq000", ...],
  "test":  ["subj001_take002_seq004", ...]
}
```

Chunk-level random split (`seed = 42`); the same take can contribute chunks to both splits. **Chunks containing any synthesized wrist-IMU frames are kept on the train side only**, so the test set is fully measured (see `has_synthetic_imu` below and `synth_imu_frames` under [`info`](#info--metadata)).

`json/dataset_statistics.json` contains per-sequence records:

```
{subject_id, take_id, sequence_id, chunk_idx,
 start_frame, end_frame, total_frames, has_synthetic_imu, file_name}
```

Use `start_frame:end_frame` to slice the chunk out of the corresponding `.pkl`.

---

## Quick start

```bash
conda create -n prism python=3.11 -y && conda activate prism
pip install --no-build-isolation -r requirements.txt
```

Tested on Python 3.11 with NumPy 1.26.x. The `.pkl` files were pickled with NumPy 1.x and load on NumPy 1.x or 2.x. `--no-build-isolation` is needed because `chumpy`'s `setup.py` imports `pip._internal` (otherwise hidden by PEP 517). See [`requirements.txt`](requirements.txt) for pinning rationale and SMPL model setup notes.

**Load a take:**

```python
import pickle

with open("data/PRISM/subj001/take002.pkl", "rb") as f:
    data = pickle.load(f)

poses    = data["smpl_params"]["poses"]                # [F, 72]
foot_acc = data["imu"]["L_Foot"]["acc_world_filt"]     # [F, 3]
contacts = data["insole"]["L_Foot"]["contacts"]        # [F, 2]
head     = data["slam"]["head_traj"]                   # [F, 3]
```

**Iterate the split:**

```python
import json, pickle

split = json.load(open("json/dataset_split.json"))
stats = json.load(open("json/dataset_statistics.json"))
meta  = {f"{r['subject_id']}_{r['take_id']}_{r['sequence_id']}": r
         for r in stats["train_sequences"] + stats["test_sequences"]}

for seq_id in split["train"]:
    r = meta[seq_id]
    take = pickle.load(open(f"data/PRISM/{r['subject_id']}/{r['file_name']}", "rb"))
    chunk = take["smpl_params"]["poses"][r["start_frame"]:r["end_frame"]]  # [1000, 72]
```

**Interactive visualization** — overlays SMPL, IMU, GRF, foot contact, head trajectory, point cloud, and object mesh:

```bash
python data_viewer.py                              # random order, infinite loop
python data_viewer.py -s subj001                   # all takes from subj001 (one pass)
python data_viewer.py -s subj001 -t take002        # one specific take
python data_viewer.py --data-dir other/PRISM       # use a different data root
```
