# RTSM Evaluation Workspace

Ground-truth collection and evaluation for RTSM. See
`.claude/plans/permanent-plan/RTSM Evaluation & Metrics System — Implementation Plan.md`
for the full plan.

## Layout

```
eval/
├── README.md                       # this file
├── marker_assignments.csv          # edit this: which marker is taped to which object
├── markers/                        # generated ArUco marker images (printable)
├── datasets/                       # one folder per recorded scene
│   └── <scene_id>/
│       ├── rgb/                    # frames (from Calabi Lens)
│       ├── depth_npy/
│       ├── FrameTrajectory.txt
│       ├── KeyFrameTrajectory.txt
│       └── ground_truth.yaml       # generated from marker_assignments.csv
└── scripts/
    ├── generate_aruco_sheet.py     # print this sheet, cut, tape on objects
    └── csv_to_ground_truth.py      # CSV → per-scene ground_truth.yaml
```

## Workflow

### 1. Print markers (once)

```bash
python eval/scripts/generate_aruco_sheet.py
```

Produces:
- `eval/markers/marker_00.png` … `marker_14.png` — individual markers
- `eval/markers/printable_sheet.pdf` — multi-page US Letter PDF, print at 100% scale (no fit-to-page)

Each marker is sized for **5 cm physical print**. Measure one after printing — if it's off, your printer scaled it.

### 2. Tape markers, fill CSV, record

For each scene you record:

1. Tape markers on/next to the objects you want as ground truth
2. Add rows to `marker_assignments.csv` — one row per (scene_id, marker_id) pair
3. Record a 30-second sweep with Calabi Lens → saves to `eval/datasets/<scene_id>/`

### 3. Generate ground-truth YAML

```bash
python eval/scripts/csv_to_ground_truth.py
```

Writes `eval/datasets/<scene_id>/ground_truth.yaml` for every scene in the CSV.

The 3D positions in the YAML are filled in later by the offline ArUco detector
(`rtsm/evaluation/scenes/aruco.py`, Phase 2 of the eval plan). You only type
the marker→label mapping; positions are measured automatically from the RGB
frames + ARKit pose.

## CSV format

```
scene_id,marker_id,object_id,label,notes
bedroom_01,0,mug_red,mug,red ceramic mug on desk
bedroom_01,1,keyboard,keyboard,mechanical keyboard
bedroom_01,2,person_me,person,me standing in doorway
```

| Column | Required | Notes |
|---|---|---|
| `scene_id` | yes | Folder name under `datasets/`. Reuse across rows for the same scene. |
| `marker_id` | yes | Integer 0–49 (DICT_4X4_50). Must match the printed marker. |
| `object_id` | yes | Unique within a scene. Used as the GT object's stable ID. |
| `label` | yes | Semantic label (mug, keyboard, person, …). Used for per-category recall. |
| `notes` | no | Free text for your own reference. Ignored by the script. |
