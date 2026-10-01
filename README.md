# Helmet Violation Detection: Streamlit proof of concept

Runs the thesis's final three-stage cascade (YOLO11s → YOLOv8s-pose → YOLO11s-cls) on uploaded CCTV frames or short clips.
The inference code in `pipeline.py` is copied from `Thesis_Pipeline_v2_Final_v1_9.ipynb` (cells 3.1, 3.4a, 5.1, RQ2 B), so it gives the same decisions as the reported system.

## Files you must add (from Google Drive)
Run cell **DEPLOY.1** (`export_cell.py`) in the final notebook after cells 0.1–0.3. It writes
`MyDrive/helmet_thesis_v2/deploy/helmet_poc_files.zip`. Unzip it into this folder so you get:

```
helmet_poc/
├── app.py
├── pipeline.py
├── requirements.txt
├── models/
│   ├── stage1_yolo11s-rf8_1280_s0.pt     (runs/stage1_yolo11s-rf8_1280_s0/weights/best.pt)
│   ├── stage2_yolov8s_rf3_s0.pt          (runs/stage2_yolov8s_rf3_s0/weights/best.pt)
│   └── stage3_yolo11s_rf6r15_s2.pt       (weights/stage3_yolo11s_rf6r15_s2.pt: the epoch-43 checkpoint, NOT best.pt)
└── config/
    ├── e2e_final_config_s123.json        (anchor, gate T, Stage 3 threshold 0.50, head offset and scale)
    ├── stage1_selected_rf8.json          (Stage 1 confidence 0.41, input 1280)
    ├── stage2_selected_rf3.json          (Stage 2 confidence 0.3, NMS IoU 0.5)
    └── stage3_selected_rf6.json          (Stage 3 run name)
```

## Run on your computer
```
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```
It opens at http://localhost:8501. No GPU needed: about 1–3 s per frame at 1280 px on a laptop CPU.

## Put it online (Streamlit Community Cloud, free)
1. Create a GitHub repository and push this whole folder, including `models/` and `config/`
   (each weight file is under 100 MB, so Git LFS is not needed).
2. Go to share.streamlit.io → Create app → choose the repository, branch and `app.py` → Deploy.
3. Under Advanced settings, choose Python 3.11 or 3.12.
Keep the repository private if the weights or footage should not be public.

## What the app shows
- **Image tab:** motorcycle boxes (blue), rider boxes (grey), head boxes (green OK / red NC) labelled with the
  non-compliant probability; `(*)` marks heads under 32 px. Per-rider table, time per stage, PNG and CSV downloads.
- **Video tab:** processes every Nth frame (no tracking between frames); annotated MP4 and per-frame CSV.
- **Sidebar:** uses the thesis settings by default; an "exploration" toggle lets you move the Stage 1 and Stage 3 thresholds.
- **About tab:** how a frame is processed, the thesis results and the limitations.
