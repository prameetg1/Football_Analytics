#!/usr/bin/env python3
"""Generate notebooks/train_pitch_colab.ipynb — Colab recipe for the pitch
keypoint fine-tune (reference: References/train_pitch_keypoint_detector.py).

Produces a valid nbformat JSON the user runs in Google Colab, then drops the
resulting best.pt back into runs/pose/runs/pose/field_v12/weights/.
"""

import json
import sys
from pathlib import Path


def md(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": text}


def code(text: str) -> dict:
    return {"cell_type": "code", "execution_count": None,
            "metadata": {"colab": {"base_uri": "https://localhost:8080/"},
                         "id": "cell"},
            "outputs": [], "source": text}


TITLE = md(
    "# Train pitch keypoint detector (Colab)\n\n"
    "Fine-tunes `yolov8x-pose` on the `football-field-detection-f07vi` dataset "
    "(v12, 32 keypoints) so homography works **fully locally** in "
    "PerformanceAnalyzer.\n\n"
    "**Setup (once):**\n"
    "1. Runtime -> Change runtime type -> GPU (T4 is fine; A100 faster).\n"
    "2. In the left **Secrets** (key) panel add `ROBOFLOW_API_KEY` "
    "(app.roboflow.com -> Settings -> API Keys).\n"
    "3. (Optional) Mount Drive so `best.pt` survives the session."
)

INSTALL = code(
    "!pip -q install ultralytics roboflow\n"
    "import ultralytics\n"
    "print('ultralytics', ultralytics.__version__)"
)

API_KEY = code(
    "try:\n"
    "    from google.colab import userdata\n"
    "    API_KEY = userdata.get('ROBOFLOW_API_KEY')\n"
    "except Exception:\n"
    "    API_KEY = None\n"
    "if not API_KEY:\n"
    "    import getpass\n"
    "    API_KEY = getpass.getpass('ROBOFLOW_API_KEY: ')\n"
    "print('key loaded:', bool(API_KEY))"
)

DATASET = code(
    "import os\n"
    "from roboflow import Roboflow\n"
    "rf = Roboflow(api_key=API_KEY)\n"
    "proj = rf.workspace('roboflow-jvuqo').project('football-field-detection-f07vi')\n"
    "ds = proj.version(12).download('yolov8', location='/content/field_detection')\n"
    "DATA = '/content/field_detection/data.yaml'\n"
    "print(open(DATA).read())"
)

TRAIN = code(
    "# Tune these:\n"
    "MODEL   = 'yolov8x-pose.pt'   # x = best accuracy; 'm' for ~3x faster\n"
    "IMGSZ   = 960                 # more pixels on line markers = better kpts\n"
    "EPOCHS  = 150\n"
    "BATCH   = 8 if (IMGSZ >= 960 and MODEL.startswith('yolov8x')) else 16\n"
    "print(f'training {MODEL} imgsz={IMGSZ} epochs={EPOCHS} batch={BATCH}')\n"
    "!yolo task=pose mode=train model={MODEL} data={DATA} epochs={EPOCHS} \\\n"
    "    imgsz={IMGSZ} batch={BATCH} mosaic=0.0 plots=True project=/content/runs\n"
    "import glob\n"
    "weights = glob.glob('/content/runs/**/weights/best.pt', recursive=True)\n"
    "print('BEST_PT', weights[0] if weights else 'MISSING')"
)

METRICS = code(
    "import glob, subprocess\n"
    "w = glob.glob('/content/runs/**/weights/best.pt', recursive=True)[0]\n"
    "!yolo task=pose mode=val model={w} data={DATA} split=valid"
)

SAVE = code(
    "import glob, shutil, os\n"
    "src = glob.glob('/content/runs/**/weights/best.pt', recursive=True)[0]\n"
    "out = '/content/best_pitch.pt'\n"
    "shutil.copy(src, out)\n"
    "# 1) save to Drive (if mounted)\n"
    "try:\n"
    "    from google.colab import drive\n"
    "    drive.mount('/content/drive')\n"
    "    shutil.copy(out, '/content/drive/MyDrive/best_pitch.pt')\n"
    "    print('saved to MyDrive/best_pitch.pt')\n"
    "except Exception as e:\n"
    "    print('drive skip:', e)\n"
    "# 2) or download to your machine\n"
    "from google.colab import files\n"
    "files.download(out)\n"
    "print('done')"
)

FINISH = md(
    "## Bring it home\n\n"
    "Copy `best_pitch.pt` into the project:\n\n"
    "```bash\n"
    "cp best_pitch.pt runs/pose/runs/pose/field_v12/weights/best.pt\n"
    "PYTHONPATH=. .venv/bin/python scripts/validate_pitch.py \\\n"
    "    --video data/raw/0bfacc_0.mp4 --max-frames 40\n"
    "```\n\n"
    "If RMSE improves, it is automatically used by `src/config.py` "
    "(`PITCH_KEYPOINT_MODEL_WEIGHTS`)."
)

cells = [TITLE, INSTALL, API_KEY, DATASET, TRAIN, METRICS, SAVE, FINISH]

notebook = {
    "nbformat": 4,
    "nbformat_minor": 0,
    "metadata": {
        "colab": {"provenance": [], "gpuType": "T4"},
        "kernelspec": {"display_name": "Python 3", "name": "python3"},
        "language_info": {"name": "python"},
        "accelerator": "GPU",
    },
    "cells": cells,
}

out = Path(__file__).resolve().parents[1] / "notebooks" / "train_pitch_colab.ipynb"
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(notebook, indent=1))
print(f"wrote {out}")
