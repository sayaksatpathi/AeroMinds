---
title: AeroMinds
emoji: 🌿
colorFrom: green
colorTo: blue
sdk: docker
app_port: 7860
---

## Overview

AeroMinds turns raw aerial drone footage into actionable environmental intelligence. Using a fine-tuned **YOLOv8** detector, it identifies illegal dumping sites, scores their severity, and routes them through a response pipeline — helping municipalities keep cities clean without manual monitoring.

## Key Features

- **Aerial image & video inference** — drag-and-drop support for high-resolution images and drone flyover videos.
- **Severity assessment engine** — computes waste coverage %, cluster counts, and assigns a `HIGH / MEDIUM / LOW` severity score.
- **Automated incident pipeline** — tracks each detection through a visual timeline: `DETECTED → PENDING → ASSIGNED → CLEARED`.
- **Live intelligence dashboard** — a dark-mode command center for monitoring sanitation across zones.
- **Hardware-accelerated** — PyTorch pipeline optimized for CUDA for fast video-frame processing.

## Tech Stack

| Layer | Tools |
|------|-------|
| Model | YOLOv8 (fine-tuned on aerial imagery), PyTorch, CUDA |
| App | Streamlit dashboard, Flask fallback server |
| Deploy | Docker, Hugging Face Spaces |
| Data | Custom aerial dumping dataset |

## Model

Fine-tuned YOLOv8 (`models/aerominds_dumping_v2.pt`) trained on aerial imagery for illegal-waste detection. Baseline confidence threshold `0.40`, with tuned Non-Maximum Suppression for dense clusters.

## Quick Start

```bash
git clone https://github.com/sayaksatpathi/AeroMinds.git
cd AeroMinds
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run streamlit_app.py                       # opens http://localhost:8501
```

### Docker

```bash
docker build -t aerominds .
docker run -p 7860:7860 aerominds
```

## Results & Honest Limitations

**Works well on:** high-contrast solid dumping masses, scattered roadside debris (grouped into clusters), and geometric/industrial waste.

**Known failure modes:**
- **False positives** on light-colored boulders, dry brush, and sun-glare (texture resembles debris at altitude).
- **False negatives** on micro-debris and sites under tree-canopy/shadow (Nano model tradeoff).
- **Prototype constraints:** synchronous video processing (blocks UI on large files — production needs an async queue like Celery); hardcoded location metadata (production needs GPS from drone EXIF/SRT); SQLite wiped on container restart without volumes.

## Team

Roushan Srivastava · Sayak Satpathi

---

*Built to keep our environment clean, one flight at a time.*
