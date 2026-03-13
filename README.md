## CaptAin-PRVR

**Paper**: CaptAin (2026 Findings of CVPR)  
**Code layout**: `train/` (data preparation & training), `test/` (evaluation), `depends/` (extra scripts/deps).


---

## 1. Environment Setup

See `train/README.md` for detailed environment setup instructions, including:

- Conda environments for caption generation and training;
- Required Python packages and LLM-related dependencies (e.g., Qwen2.5 series).

---

## 2. Data, Caption Generation & Training

See `train/README.md` for:

- Dataset preparation and download links;
- Caption generation pipeline (local/global captions and feature conversion);
- Training scripts and hyper-parameter descriptions for all supported datasets.

---

## 3. Test & Evaluation

See `test/readme.md` for:

- How to choose and configure checkpoints;
- How to set `ckpt_path` in `test/method/train.py`;
- How to run the `do_*.sh` scripts for evaluation on each dataset.

