# CaptAin (2026 Findings of CVPR)

## Environment Setup
Strict version replication is not required. If version conflicts occur, install the versions supported by your local machine.

```bash
# This repo uses TWO conda environments:
# - captain_llm: for "Preparation (Caption Generation)"
# - captain_prvr: for "Training"

# Preparation env (caption generation)
conda activate captain_llm
cd train
pip install -r requirements.captain_llm.txt

# Training env (MS-SL)
conda activate captain_prvr
cd train
pip install -r requirements.captain_prvr.txt
```

## Data Dependencies
Only the **Charades-STA** dataset needs to be downloaded for reproduction (smaller in size than ActivityNet and TVR). Complete dependency data download:  

[Charades-STA Data Download Link](https://huggingface.co/datasets/e-gf/CaptAin_depends/tree/main/charades)

[Activitynet Data Download Link](https://huggingface.co/datasets/e-gf/CaptAin_depends/tree/main/activitynet)

[Youcook2-PRVR Data Download Link](https://huggingface.co)

## Caption Generation LLM Preparation
For caption generation, you need to prepare the LLM checkpoints:

[Qwen2.5-32B-Instruct](https://huggingface.co/Qwen/Qwen2.5-32B-Instruct)

[Qwen2.5-VL-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct)

If you run into `transformers` version issues (e.g., `KeyError: 'qwen2'` / `KeyError: 'qwen2_5_vl'`), install the latest `transformers` from source:

```bash
pip install git+https://github.com/huggingface/transformers accelerate
```

## Preparation (Caption Generation)
Before training, you need to generate captions for the dataset. (using env `captain_llm`)

### Step 1: Generate local captions (example of charades-STA)
Run:

```bash
cd root_path # Project root directory
cd train/caption_generation
python charades_local.py
```

### Step 2: Redundancy-aware merge for local captions
Local captions require an additional merging step to produce the final local captions:

```bash
cd train/caption_generation
python merge_local.py
```

### Step 3: Generate global captions (example of charades-STA)
Run:

```bash
cd train/caption_generation
python charades_global.py
```

### Step 4: Convert generated JSON to feature files
Convert the generated JSON files into feature files (HDF5):

```bash
cd train/caption_generation
python json2hdf5.py
```

After finishing the above steps, the data preparation is complete.



## Training
Each dataset has a separate training script. Execute the following commands to start training (using env `captain_prvr`):

### Charades-STA
```bash
cd train/mssl
bash do_charades.sh
```

### ActivityNet
```bash
cd train/mssl
bash do_activitynet.sh
```

### Youcook2-PRVR
```bash
cd train/mssl
bash do_youcook.sh
```

### Training Parameter Explanation
`local_hinge_weight`: Weight of the local-level hinge loss. You can tune this to balance local alignment strength.

`global_soft_pos_weight`: Weight of the global-level soft positive loss. You can tune this to balance global alignment strength.

`hca_loss_type`: Type of loss used for hard caption alignment (e.g., `margin`, `infonce`).

`caption_rate`: Ratio of caption-based samples used during training. Usually kept as is.

`query_or_caption`: Controls whether to use query, caption, or both. Usually kept as is.

`local_margin / soft_pos_margin`: Margin hyperparameters for the loss. In most cases you **do not need to change** these.
