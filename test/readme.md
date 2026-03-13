# Test / Evaluation

This directory contains the evaluation code for the trained model.

The test pipeline loads the checkpoint generated during training and evaluates the model on three datasets:

- **Charades**
- **ActivityNet**
- **YouCook**

---

## Project Structure

```
finalproject
│
├── depends
├── train
├── test
│   └── method
│       ├── train.py
│       ├── do_charades.sh
│       ├── do_activitynet.sh
│       ├── do_youcook.sh
│       └── ...
└── .gitignore
```

---

## Prerequisites

Before running the evaluation, make sure you have already completed the **training stage**.

After training finishes, a model checkpoint (`ckpt`) will be generated at:

```
train/mssl/[dataset_name]/result/
```

Example:

```
train/mssl/charades/result/model.ckpt
```

---

## Setup Checkpoint Path

Open the following file:

```
test/method/train.py
```

Find the following line:

```python
ckpt_path = "path_to_your_model_ckpt"
```

Replace it with the path to your trained checkpoint.

Example:

```python
ckpt_path = "../../train/mssl/charades/result/model.ckpt"
```

---

## Run Evaluation

Navigate to the evaluation directory:

```bash
cd test/method/
```

Then run the evaluation scripts for each dataset.

### Evaluate on Charades

```bash
bash do_charades.sh
```

### Evaluate on ActivityNet

```bash
bash do_activitynet.sh
```

### Evaluate on YouCook

```bash
bash do_youcook.sh
```

Each script will automatically load the specified checkpoint and perform evaluation on the corresponding dataset.

---

## Notes

- Ensure the checkpoint path in `train.py` is correctly set before running evaluation.
- Make sure the dataset paths and dependencies are correctly configured.