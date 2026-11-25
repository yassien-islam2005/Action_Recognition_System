# Group B – Two-Stream Action Recognition

A PyTorch implementation of two-stream video action recognition on a processed
100-class subset of UCF101.

## Dataset status and scope

The project uses the processed UCF101 archive collection (`UCF101_processed`).
The audited project split contains 100 classes; **Archery is excluded** because
it is not represented sufficiently in the processed collection used for this
experiment. The original official test split is retained.

| Split | Videos |
|---|---:|
| Train | 8,492 |
| Validation | 941 |
| Test | 3,742 |
| **Total** | **13,175** |

The split files and audit report are committed under
`data/splits/ucf101/split_1/`. See `audit_report.txt` and `split_info.json`
for the recorded split procedure and checks.

**The dataset archives and model checkpoints are intentionally not stored in
this Git repository.** Keep the 51 processed ZIP archives in a separate
location and pass their parent directory to the scripts.

## Model currently implemented

The active model is `src/models/two_stream_model.py`:

1. **Spatial stream:** ResNet-18 processes RGB frames (3 channels).
2. **Temporal stream:** a separate ResNet-18 processes two-channel optical
   flow (Flow X and Flow Y).
3. **Temporal modelling:** each stream has its own bidirectional GRU followed
   by learnable temporal attention.
4. **Fusion:** a learnable gate predicts two stream weights and combines the
   stream representations.
5. **Classifier:** a fully connected head predicts one of the 100 classes.

The expected model inputs are RGB `(B, 32, 3, 224, 224)` and optical flow
`(B, 32, 2, 224, 224)`. The output is `(B, 100)` logits.

The loader reads the processed JPEG frames directly from the ZIP archives.
RGB frames use ImageNet mean/std normalization. Flow images are decoded using
`FLOW_BOUND = 20` and scaled approximately to `[-1, 1]`, as implemented in
`src/datasets/ucf101_dataset.py`.

> Note: `src/models/rgb_resnet.py` and `src/models/flow_resnet.py` are retained
> legacy/reference implementations from the repository work. The active
> `TwoStreamActionRecognition` model imports the backbones under
> `src/spatial/` and `src/temporal/`.

## Repository layout

```text
.
├── configs/
├── data/
│   └── splits/
│       ├── ucf101/split_1/
│       └── ucf101_official/
├── docs/
├── scripts/
│   └── create_splits.py
├── src/
│   ├── datasets/
│   ├── fusion/
│   ├── models/
│   ├── spatial/
│   ├── temporal/
│   └── training/
├── tests/
├── .gitignore
├── README.md
└── requirements.txt
```

## Environment setup

Python 3.10 or newer is recommended. Install a PyTorch and TorchVision pair
compatible with your machine/runtime first, following the official PyTorch
installation selector. Then install the remaining project dependencies:

```bash
pip install -r requirements.txt
```

## Expected data directory

The `--data-root` argument must point to the directory containing the
processed class ZIP archives, for example:

```text
UCF101_processed/
├── ApplyEyeMakeup.zip
├── ApplyLipstick.zip
├── ...
└── YoYo.zip
```

Each archive is expected to contain a `manifest.csv` and the processed
per-video folders with `rgb/`, `flow_x/`, and `flow_y/` images.

## Run the dataloader checks

From the repository root:

```bash
python -m src.datasets.test_dataloader --data-root "/path/to/UCF101_processed"
```

The script checks train, validation, and test dataset counts, batch tensor
shapes, finite values, and label consistency.

## Run a one-batch training-pipeline smoke test

This runs one forward pass, computes cross-entropy loss, checks gradients,
and performs one optimizer update. It is a smoke test, **not** the full
training program:

```bash
python -m src.training.test_training_pipeline   --data-root "/path/to/UCF101_processed"   --device cpu
```

Use `--device cuda` when CUDA is available. Add `--pretrained` to request
ImageNet initialization; this may download weights on first use.

## Rebuild the group-aware split

The split-generation script uses the official split-1 lists and
`StratifiedGroupKFold` to create a group-aware validation subset from the
official training videos. Run from the repository root:

```bash
python scripts/create_splits.py
```

This rewrites the generated `train.txt`, `val.txt`, `test.txt`,
`split_info.json`, and `audit_report.txt` in the split directory. Review the
audit report after regenerating.

## Important limitations

- This repository snapshot contains the model components, dataset loader,
  split files, and small validation scripts.
- It does **not** contain a complete epoch-based training entry point,
  experiment configuration system, trained weights, or the dataset archives.
- `test_training_pipeline.py` only validates one real batch and one optimizer
  step. It must not be described as the full training implementation.
- The current README and requirements document the code present in this
  snapshot; they do not claim that a full experiment has been trained.
