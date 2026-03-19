<div align="center">

# SATTC: Structure-Aware Label-Free Test-Time Calibration for Cross-Subject EEG-to-Image Retrieval

**CVPR 2026**

[Qunjie Huang](mailto:huangqunjie@stu.ynu.edu.cn) · [Weina Zhu](mailto:zhuweina@ynu.edu.cn)<sup>✉</sup>

Yunnan University, China

<!-- Uncomment when available:
[![Paper](http://img.shields.io/badge/Paper-arxiv.XXXX.XXXXX-B31B1B.svg)](https://arxiv.org/abs/XXXX.XXXXX)
[![Project Page](https://img.shields.io/badge/Project-Page-blue)](https://your-project-page.github.io)
-->

</div>

---

## Abstract

Cross-subject EEG-to-image retrieval for visual decoding is challenged by **subject shift** and **hubness** in the embedding space, which distort similarity geometry and destabilize top-*k* rankings, making small-*k* shortlists unreliable. We introduce **SATTC** (Structure-Aware Test-Time Calibration), a label-free calibration head that operates directly on the similarity matrix of frozen EEG and image encoders. SATTC combines a **geometric expert**—subject-adaptive whitening of EEG embeddings with an adaptive variant of Cross-domain Similarity Local Scaling (CSLS)—and a **structural expert** built from mutual nearest neighbors, bidirectional top-*k* ranks, and class popularity, fused via a simple Product-of-Experts rule. On THINGS-EEG under a strict leave-one-subject-out protocol, standardized inference with cosine similarities, ℓ₂-normalized embeddings, and candidate whitening already yields a strong cross-subject baseline over the original ATM retrieval setup. Building on this baseline, SATTC further improves Top-1 and Top-5 accuracy, reduces hubness and per-class imbalance, and produces more reliable small-*k* shortlists. These gains transfer across multiple EEG encoders, supporting SATTC as an encoder-agnostic, label-free test-time calibration layer for cross-subject neural decoding.

<!-- Framework figure — uncomment when figure is added to repo:
<p align="center">
  <img src="assets/fig-framework.png" width="90%"/>
</p>
-->

## News

- **[2026/03]** Code released for SATTC.
<!-- - **[2026/XX]** Paper accepted to CVPR 2026. -->

## Installation

### Requirements

- Python ≥ 3.10
- PyTorch ≥ 2.0 with CUDA support
- GPU: ≥ 16 GB VRAM (tested on RTX 4080 Super)
- **CPU RAM: ≥ 36 GB** — each training subject uses ~4.2 GB; full 9-subject LOSO requires ~38 GB at peak during dataset loading. Reduce the subject list (`--subjects`) if RAM is limited.
- See [`requirements.txt`](requirements.txt) for full list

### Quick Setup

```bash
# Option A: using the setup script (creates a conda env named 'sattc')
bash setup.sh

# Option B: manual install
conda create -n sattc python=3.10 -y
conda activate sattc
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

## Data Preparation

We use the **THINGS-EEG** dataset. Download the preprocessed data from [Hugging Face](https://huggingface.co/datasets/LidongYang/EEG_Image_decode/tree/main) or [OSF](https://osf.io/3jk45/).

Organize the data as follows:

```
<your_data_root>/
├── preprocessed_data/
│   ├── sub-01/
│   │   ├── preprocessed_eeg_training.npy
│   │   └── preprocessed_eeg_test.npy
│   ├── sub-02/
│   └── ...
└── image_set/
    ├── training_images/
    └── test_images/
```

Update the data paths in [`Retrieval/data_config.json`](Retrieval/data_config.json).

### EEG Preprocessing (Optional)

If you want to preprocess from raw EEG data:

```bash
cd EEG-preprocessing/
python preprocessing.py
```

## Usage

### EEG-to-Image Retrieval (Leave-One-Subject-Out)

Run the full LOSO evaluation with all SATTC components (SAW, CSLS, Ada-CSLS, PoE are **enabled by default**):

```bash
cd Retrieval/
python run_sattc_loso.py \
    --logger True \
    --gpu cuda:0 \
    --batch_size 1024 \
    --output_dir ./outputs/loso_sattc
```

This reproduces the paper's LOSO protocol: each of `sub-01` – `sub-10` is held out as the test subject in turn, with the remaining 9 subjects used for training.  Results are aggregated across all folds.

Expected performance (THINGS-EEG, 200-way retrieval):

| Method | Top-1 | Top-5 |
|--------|------:|------:|
| Baseline (cosine, ℓ₂-norm) | ~8% | ~27% |
| **SATTC (SAW + Ada-CSLS + PoE)** | **~13%** | **~37.5%** |

To ablate individual components:

```bash
# Disable SAW (Subject-Adaptive Whitening)
python run_sattc_loso.py --no_saw ...

# Disable CSLS / Ada-CSLS
python run_sattc_loso.py --no_csls ...

# Disable Product-of-Experts fusion
python run_sattc_loso.py --disable_poe ...
```

### Baseline Comparison (e.g., EEGNetV4)

```bash
cd Retrieval/
python contrast_retrieval.py \
    --encoder_type EEGNetv4_Encoder \
    --epochs 30 \
    --batch_size 1024
```

## Project Structure

```
SATTC-CVPR2026/
├── Retrieval/                    # Core retrieval pipeline
│   ├── run_sattc_loso.py         #   Main LOSO training & evaluation entry
│   ├── contrast_retrieval.py     #   Baseline encoder comparison
│   ├── eegdatasets_leaveone.py   #   LOSO dataset loader
│   ├── split_generator.py        #   Cross-validation split generator
│   ├── loss.py                   #   Loss functions
│   ├── metrics.py                #   Evaluation metrics
│   ├── scoring.py                #   Scoring utilities
│   ├── eval_runner.py            #   Evaluation runner
│   ├── fold_aggregate.py         #   Fold-level result aggregation
│   ├── best_val_evaluator.py     #   Best-validation evaluator
│   ├── util.py                   #   General utilities
│   ├── data_config.json          #   Data path configuration
│   ├── subject_layers/           #   Transformer & attention modules
│   └── utils/                    #   Additional utility modules
├── EEG-preprocessing/            # Raw EEG preprocessing scripts
│   ├── preprocessing.py
│   └── preprocessing_utils.py
├── requirements.txt
├── setup.sh
├── LICENSE
└── README.md
```

## Citation

If you find this work useful, please cite:

```bibtex
@inproceedings{huang2026sattc,
  title     = {SATTC: Structure-Aware Label-Free Test-Time Calibration for Cross-Subject EEG-to-Image Retrieval},
  author    = {Huang, Qunjie and Zhu, Weina},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year      = {2026}
}
```

## Acknowledgments

We thank the authors of the following works for their open-source contributions:

- [ATM / Visual Decoding via EEG Embeddings](https://arxiv.org/abs/2403.07721) — Li et al., NeurIPS 2024
- [THINGS-EEG Dataset](https://www.sciencedirect.com/science/article/pii/S1053811922008758) — Gifford et al.
- [EEG Natural Image Decoding](https://arxiv.org/abs/2308.13234) — Song et al.

## License

This project is licensed under the [MIT License](LICENSE).
