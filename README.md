# The Linear Recoverability Hypothesis

**When Are Visual Concepts Linearly Recoverable?**


---

## Overview

The Linear Representation Hypothesis (LRH) claims that meaningful concepts correspond to linear directions in a model's latent space. We test whether this holds uniformly across visual concept complexity in CLIP, or whether it breaks down for compositional and relational concepts — and if so, whether the failure is driven by the concept itself or by the image distribution.

Our diagnostic is the **linearity gap**: the AUC difference between matched linear and nonlinear (MLP) probes trained on frozen features. We evaluate 28 concepts spanning four complexity tiers across synthetic (CLEVR) and natural (Visual Genome) image distributions, and extend the analysis to MAE and DINO to test whether linearity is specific to contrastive pretraining.

## Key Findings

- **Image realism, not concept complexity, drives nonlinearity.** The same compositional task produces a 4–6× larger linearity gap on natural Visual Genome images than on synthetic CLEVR.
- **Color–shape binding is approximately additive.** Predicted and actual CLIP embeddings for all 24 CLEVR (color, shape) pairs achieve mean cosine similarity 0.999, with low-rank, color-dominated residual structure.
- **Training objective shapes geometry.** MAE stores substantial semantic information in a less linear geometry (ΔAUC up to 0.17), while DINO matches CLIP's linear organization. CLIP and DINO linearity gaps are near zero or negative; every MAE gap is positive.
- **Linear probes can underestimate representation quality.** MAE's linear probe underperforms by up to 17 AUC points on concepts that nonlinear probes recover well.

## Models

| Model | Backbone | Pretraining | Source |
|-------|----------|-------------|--------|
| CLIP | ViT-B/32 | Contrastive (LAION-2B) | `open_clip` |
| MAE | ViT-B/16 | Masked autoencoding | `timm` |
| DINO | ViT-B/16 | Self-distillation | `torch.hub` |

All backbones are frozen throughout.

## Concept Battery (28 concepts)

| Tier | Dataset | Examples |
|------|---------|----------|
| Atomic | CLEVR | color, shape, size, material |
| Binding | CLEVR | red sphere, red cube vs blue cube |
| Relational | CLEVR | left-of, in-front-of |
| Counting | CLEVR | numerosity (3-way) |
| Texture | DTD | striped, dotted, chequered, ... (12 total) |
| Object / Superclass | ImageNet-Mini | has_dog, dog_vs_cat, animal_vs_vehicle, supercategory_4way |
| Binding / Relational | ARO / Visual Genome | matched natural-image versions of CLEVR tasks |

## Methods

- **Linear probe**: ℓ₂-regularized logistic regression (C=1.0, LBFGS)
- **Nonlinear probes**: 3 MLPs with hidden widths (128), (128, 64), (128, 64, 32); ReLU, Adam, early stopping
- **Evaluation**: 5-fold stratified CV with shared fold splits and random seeds across probe types; AUC metric
- **Additivity test**: vector decomposition of color–shape embeddings with SVD residual analysis
- **Cross-model mapping**: linear ridge regression from MAE → DINO feature space
- **Geometry**: linear CKA between probe weight vectors, Kernel PCA, t-SNE, silhouette scores

## Repository Structure

```
clip-linearity-mapping/
├── Code/
│   └── Clip_Linear_Recoverability.ipynb   # Main experiment notebook
├── figures/                                # All generated figures
├── results/                                # Cached probe results (.pkl)
└── README.md
```

## Running the Notebook

The notebook is designed to run on a GPU-equipped machine (tested on NVIDIA L40S). It downloads all datasets (CLEVR, DTD, ImageNet-Mini, ARO) automatically.

```bash
pip install open_clip_torch timm torch torchvision scikit-learn matplotlib seaborn
```

Then open `Code/Clip_Linear_Recoverability.ipynb` and run cells sequentially. Cached feature extractions and probe results are stored in `results/` to avoid redundant computation.

## Citation

```
@misc{chughtai2026linearity,
  title={The Linear Recoverability Hypothesis: When Are Visual Concepts Linearly Recoverable?},
  author={Chughtai, Ayela and Mokhtar, Sarah},
  year={2026},
  note={6.8300 Advances in Computer Vision, MIT}
}
```
