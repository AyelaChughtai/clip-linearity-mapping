# CLIP Linearity Mapping

This repository investigates when visual concepts are linearly recoverable from CLIP representations. The project tests the linear representation hypothesis across a graded set of visual concepts, ranging from simpler perceptual attributes to more compositional and relational concepts.

The goal is to understand not only whether CLIP encodes concepts linearly, but also where linear probing begins to fail and what those failures reveal about the geometry of CLIP’s representation space.

## Project Overview

Many interpretability methods assume that high-level concepts can be recovered as linear directions in a model’s representation space. This project evaluates that assumption in the vision-language setting by probing CLIP embeddings across several types of visual concepts.

The analysis focuses on three main questions:

1. Are simple visual concepts linearly recoverable from CLIP embeddings?
2. Do compositional or relational concepts show weaker linear structure?
3. Can geometry metrics such as separability, intrinsic dimension, and representation similarity explain where linear probes succeed or fail?

## Datasets and Concept Types

The experiments use a mix of natural, synthetic, and relational datasets:

- **DTD**: texture concepts
- **ImageNet-style concepts**: object and superclass-level concepts
- **CLEVR-style concepts**: synthetic compositional concepts such as color, shape, material, and relations
- **ARO / Visual Genome**: attribution and relation examples for testing compositional and relational understanding

Concepts are grouped into tiers such as:

- Texture
- Object
- Atomic
- Superclass
- Binding
- Relational
- Counting

## Methods

The repository evaluates CLIP representations using several complementary analyses:

### Linear probing

Linear classifiers are trained on CLIP embeddings to measure how well different visual concepts can be recovered from the representation space.

### Layerwise analysis

Probe performance is compared across CLIP layers to study where different concept types emerge and whether higher-level concepts become more linearly accessible deeper in the model.

### Geometry analysis

The project also analyzes representation geometry using metrics such as:

- Cluster separability
- Silhouette score
- Intrinsic dimension
- Centered kernel alignment (CKA)
- UMAP visualizations
- Residual/error analysis

### Compositionality and interaction tests

Additional experiments examine whether concept directions combine additively and whether failures are driven by interaction effects between attributes, relations, or object bindings.

## Repository Structure

```text
clip-linearity-mapping/
│
├── Code/
│   └── Clip_Linear_Recoverability.ipynb
│
├── figures/
│   ├── additivity_tests.png
│   ├── aro_lowest_acc_groups.png
│   ├── cka.png
│   ├── clevr_lin_gap.png
│   ├── clevr_probe_accuracies.png
│   ├── cluster_separability.png
│   ├── combined_geometry.png
│   ├── dtd_lin_probe_acc.png
│   ├── geometry_clip_cleaner.png
│   ├── imagenet_lin_gap.png
│   ├── layerwise_heatmaps.png
│   ├── layerwise_line_plots.png
│   ├── lin_recoverability.png
│   ├── resid_analysis.png
│   ├── synth_v_natural.png
│   └── umap_clip.png
│
├── results/
│   ├── all_results.pkl
│   ├── additivity_results.pkl
│   ├── aro_results.pkl
│   ├── concept_vectors.pkl
│   ├── dtd_results.pkl
│   ├── layerwise_results.pkl
│   ├── sil_df.pkl
│   └── ...
│
└── README.md
