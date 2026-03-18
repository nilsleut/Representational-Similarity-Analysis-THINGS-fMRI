# Representational Similarity Analysis — THINGS-fMRI

Comparing how ResNet-50, ViT-B/16, and CLIP represent objects relative to human brain activity across the visual hierarchy.

## Overview

This project implements Representational Similarity Analysis (RSA) to measure the correspondence between deep learning model representations and fMRI responses in human visual cortex. Models are compared against six visual areas (V1–V4, LOC, IT) using the THINGS-fMRI dataset (Hebart et al. 2023).

**Key finding:** ResNet-50 outperforms both ViT-B/16 and CLIP in higher visual areas (LOC, IT), suggesting that visual hierarchy — not language-grounded semantics — is the dominant organizing principle for isolated object images. CLIP did not improve upon ResNet in IT cortex, constituting a meaningful negative result.

## Method

```
fMRI responses (THINGS-fMRI)          CNN/Transformer activations
        ↓                                        ↓
   RDM (100×100)          ←→           RDM (100×100)
        ↓                                        ↓
              Spearman ρ  (RSA score)
```

1. Select 100 test stimuli from THINGS-fMRI (sub-01), averaged across repetitions
2. Extract activations from intermediate layers of each model
3. Compute Representational Dissimilarity Matrices (RDMs) via correlation distance
4. Compare model RDMs to fMRI RDMs using Spearman rank correlation
5. Compute noise ceiling via split-half reliability (Spearman-Brown corrected)

## Results

| ROI | ResNet-50 ρ | ViT-B/16 ρ | CLIP ρ | Noise Ceiling | Winner |
|-----|------------|------------|--------|---------------|--------|
| V1  | 0.160 | 0.201 | 0.160 | 0.514 | ViT-B/16 |
| V2  | 0.126 | 0.163 | 0.131 | 0.449 | ViT-B/16 |
| V3  | 0.099 | 0.125 | 0.109 | 0.378 | ViT-B/16 |
| V4  | 0.121 | 0.127 | 0.124 | 0.232 | ViT-B/16 |
| LOC | 0.116 | 0.064 | 0.083 | 0.398 | ResNet-50 |
| IT  | 0.136 | 0.086 | 0.121 | 0.407 | ResNet-50 |

ViT-B/16 matches early visual cortex better (V1–V4); ResNet-50 matches higher areas better (LOC, IT). CLIP does not improve over ResNet in semantically-sensitive regions, suggesting isolated object perception is driven more by visual structure than language-grounded concepts.

<img width="1874" height="1296" alt="rsa_results" src="https://github.com/user-attachments/assets/ff6ea898-85b7-419b-a3a3-7c7310c491c5" />

<img width="2627" height="711" alt="rsa_compare_heatmap" src="https://github.com/user-attachments/assets/75d41982-43ab-41f4-bbbe-970aacdfd6e5" />


<img width="1425" height="584" alt="rsa_compare_bar" src="https://github.com/user-attachments/assets/76226b09-5116-467a-88cb-2d48df510530" />


## Models

| Model | Architecture | Training | Layers extracted |
|-------|-------------|----------|-----------------|
| ResNet-50 | CNN | ImageNet classification | layer1–layer4 |
| ViT-B/16 | Transformer | ImageNet classification | block3, 6, 9, 12 |
| CLIP ViT-B/32 | Transformer | 400M image-text pairs | block3, 6, 9, 12 |

## Dataset

**THINGS-fMRI** (Hebart et al. 2023, eLife) — OpenNeuro ds004192  
- 1 subject, 9,840 object images from the THINGS database  
- 6 ROIs: V1, V2, V3, hV4, LOC, IT  
- Voxel-wise responses averaged across ~3 repetitions per image

## Setup

```bash
pip install torch torchvision clip-anytorch h5py scipy pandas matplotlib tqdm
```

Download THINGS-fMRI from [OpenNeuro ds004192](https://openneuro.org/datasets/ds004192) and set paths in `RSA_COMPARE_v2.ipynb`.

## References

- Kriegeskorte, Mur & Bandettini (2008). *Representational similarity analysis.* Frontiers in Systems Neuroscience 2:4.
- Hebart et al. (2023). *THINGS-data.* eLife 12:e82580.
- Radford et al. (2021). *Learning transferable visual models from natural language supervision.* ICML.
- Conwell et al. (2022). *Can deep learning models explain human IT representations?* bioRxiv.
