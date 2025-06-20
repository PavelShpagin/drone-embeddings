# FoundLoc Reproduction with Local Data

This folder contains scripts to reproduce the evaluation pipeline of the FoundLoc paper ([arXiv:2310.16299](https://arxiv.org/pdf/2310.16299)) using your own earth imagery data as a stand-in for the Nardo-Air dataset.

## Pipeline Overview

1. **Feature Extraction:**
   - Extract features from all images in `data/earth_imagery/loc1` ... `loc10` using a pretrained DINOvIT model.
2. **VLAD Vocabulary Building:**
   - Run k-means clustering on local features from reference images to build the VLAD codebook.
3. **VLAD Descriptor Computation:**
   - Aggregate features into VLAD descriptors for all reference and query images.
4. **Retrieval & Evaluation:**
   - Compute Recall@1 and Recall@5 by matching queries to references.

## Dependencies

- Python 3.8+
- torch
- timm
- scikit-learn
- numpy
- Pillow

Install with:

```bash
pip install torch timm scikit-learn numpy pillow
```

## Usage

1. Place all your reference/query images in `data/earth_imagery/loc1` ... `loc10`.
2. Run the provided scripts in order:
   - `extract_features.py`
   - `build_vlad_vocab.py`
   - `compute_vlad_descriptors.py`
   - `evaluate_recall.py`

Each script is documented and can be run independently.

## Notes

- This pipeline uses only open-source, pretrained models (no fine-tuning).
- The ground truth for each query is assumed to be the image with the same filename in the reference set.
- For questions or improvements, please open an issue or PR.
