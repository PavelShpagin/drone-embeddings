# VPAIR Visual Place Recognition Benchmark

A comprehensive benchmark for evaluating visual place recognition methods on village/suburban aerial imagery using the VPAIR dataset.

## Quick Start

### Prerequisites

- Python 3.8+
- PyTorch with CUDA support
- Required packages: `opencv-python`, `scikit-learn`, `pandas`, `tqdm`, `matplotlib`

### Install Dependencies

```bash
pip install torch torchvision opencv-python scikit-learn pandas tqdm matplotlib
```

### Run Complete Benchmark

```bash
# Run with default settings (1000 clusters, 25m threshold)
python src/scripts/run_vpair_benchmark.py

# Custom settings
python src/scripts/run_vpair_benchmark.py \
    --n_clusters 1000 \
    --distance_threshold 25.0 \
    --output_dir outputs/my_results
```

## Dataset Structure

The VPAIR dataset should be organized as:

```
third_party/vpair_sample/
├── queries/                 # 200 query images
├── reference_views/         # 200 reference images
├── distractors/            # 200 distractor images
├── poses_query.txt         # Query GPS coordinates
└── poses_reference_view.txt # Reference GPS coordinates
```

## What This Benchmark Does

1. **Generates VLAD Vocabulary**: Uses all 600 VPAIR images to create a robust 1000-cluster vocabulary
2. **Extracts Features**: DINOv2-ViT-B/14 features from image patches
3. **Evaluates Two Methods**:
   - **Concatenated VLAD**: Traditional VLAD descriptors with cosine similarity
   - **Chamfer Similarity**: Direct feature set comparison
4. **Reports Results**: Recall@1, R@5, R@10, R@20 with distance-based ground truth

## Key Advantages

- **Proper Scale**: 600 images (14x larger than typical small benchmarks)
- **Statistical Significance**: Each query = 0.5% of results (vs 8.33% in 44-image benchmarks)
- **Standard Evaluation**: Distance-based recall@K following VPR conventions
- **Village/Suburban Focus**: Addresses gap in aerial VPR research
- **Reproducible**: Complete open-source pipeline

## Expected Performance

Based on method characteristics:

- **Concatenated VLAD**: 60-80% R@20 (better for global similarity)
- **Chamfer Similarity**: 50-70% R@20 (better for local features)

## Output Files

After running the benchmark:

```
outputs/vpair_benchmark/
├── vocabulary/
│   ├── vlad_vocabulary_1000clusters.pkl
│   └── vocabulary_generation_summary.txt
└── evaluation/
    ├── vpair_evaluation_results.pkl
    └── vpair_evaluation_summary.txt
```

## Individual Components

### Generate Vocabulary Only

```bash
python src/scripts/vpair_vlad_vocabulary_generation.py \
    --n_clusters 1000 \
    --output_dir outputs/vocab
```

### Evaluate with Existing Vocabulary

```bash
python src/scripts/vpair_vlad_evaluation.py \
    --vocab_path outputs/vocab/vlad_vocabulary_1000clusters.pkl \
    --output_dir outputs/eval
```

## Research Context

This benchmark addresses critical limitations in visual place recognition research:

### Problem with Small Benchmarks

- Previous benchmarks: 44 total images (inadequate for statistical significance)
- Each query = 8.33% of results (no meaningful discrimination)
- Vocabulary from 116 images (insufficient for robust clustering)

### VPAIR Solution

- **600 total images**: Statistically meaningful scale
- **200 queries**: Proper evaluation power
- **GPS ground truth**: Standard distance-based evaluation
- **1000-cluster vocabulary**: Robust feature quantization

### Comparison to Literature

| Dataset        | Total Images | Queries | Domain             | Use Case           |
| -------------- | ------------ | ------- | ------------------ | ------------------ |
| Previous Small | 44           | 12      | Mixed              | Research demo      |
| **VPAIR**      | **600**      | **200** | **Aerial/Village** | **Drone VPR**      |
| Pittsburgh30k  | 30,000       | 6,816   | Street-level       | Autonomous driving |
| MSLS           | 1.6M         | 30,000+ | Street-level       | Global VPR         |

## Citation

If you use this benchmark in your research, please cite:

```bibtex
@misc{vpair_benchmark_2024,
  title={VPAIR Visual Place Recognition Benchmark},
  author={Your Name},
  year={2024},
  note={Village/Suburban Aerial VPR Evaluation Framework}
}
```

## Contributing

To extend this benchmark:

1. Add new methods in evaluation script
2. Test different vocabulary sizes
3. Experiment with distance thresholds
4. Extend to other aerial datasets

## License

This benchmark is provided for research purposes. Please respect the original VPAIR dataset license terms.
