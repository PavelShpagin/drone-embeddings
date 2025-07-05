#!/bin/bash

# FoundLoc-Style DINOv2+VLAD Pipeline with Improved Vocabulary
# This script addresses the issues preventing 90%+ recall performance

set -e  # Exit on any error

echo "============================================================================"
echo "FOUNDLOC-STYLE DINOV2+VLAD PIPELINE WITH IMPROVED VOCABULARY"
echo "============================================================================"
echo "This pipeline addresses the key issues preventing high recall performance:"
echo "1. Insufficient vocabulary data (50k+ crops vs previous 1k)"
echo "2. Domain-specific aerial imagery vocabulary (VPair dataset)"
echo "3. More VLAD clusters (64 vs 32)"
echo "4. Better feature diversity and foundation model utilization"
echo "============================================================================"

# Configuration
DEVICE="cuda"
DATASET_PATH="datasets/earth_benchmark"
VOCAB_PATH="c_centers_improved.pt"
N_CLUSTERS=64

# Check if running on H100
if nvidia-smi | grep -q "H100"; then
    echo "✓ Detected H100 GPU - optimal for large-scale vocabulary generation"
else
    echo "⚠ Not running on H100 - performance may be slower"
fi

# Step 1: Check dependencies
echo ""
echo "Step 1: Checking dependencies..."
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import numpy; print(f'NumPy: {numpy.__version__}')"
python -c "from PIL import Image; print('PIL: OK')"
python -c "from tqdm import tqdm; print('tqdm: OK')"

# Step 2: Check VPair dataset
echo ""
echo "Step 2: Checking VPair dataset..."
if [ ! -d "third_party/vpair_sample/reference_views" ]; then
    echo "ERROR: VPair reference_views not found!"
    echo "Expected: third_party/vpair_sample/reference_views/"
    exit 1
fi

if [ ! -d "third_party/vpair_sample/distractors" ]; then
    echo "ERROR: VPair distractors not found!"
    echo "Expected: third_party/vpair_sample/distractors/"
    exit 1
fi

REF_COUNT=$(find third_party/vpair_sample/reference_views -name "*.png" | wc -l)
DIST_COUNT=$(find third_party/vpair_sample/distractors -name "*.png" | wc -l)

echo "✓ Found $REF_COUNT reference images"
echo "✓ Found $DIST_COUNT distractor images"

if [ $REF_COUNT -lt 100 ]; then
    echo "WARNING: Low number of reference images. Need more for robust vocabulary."
fi

# Step 3: Create dataset if needed
echo ""
echo "Step 3: Checking/creating benchmark dataset..."
if [ ! -d "$DATASET_PATH" ]; then
    echo "Creating benchmark dataset..."
    cd deep-visual-geo-localization-benchmark
    python create_earth_dataset.py
    cd ..
else
    echo "✓ Benchmark dataset exists"
fi

# Step 4: Generate improved vocabulary
echo ""
echo "Step 4: Generating improved FoundLoc-style vocabulary..."
if [ -f "$VOCAB_PATH" ]; then
    echo "Found existing vocabulary at $VOCAB_PATH"
    read -p "Regenerate vocabulary? [y/N]: " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "Regenerating vocabulary..."
        python build_improved_anyloc_vlad_vocab.py
    else
        echo "Using existing vocabulary"
    fi
else
    echo "Generating new vocabulary (this may take 15-30 minutes on H100)..."
    python build_improved_anyloc_vlad_vocab.py
fi

# Verify vocabulary was created
if [ ! -f "$VOCAB_PATH" ]; then
    echo "ERROR: Vocabulary generation failed!"
    exit 1
fi

echo "✓ Vocabulary generated successfully"

# Step 5: Run improved evaluation
echo ""
echo "Step 5: Running improved evaluation with FoundLoc-style vocabulary..."
python eval_improved_anyloc_benchmark.py \
    --dataset_path "$DATASET_PATH" \
    --vocab_path "$VOCAB_PATH" \
    --device "$DEVICE" \
    --n_clusters "$N_CLUSTERS"

# Step 6: Compare with original results
echo ""
echo "Step 6: Comparing with original results..."
echo "============================================================================"
echo "EXPECTED IMPROVEMENTS:"
echo "============================================================================"
echo "Original (small vocabulary, 32 clusters):"
echo "  - Concatenated VLAD: R@1=33.3%, R@5=33.3%, R@10=33.3%, R@20=41.7%"
echo "  - Chamfer Similarity: R@1=33.3%, R@5=41.7%, R@10=66.7%, R@20=75.0%"
echo ""
echo "Improved (FoundLoc-style, 64 clusters, 50k+ vocabulary):"
echo "  - Expected: R@1=70-90%+, R@5=80-95%+, R@10=85-98%+, R@20=90-99%+"
echo ""
echo "Key improvements implemented:"
echo "  ✓ 50x larger vocabulary dataset (50k vs 1k crops)"
echo "  ✓ Domain-specific aerial imagery from VPair"
echo "  ✓ 2x more VLAD clusters (64 vs 32)"
echo "  ✓ Better foundation model feature utilization"
echo "  ✓ Improved feature diversity and representation"
echo "============================================================================"

# Step 7: Provide next steps
echo ""
echo "NEXT STEPS FOR FURTHER IMPROVEMENTS:"
echo "============================================================================"
echo "1. Use even larger vocabulary (100k+ crops) if memory allows"
echo "2. Experiment with different DINOv2 layers (9, 10, 11)"
echo "3. Try different facets ('key', 'value', 'token')"
echo "4. Use more VLAD clusters (128, 256) for very large datasets"
echo "5. Implement learned aggregation instead of hand-crafted VLAD"
echo "6. Add spatial verification for geometric consistency"
echo "============================================================================"

echo ""
echo "Pipeline completed successfully!"
echo "Check the results above to see the performance improvements." 