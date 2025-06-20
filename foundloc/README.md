# FoundLoc Reproduction (from https://github.com/AnyLoc/FoundLoc)

This folder contains a full copy of the official FoundLoc codebase and scripts to reproduce the results from the paper:

- Paper: [FoundLoc: Vision-based Onboard Aerial Localization in the Wild](https://arxiv.org/pdf/2310.16299)
- Official Repo: https://github.com/AnyLoc/FoundLoc

## Setup Instructions

1. **Clone the FoundLoc repository**

   ```bash
   git clone https://github.com/AnyLoc/FoundLoc.git foundloc/FoundLoc
   ```

2. **Install dependencies**

   FoundLoc uses a Conda environment. From the `foundloc/FoundLoc` directory:

   ```bash
   conda env create -f conda-environment.yml
   conda activate foundloc
   ```

3. **Download the Nardo-Air dataset**

   - Follow the instructions in the FoundLoc repo or paper to obtain the Nardo-Air dataset.
   - If not available, use the provided scripts or contact the authors.

4. **Build VLAD Vocabulary**

   - Use the provided scripts in `FoundLoc/scripts/` to build the VLAD vocab as described in the paper.

5. **Run Evaluation**

   - Use the evaluation scripts to compute Recall@1 and other metrics.
   - Example command (see FoundLoc documentation for details):

   ```bash
   python scripts/evaluate.py --config configs/anyloc_dinovit.yaml --dataset nardo-air
   ```

## Notes

- This folder is a direct copy of the official FoundLoc codebase for reproducibility.
- For questions or issues, see the [FoundLoc GitHub Issues](https://github.com/AnyLoc/FoundLoc/issues).
