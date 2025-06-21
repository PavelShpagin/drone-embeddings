# Geolocalization Simulation: Algorithm and Approach

This document describes the algorithmic approach used in the geolocalization simulation, focusing on the confidence circle, probabilistic reasoning, dynamic database, and correction logic. The system is designed for robust GPS-denied localization using a patch-based map, VIO (Visual-Inertial Odometry) prediction, and image-based measurement updates.

---

## 1. Confidence Circle

- **Purpose:** Limits the search space to a circular region around the estimated position, reducing computation and memory usage.
- **Initialization:**
  - Centered at the initial world position (meters from map center).
  - Radius set to `INITIAL_RADIUS_M` (e.g., 500m).
- **Update:**
  - Center and radius are updated based on VIO motion and correction events.
  - Only patches within the current circle are considered active.

---

## 2. Probability Management

- **Patch Probabilities:**
  - Each patch (grid cell) within the confidence circle has an associated probability representing the belief that the drone is in that patch.
  - Probabilities are stored as a dictionary: `{(grid_row, grid_col): probability}`.
- **Initialization:**
  - Uniform distribution over all patches in the initial circle.
- **Normalization:**
  - After every update, probabilities are normalized to sum to 1.
- **Pruning:**
  - Patches with probability below `MIN_PROBABILITY_THRESHOLD` are pruned to save memory.

---

## 3. Motion Prediction (VIO Update)

- **VIO Delta:**
  - The drone's estimated movement (`vio_delta_m`) is used to shift the probability distribution.
- **Convolution:**
  - Probabilities are converted to a dense grid and shifted by the VIO delta (in grid units).
  - Gaussian blur is applied to model motion uncertainty:
    - Standard deviation: `VIO_ERROR_STD_M / GRID_PATCH_SIZE_M` (in grid units).
    - 1D convolutions are used for efficiency.
- **Circle Growth:**
  - The confidence circle's radius increases with VIO error (`epsilon_m`), up to `MAX_RADIUS_M`.

**Formula:**

- Let $P_{t-1}(i, j)$ be the probability for patch $(i, j)$ at time $t-1$.
- After VIO update:
  $$
  P_t(i, j) = \text{GaussianBlur}(\text{Shift}(P_{t-1}, \Delta_{VIO}))
  $$
  where $\Delta_{VIO}$ is the VIO delta in grid units.

---

## 4. Measurement Update (Image Matching)

- **Embedding Similarity:**
  - The current camera view is embedded and compared to all active patches using L2 distance.
- **Likelihood Calculation:**
  - Similarity is converted to likelihood using an exponential function:
    $$
    \text{likelihood}(d) = \exp\left(-\frac{d - d_{min}}{T}\right)
    $$
    where $d$ is the embedding distance, $d_{min}$ is the minimum distance, and $T$ is a temperature parameter.
- **Bayesian Update:**
  - Posterior probability for each patch:
    $$
    P_{\text{post}}(i, j) \propto \text{likelihood}(i, j) \cdot P_{\text{prior}}(i, j)
    $$
  - Normalize so that $\sum_{i, j} P_{\text{post}}(i, j) = 1$.
- **SuperPoint Refinement:**
  - For the top-5 candidate patches, SuperPoint keypoint matching is used to further refine likelihoods.

---

## 5. Correction Trigger

- **When to Correct:**
  - Correction is triggered if:
    - The confidence circle radius exceeds `CORRECTION_THRESHOLD_M`.
    - There is a confident cluster (patch with probability above threshold and sufficiently far from center).
- **Peak-to-Average Ratio:**
  - Used to assess confidence:
    $$
    \text{PAR} = \frac{\max(P)}{\text{mean}(P)}
    $$
  - Correction is considered if $\text{PAR} > 5.0$ (configurable).
- **Minimum Correction Distance:**
  - Correction is only applied if the most confident patch is at least `MIN_CORRECTION_DISTANCE_M` from the current center.

---

## 6. Correction Application

- **Recenter:**
  - The confidence circle is recentered on the most confident patch.
- **Radius Reset:**
  - The radius is reset to `INITIAL_RADIUS_M` (or shrunk by a factor).
- **Database Update:**
  - Only patches within the new circle are kept in memory; others are pruned.

---

## 7. Dynamic Database

- **Patch Management:**
  - Only patches within the current confidence circle are kept in memory (embeddings, images, probabilities).
  - When the circle moves or grows, new patches are added:
    - Their probabilities are set based on the average of neighboring patches, scaled by `NEW_PATCH_PROBABILITY_FACTOR`.
    - If no neighbors, a small default probability is used (`NEW_PATCH_DEFAULT_PROB`).

---

## 8. Visualization

- **Confidence Circle:**
  - Drawn as a circle on the map, showing the current search region.
- **Probability Heatmap:**
  - Patch probabilities are visualized as a heatmap (color intensity proportional to probability).
- **Trajectories:**
  - True, VIO, and confidence circle center trajectories are plotted for analysis.

---

## 9. Key Config Parameters

- `INITIAL_RADIUS_M`, `MAX_RADIUS_M`, `CORRECTION_THRESHOLD_M`, `MIN_CORRECTION_DISTANCE_M`
- `VIO_ERROR_STD_M`, `VIO_X_VARIANCE`, `VIO_Y_VARIANCE`
- `GRID_PATCH_SIZE_M`, `M_PER_PIXEL`
- `NEW_PATCH_PROBABILITY_FACTOR`, `MIN_PROBABILITY_THRESHOLD`, `NEW_PATCH_DEFAULT_PROB`

---

## 10. Key Formulas

- **Motion Prediction:**
  $$
  P_t = \text{GaussianBlur}(\text{Shift}(P_{t-1}, \Delta_{VIO}))
  $$
- **Measurement Update:**
  $$
  \text{likelihood}(d) = \exp\left(-\frac{d - d_{min}}{T}\right)
  $$
  $$
  P_{\text{post}} \propto \text{likelihood} \cdot P_{\text{prior}}
  $$
- **Peak-to-Average Ratio:**
  $$
  \text{PAR} = \frac{\max(P)}{\text{mean}(P)}
  $$

---

## 11. References

- See `state.py`, `new_localization.py`, and `database.py` for implementation details.
- Configurable parameters are in `config.py` or `new_config.py`.
- Visualization logic is in `visualizer.py` and `new_visualizer.py`.

---

This approach enables scalable, memory-efficient, and robust probabilistic localization with dynamic search space and principled uncertainty management.
