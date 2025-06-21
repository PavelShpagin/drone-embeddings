import torch

# --- New Global Localization Algorithm Configuration ---

class Config:
    """Configuration for the new global localization algorithm."""
    # --- Map and Coordinate System ---
    MAP_IMAGE_PATH = "inference/46.6234, 32.7851.jpg"  # Satellite image with lat/lng in filename
    MAP_CENTER_LAT = 46.6234
    MAP_CENTER_LNG = 32.7851
    M_PER_PIXEL = 4000.0 / 8192.0  # Meters per pixel for the satellite image, based on 8192px width = 4000m

    # --- Model Configuration ---
    MODEL_WEIGHTS_PATH = "training_results/efficientnet_b0/checkpoints/checkpoint_epoch_11.pth"
    BACKBONE_NAME = 'efficientnet_b0'
    SUPERPOINT_WEIGHTS_PATH = "superpoint_uav_trained/superpoint_uav_final.pth"

    # --- Patch and Grid Parameters ---
    GRID_PATCH_SIZE_M = 100.0  # 100m x 100m patches
    CROP_SIZE_PX = 224  # Model input size (224x224 pixels)

    # --- Confidence Circle Parameters ---
    INITIAL_RADIUS_M = 500.0       # Initial radius of the confidence circle in meters (increased to ensure containment)
    MAX_RADIUS_M = 2000.0          # Maximum allowed radius (increased to allow for larger growth)
    RADIUS_GROWTH_FACTOR = 2.0     # How much radius grows per meter of VIO error (increased for more aggressive containment)
    CORRECTION_THRESHOLD_M = 300.0  # Radius threshold to trigger correction

    # --- VIO and Motion Model Parameters ---
    VIO_ERROR_STD_M = 1.5      # Standard deviation of VIO measurement error
    VIO_X_VARIANCE = 1.0       # X-direction variance for 1D convolution
    VIO_Y_VARIANCE = 1.0       # Y-direction variance for 1D convolution
    UPDATE_INTERVAL_M = 5.0    # Update localization every 5 meters of travel
    NUM_TOP_SP_CANDIDATES = 5 # Number of top candidates to run SuperPoint on during measurement update

    # --- Drone Movement Parameters ---
    STEP_SIZE_M = 1.5          # Average distance drone moves per simulation step
    MAX_SPEED_MPS = 3.0        # Maximum drone speed (adjusted for 1.5m/s average)
    MIN_SEGMENT_LENGTH_M = 200.0  # Minimum length of a straight trajectory segment
    MAX_SEGMENT_LENGTH_M = 500.0 # Maximum length of a straight trajectory segment
    TURN_PROBABILITY = 0.05       # Probability of turning at each step
    MAX_TURN_ANGLE_RAD = 1.57  # Maximum turn angle (90 degrees)
    MAX_TURN_RATE_RAD = 0.087  # Maximum turn rate per step (5 degrees)

    # --- Correction Parameters ---
    CORRECTION_SPEED_FACTOR = 1.5  # Speed multiplier during correction
    RADIUS_SHRINK_FACTOR = 0.5     # Factor to shrink radius during correction
    MIN_CORRECTION_DISTANCE_M = 10.0  # Minimum distance to trigger correction

    # --- Probability Management ---
    NEW_PATCH_PROBABILITY_FACTOR = 0.5  # Factor for new patch probabilities relative to neighbors
    MIN_PROBABILITY_THRESHOLD = 1e-9    # Minimum probability to keep in memory
    NEW_PATCH_DEFAULT_PROB = 1e-6 # Default probability for new patches if no neighbors

    # --- Simulation Parameters ---
    NUM_STEPS = 300           # Total number of simulation steps (for 5 minutes at 30 FPS)
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # --- Visualization Parameters ---
    VISUALIZATION_FPS = 30
    CIRCLE_COLOR = (0, 255, 0)      # Green for confidence circle
    TRUE_POS_COLOR = (255, 0, 0)    # Red for true position
    VIO_POS_COLOR = (0, 0, 255)     # Blue for VIO position
    TRAJECTORY_COLOR = (128, 128, 128)  # Gray for trajectory
    PROBABILITY_ALPHA = 0.6         # Transparency for probability visualization 