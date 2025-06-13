# --- New Global Localization Algorithm Configuration ---

# --- Map and Coordinate System ---
MAP_IMAGE_PATH = "inference/46.6234, 32.7851.jpg"  # Satellite image with lat/lng in filename
MAP_CENTER_LAT = 46.6234
MAP_CENTER_LNG = 32.7851
M_PER_PIXEL = 0.487  # Meters per pixel for the satellite image

# --- Model Configuration ---
MODEL_WEIGHTS_PATH = "training_results/efficientnet_b0/checkpoints/checkpoint_epoch_11.pth"
BACKBONE_NAME = 'efficientnet_b0'

# --- Patch and Grid Parameters ---
GRID_PATCH_SIZE_M = 100.0  # 100m x 100m patches
CROP_SIZE_PX = 224  # Model input size (224x224 pixels)

# --- Confidence Circle Parameters ---
INITIAL_RADIUS_M = 150.0   # Initial confidence circle radius
MAX_RADIUS_M = 400.0       # Maximum radius before correction is triggered
CORRECTION_THRESHOLD_M = 300.0  # Radius threshold to trigger correction

# --- VIO and Motion Model Parameters ---
VIO_ERROR_STD_M = 1.5      # Standard deviation of VIO measurement error
VIO_X_VARIANCE = 1.0       # X-direction variance for 1D convolution
VIO_Y_VARIANCE = 1.0       # Y-direction variance for 1D convolution
UPDATE_INTERVAL_M = 20.0    # Update localization every 20 meters of travel

# --- Drone Movement Parameters ---
STEP_SIZE_M = 1.0          # Each simulation step represents 1 meter
MAX_SPEED_MPS = 5.0        # Maximum drone speed
MIN_SEGMENT_LENGTH_M = 50.0   # Minimum straight segment length
MAX_SEGMENT_LENGTH_M = 200.0  # Maximum straight segment length
TURN_PROBABILITY = 0.2     # Probability of turning at each new segment
MAX_TURN_ANGLE_RAD = 1.57  # Maximum turn angle (90 degrees)
MAX_TURN_RATE_RAD = 0.087  # Maximum turn rate per step (5 degrees)

# --- Correction Parameters ---
CORRECTION_SPEED_FACTOR = 1.5  # Speed multiplier during correction
RADIUS_SHRINK_FACTOR = 0.9     # Factor to shrink radius during correction
MIN_CORRECTION_DISTANCE_M = 10.0  # Minimum distance to trigger correction

# --- Probability Management ---
NEW_PATCH_PROBABILITY_FACTOR = 0.1  # Factor for new patch probabilities relative to neighbors
MIN_PROBABILITY_THRESHOLD = 1e-9    # Minimum probability to keep in memory

# --- Simulation Parameters ---
NUM_STEPS = 1000
DEVICE = "cuda"

# --- Visualization Parameters ---
VISUALIZATION_FPS = 30
CIRCLE_COLOR = (0, 255, 0)      # Green for confidence circle
TRUE_POS_COLOR = (255, 0, 0)    # Red for true position
VIO_POS_COLOR = (0, 0, 255)     # Blue for VIO position
TRAJECTORY_COLOR = (128, 128, 128)  # Gray for trajectory
PROBABILITY_ALPHA = 0.6         # Transparency for probability visualization 