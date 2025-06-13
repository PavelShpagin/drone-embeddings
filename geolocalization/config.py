# --- Simulation Parameters ---
MAP_IMAGE_PATH = "inference/46.6234, 32.7851.jpg"  # Updated to use the correct satellite image
MODEL_WEIGHTS_PATH = "training_results/efficientnet_b0/final_model.pth"  # Use the best trained model
NUM_STEPS = 2400  # Increased for 2-minute video at 20 FPS (2400 steps = 2 minutes)
VIO_ERROR_STD_M = 2.0  # Standard deviation of VIO error

# Import torch for device detection
import torch
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --- Map and Coordinate System ---
# The map center coordinates are encoded in the filename: 46.6234, 32.7851
MAP_CENTER_LAT = 46.6234
MAP_CENTER_LNG = 32.7851
M_PER_PIXEL = 0.5  # Approximate meters per pixel for the satellite image

# --- Localization State Parameters ---
GRID_PATCH_SIZE_M = 100.0  # Updated to 100m patches as specified
INITIAL_RADIUS_M = 100.0   # Initial confidence circle radius
MAX_RADIUS_M = 500.0       # Increased maximum radius for longer trajectories
MIN_RADIUS_M = 50.0        # Minimum radius after corrections
CORRECTION_THRESHOLD_M = 250.0  # Increased radius threshold to trigger correction

# Motion model parameters for VIO prediction (from paper reference)
VIO_X_VARIANCE = 1.0  # x-direction variance for VIO measurements  
VIO_Y_VARIANCE = 1.0  # y-direction variance for VIO measurements
VIO_THETA_VARIANCE = 0.1  # heading variance (not used in 2D case)

# --- Update Schedule ---
UPDATE_INTERVAL_M = 2.0  # Update every 2 meters of travel for longer segments

# --- Drone Parameters ---
FLIGHT_SPEED_MPS = 8.0  # Increased speed for covering more distance
STEP_SIZE_M = 2.0      # Increased step size for longer movements

# --- Trajectory Parameters (NEW) ---
SEGMENT_LENGTH_MIN_M = 400.0    # Much longer minimum segments (400m)
SEGMENT_LENGTH_MAX_M = 1200.0   # Very long maximum segments (1200m) 
TURN_PROBABILITY = 0.03         # Very low probability of turning (3% per step)
MAX_TURN_ANGLE_DEG = 45.0       # Larger turn angles for exploration
EXPLORATION_RADIUS_M = 1500.0   # Large exploration radius to cover whole map

# --- Image Matching Parameters ---
LIKELIHOOD_STD_DEV = 2.0  # Standard deviation for likelihood calculation
TOP_K_MATCHES = 20        # Number of top matches to consider for each measurement 

# --- VIO Error Model (Realistic Drift) ---
VIO_BIAS_DRIFT_RATE = 0.02      # How fast the bias changes (m/step)
VIO_BIAS_MAX = 5.0              # Maximum bias magnitude (m)
VIO_RANDOM_NOISE_STD = 0.5      # Random noise component (m)
INITIAL_VIO_BIAS = (1.0, 0.5)   # Initial bias in (x, y) directions 