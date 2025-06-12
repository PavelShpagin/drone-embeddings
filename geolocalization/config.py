# --- Simulation Parameters ---
MAP_IMAGE_PATH = "data/earth_imagery/loc1/45.3395, 29.6287.summer.jpg"
NUM_STEPS = 300
VIO_ERROR_MEAN_M = 1.0
VIO_ERROR_STD_M = 2.0

# --- Localization State Parameters ---
GRID_PATCH_SIZE_M = 20.0
INITIAL_RADIUS_M = 10.0
MIN_RADIUS_M = 10.0
RADIUS_SHRINK_FACTOR = 0.95 # Shrink radius by 5% on successful measurement
RADIUS_GROWTH_FACTOR = 0.25  # Grow radius by 25% of the VIO error per step
MOTION_UNCERTAINTY_STD_DEV_M = 1.0 # Base motion uncertainty
LIKELIHOOD_STD_DEV = 5.0

# --- Drone Parameters ---
FLIGHT_SPEED_MPS = 10.0  # Meters per second 