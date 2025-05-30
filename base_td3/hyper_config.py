### Configuration for ROBO Advisor Training

# Data parameters
HIST_WINDOWS = [15]
PV_HIST_WINDOWS = [15]
ACTION_HIST_WINDOWS = [15]

# Model architecture
EMBEDDING_DIMS = [128]
LAYER1_SIZE = [400]
LAYER2_SIZE = [300]

# Training parameters
NUM_EPISODES = [50]
VALIDATION_EVERY = [5]
RESUME_EVERY = [5]  # frequency for rolling checkpoint saves

# RL hyperparameters
ALPHA = [1e-4]            # Actor learning rate
BETA = [1e-3]             # Critic learning rate
GAMMA = [0.99]            # Discount factor
TAU = [0.005]             # Target network update rate
NOISE_SCALE = [0.1]       # Exploration noise
WARM_UP = [0]            # Warm-up steps before using policy

# Environment parameters
SHARPE_SCALING = [0.1]
SHARPE_SCALING_MONTHLY = [0.2]
SHARPE_SCALING_YEARLY = [0.2]
THETA_MAX = [3.0]
INITIAL_PV = [1000]

USE_VOLUME   = [True]
USE_VOL_10D  = [False]
USE_VOL_30D  = [False]
USE_VOL_90D  = [False]

W_VOL_10D = [0.5]
W_VOL_30D = [0.3]
W_VOL_90D = [0.2]

# Checkpoint labels
LABEL_LATEST = ["latest"]
LABEL_BEST_VAL = ["best_val"]
LABEL_FINAL = ["final"]

# File paths
DATA_PATH = "/home/student/robo_advisor_new/base_td3/data_prep/ECL_NEM_APD_final_input_data.xlsx"
CHECKPOINT_DIR = "base_td3"
LOG_FILE = ["all_logs.xlsx"]
ASSET_LIST = ["ECL", "NEM", "APD"]
