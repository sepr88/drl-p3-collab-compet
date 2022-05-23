
SEED = 7
N_EPISODES = 5000
N_TIMESTEPS_PER_EPISODE = 10000
TRAIN_EVERY = 7
REPEAT_TRAINING = 5
WAIT_UNTIL_TRAINING = 0
TARGET_SCORE = 0.5

# Experience Replay
BUFFER_SIZE = int(1e6)
BATCH_SIZE = 256

GAMMA = 0.99
TAU = 1e-3
WEIGHT_DECAY = 0
NOISE_DECAY = 0.995

# Actor
FC1_UNITS = 400
FC2_UNITS = 300
LR_ACTOR = 1e-4

# Critic
FCS1_UNITS = 400
FCS2_UNITS = 300
LR_CRITIC = 3e-4

# Ornstein-Uhlenbeck Noise
MU = 0.
THETA = 0.15
SIGMA = 0.2