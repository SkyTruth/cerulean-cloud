"""Constants for AIS analysis handler
"""
import numpy as np

# temporal parameters for AIS trajectory estimation
HOURS_BEFORE = 6
HOURS_AFTER = 2
TIMESTEPS_PER_HOUR = 6
NUM_TIMESTEPS = HOURS_BEFORE * TIMESTEPS_PER_HOUR

# buffering parameters for AIS trajectories
AIS_BUFFER = 20000  # buffer around GRD envelope to capture AIS
SPREAD_RATE = 1000  # meters/hour perpendicular to vessel track
BUF_START = 100
BUF_END = BUF_START + SPREAD_RATE * HOURS_BEFORE
BUF_VEC = np.linspace(BUF_START, BUF_END, NUM_TIMESTEPS)

# weighting parameters for AIS trajectories
WEIGHT_START = 2.0
WEIGHT_END = 0.0
WEIGHT_VEC = np.linspace(WEIGHT_START, WEIGHT_END, NUM_TIMESTEPS) / NUM_TIMESTEPS

D_FORMAT = "%Y-%m-%d"
T_FORMAT = "%Y-%m-%d %H:%M:%S"
