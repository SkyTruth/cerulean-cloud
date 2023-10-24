"""Constants for AIS analysis handler
"""
import numpy as np

# temporal parameters for AIS trajectory estimation
HOURS_BEFORE = 12
HOURS_AFTER = 2
TOTAL_TIME = HOURS_BEFORE + HOURS_AFTER
NUM_TIMESTEPS = TOTAL_TIME * 10

# buffering parameters for AIS trajectories
AIS_BUFFER = 5000  # buffer around GRD envelope to capture AIS
SPREAD_RATE = 1000  # meters/hour perpendicular to vessel track
BUF_START = 100
BUF_END = BUF_START + SPREAD_RATE * TOTAL_TIME
BUF_VEC = np.linspace(BUF_START, BUF_END, NUM_TIMESTEPS)

# weighting parameters for AIS trajectories
WEIGHT_START = 1.0
WEIGHT_END = 0.1
WEIGHT_VEC = np.logspace(WEIGHT_START, WEIGHT_END, NUM_TIMESTEPS) / 10.0

D_FORMAT = "%Y-%m-%d"
T_FORMAT = "%Y-%m-%d %H:%M:%S"
