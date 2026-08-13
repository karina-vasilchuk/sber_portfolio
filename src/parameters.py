"""
Model and numerical parameters used in the numerical experiments.

The baseline specification corresponds to the parameter set
used in the final numerical experiment of the thesis.
"""

# ============================================================
# Model parameters
# ============================================================

EPSILON_1 = 0.75
EPSILON_2 = 0.25

MU_1 = 1.0
MU_2 = 0.2

NU = 0.1
T = 4.0


# ============================================================
# Initial conditions
# ============================================================

X1_0 = 2.0
X2_0 = 1.0

Z_0 = X2_0 / X1_0


# ============================================================
# Deterministic switching times
# ============================================================

TAU_DET = 0.394764
THETA_DET = 1.105905
THETA_BAR_DET = 2.001445


# ============================================================
# Baseline stochastic parameters
# ============================================================

SIGMA_1_BASE = 0.15
SIGMA_2_BASE = 0.15
RHO_BASE = 0.0


# ============================================================
# Switching-time fluctuation parameters
# ============================================================

# Characteristic scale of stochastic fluctuations in theta_bar.
# For sigma_1 = sigma_2 = 0.15, the theoretical order is O(sigma^2).
# The value 0.03 is consistent with the Monte Carlo estimate
# std(theta_bar) ≈ 0.0296 reported in the numerical experiment.
SIGMA_THETA_BAR = 0.03


# ============================================================
# Numerical parameters
# ============================================================

DT = 0.001


# ============================================================
# Derived parameters
# ============================================================

MU = MU_1 - MU_2