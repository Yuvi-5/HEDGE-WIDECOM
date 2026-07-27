"""Global constants for the HEDGE simulator. All magic numbers live here."""

# ---------------------------------------------------------------------------
# Pricing parameters (Layer 1/2/2.5 cost and markup model)
# ---------------------------------------------------------------------------
A_REF: float = 5e-10  # USD/cycle - deployment-wide reference valuation
B_SHAPE: float = 1.0  # demand curve shape parameter
ETA: float = 1.0  # markup gain (markup multiplier range: [1, 1+eta])
LAMBDA_MAX: float = 0.5  # SPA aggressiveness cap
THETA_RISK: float = 0.12  # GC risk premium
ALPHA_R: float = 0.2  # EMA smoothing factor for revenue
R_REF: float = 1.0  # USD/RFQ - EMA revenue calibration constant
PI_CO2: float = 5e-5  # USD/g - internal carbon price
MU_C: float = 1.5  # cloud markup multiplier over Layer-1 cost

# ---------------------------------------------------------------------------
# Protocol parameters
# ---------------------------------------------------------------------------
K_MAX: int = 4  # max peers per RFQ fanout
H_MAX: int = 2  # max resale hops
W_FAIL: int = 20  # sliding window length for P_hat_fail
EPSILON_TRANS: float = 0.05  # min resale margin as fraction of C1_i
TAU_LOCK: float = 0.02  # reservation timeout (seconds = 20 ms)
DELTA_Q: float = 0.1  # quote validity/broadcast cadence (seconds = 100 ms)
DELTA_TP: float = 0.1  # Kalman/EMA update cadence (seconds = 100 ms)

# ---------------------------------------------------------------------------
# Energy conversion
# ---------------------------------------------------------------------------
JOULES_PER_KWH: float = 3.6e6  # J/kWh

# ---------------------------------------------------------------------------
# Node heterogeneity ranges (edge server hardware/energy diversity)
# ---------------------------------------------------------------------------
F_MAX_MIN: float = 1e9  # Hz
F_MAX_MAX: float = 5e9  # Hz
KAPPA_MIN: float = 1e-27  # DVFS capacitance constant (farads)
KAPPA_MAX: float = 1e-25
RHO_MIN: float = 2e-4  # CAPEX depreciation rate (USD/s)
RHO_MAX: float = 1e-3
PI_E_MIN: float = 0.06  # electricity price (USD/kWh)
PI_E_MAX: float = 0.18
BETA_MIN: float = 40.0  # carbon intensity (gCO2/kWh) - Nordic hydro
BETA_MAX: float = 700.0  # carbon intensity (gCO2/kWh) - coal-heavy
P_IDLE_OPTIONS: list[float] = [20.0, 30.0, 50.0]  # idle baseline power (W)

# ---------------------------------------------------------------------------
# Task model ranges (per-task workload/data/deadline/valuation draws)
# ---------------------------------------------------------------------------
W_MIN: float = 1e9  # CPU workload lower bound (cycles)
W_MAX: float = 1e11  # CPU workload upper bound (cycles)
S_MIN: float = 1e6  # input data size lower bound (bits = 1 MB)
S_MAX: float = 2e7  # input data size upper bound (bits = 20 MB)
D_MIN: float = 0.05  # deadline lower bound (seconds)
D_MAX: float = 5.0  # deadline upper bound (seconds)
S_RET_FRACTION: float = 0.01  # return data = 1% of input size
A_BUYER_MIN: float = 0.5  # buyer valuation lower bound (USD)
A_BUYER_MAX: float = 10.0  # buyer valuation upper bound (USD)

# ---------------------------------------------------------------------------
# Network topology parameters
# ---------------------------------------------------------------------------
TAU_MESH_MAX: float = 0.005  # max peer-to-peer propagation delay (seconds = 5 ms)
TAU_C_MEAN: float = 0.035  # cloud RTT mean (seconds = 35 ms)
TAU_C_STD: float = 0.008  # cloud RTT standard deviation (seconds)
COVERAGE_RADIUS_KM_DEFAULT: float = 0.25  # C_u radio-coverage radius (km);
# calibrated against the full 125-station real EUA Melbourne dataset to hit
# the paper's "5 to 15 typical candidate nodes" (Section subsec:phase0)

# ---------------------------------------------------------------------------
# Cloud node defaults
# ---------------------------------------------------------------------------
F_MAX_CLOUD: float = 1e12  # effective cloud compute (Hz)
KAPPA_CLOUD: float = 1e-27  # efficient hyperscale DVFS constant
RHO_CLOUD: float = 1e-4  # cloud CAPEX rate (USD/s)
BETA_CLOUD: float = 80.0  # cloud carbon intensity (gCO2/kWh, green PPA)
PI_E_CLOUD: float = 0.06  # cloud electricity price (USD/kWh, bulk rate)

# ---------------------------------------------------------------------------
# Simulation defaults
# ---------------------------------------------------------------------------
N_NODES_DEFAULT: int = 50  # default topology size
WARMUP_DURATION: float = 300.0  # warmup period excluded from metrics (seconds)
