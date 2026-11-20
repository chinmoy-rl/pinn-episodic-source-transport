"""
Central configuration. Every number in the manuscript traces back to this file.
"""
import numpy as np

# ---------------------------------------------------------------- transport
R    = 1.0
DA   = 0.1
PE_LIST = [0.5, 2.0, 10.0]

# ---------------------------------------------------------------- forcing
U0   = 1.0
H    = 0.5
TON  = 1.0
TOFF = 1.75
T_END = 5.0

# ---------------------------------------------------------------- corner ramp
# RAMP_TAU > 0 ramps the inlet up from zero over 0 <= T <= RAMP_TAU with a
# smoothstep, removing the initial-inlet incompatibility at (X, T) = (0, 0)
# while leaving both switching events sharp.
RAMP_TAU = 0.0
RAMP_SWEEP = [0.01, 0.05, 0.10]      # tau = 0 is the main study

# ---------------------------------------------------------------- IC
UB      = 0.0     # initially solute-free => BETA_ST is inert
BETA_ST = 0.5

# ---------------------------------------------------------------- reference
NX_REF    = 800
NX_LADDER = [100, 200, 400, 800]
DT_SAFETY = 0.5
NEUMANN_ORDER = 2          # ghost-node outlet, second order

# ---------------------------------------------------------------- evaluation
NX_EVAL = 401
NT_EVAL = 501
X_EVAL = np.linspace(0.0, 1.0, NX_EVAL)
T_EVAL = np.linspace(0.0, T_END, NT_EVAL)
SNAPSHOTS = [0.25, 0.75, 1.0, 1.25, 1.75, 2.0, 2.5, 3.5]
DELTA_SWITCH  = 0.10       # half-width of the switching windows
CORNER_RADIUS = 0.05
# All errors are absolute RMSE (the solution is already scaled by U0).

# ---------------------------------------------------------------- network
LAYERS = [2, 64, 64, 64, 64, 1]
EPOCHS = 60000
LR     = 1e-3
LR_DECAY_AT = 45000        # step decay to 1e-4 for the final quarter
N_F  = 10000
N_IC = 1000
N_BC = 1000
SEEDS = [42, 123, 456, 789, 1024, 2024, 3031, 4099, 5113, 6007,
         7019, 8123, 9001, 10007, 11113, 12007, 13001, 14009, 15013, 16001]
# All variants share the same seeds, so comparisons between variants are paired.

# focused sampling (reallocates points; total N_F unchanged)
N_F_FOCUS   = 833
DELTA_FOCUS = 0.25

# proximity weighting of the PDE residual
BETA_W  = 10.0
SIGMA_T = 0.2
BETA_W_SWEEP = [1.0, 3.0, 30.0]      # plus 0 (= base) and 10 (= pw)

# residual-based adaptive distribution, RAD (Wu et al., 2023)
RD_EVERY = 1000
RD_POOL  = 100000
RD_K     = 1.0
RD_C     = 1.0

# causal training
CAUSAL_BINS = 40
CAUSAL_EPS  = 1.0

# domain decomposition
DD_INTERFACE = 200

# hard inlet constraint
HC_LIFT   = "exp"
HC_LENGTH = 0.05
HC_LIFT_SWEEP = [("quad", 0.0), ("exp", 0.02), ("exp", 0.10)]   # plus exp 0.05 (= hc)

# ---------------------------------------------------------------- variants
VARIANTS = ["base", "fs", "pw", "rd", "causal", "dd", "hc"]

LABELS = {
    "base":   "PINN-B",
    "fs":     "PINN-FS",
    "pw":     "PINN-PW",
    "rd":     "PINN-RD",
    "causal": "PINN-C",
    "dd":     "PINN-DD",
    "hc":     "PINN-HC",
}
LONG = {
    "base":   "baseline",
    "fs":     "focused sampling",
    "pw":     "proximity weighting",
    "rd":     "residual-adaptive sampling",
    "causal": "causal weighting",
    "dd":     "temporal domain decomposition",
    "hc":     "hard inlet constraint",
}
ABLATION = {   # sampling, residual weighting, architecture, uses switch times
    "base":   ("uniform",  "none",      "single",        "no"),
    "fs":     ("focused",  "none",      "single",        "yes"),
    "pw":     ("uniform",  "proximity", "single",        "yes"),
    "rd":     ("RAD",      "none",      "single",        "no"),
    "causal": ("uniform",  "causal",    "single",        "no"),
    "dd":     ("uniform",  "none",      "3 subnetworks", "yes"),
    "hc":     ("uniform",  "none",      "lifted ansatz", "yes"),
}
COST_WEIGHT = {"dd": 3.0}   # relative cost, used to balance parallel workers

# ---------------------------------------------------------------- inverse problem
# Recover Toff from noisy downstream breakthrough curves. The inlet step is
# smoothed with a sigmoid of width eps, annealed from INV_EPS0 to INV_EPS1 over
# the first INV_ANNEAL fraction of training.
INV_EPOCHS   = 30000
INV_SEEDS    = SEEDS
INV_TOFF0    = 2.5          # deliberately wrong initial guess
INV_EPS0     = 0.20
INV_EPS1     = 0.02
INV_ANNEAL   = 0.6
INV_W_DATA   = 1.0
INV_N_BC     = 4000         # denser inlet sampling for the inverse problem
INV_DT_OBS   = 0.05         # sampling interval at each well
INV_WELLS = {
    "near":   [0.10, 0.20, 0.30],
    "spread": [0.25, 0.50, 0.75],
    "far":    [0.60, 0.75, 0.90],
}
INV_NOISE = [0.0, 0.05, 0.10]   # sigma as a fraction of max |U| at the wells
INV_GRID = ([(2.0, w, n) for w in INV_WELLS for n in INV_NOISE]
            + [(0.5, "spread", 0.05), (10.0, "spread", 0.05)])


def inlet(T):
    """Dimensionless inlet forcing g(T)."""
    T = np.asarray(T, dtype=float)
    g = np.zeros_like(T)
    m1 = T < TON
    m3 = T >= TOFF
    g[m1] = U0 * np.exp(-H * T[m1])
    g[m3] = U0 * np.exp(-H * (T[m3] - TOFF))
    if RAMP_TAU > 0:
        r = np.clip(T / RAMP_TAU, 0.0, 1.0)
        g = g * (3 * r ** 2 - 2 * r ** 3)      # smoothstep: g(0)=0, g'(0)=0
    return g


def stable_dt(Nx, Pe, safety=DT_SAFETY):
    """von Neumann bound for the explicit FTCS scheme."""
    dX = 1.0 / Nx
    return safety * min(R * Pe * dX ** 2 / 2.0, 2.0 * R / Pe)


def signature():
    """
    Physics fingerprint stored in every cache and result file. Files whose
    signature differs from the current parameters are not reused.
    """
    return (f"R={R};Da={DA};H={H};Ton={TON};Toff={TOFF};U0={U0};"
            f"Ub={UB};Tend={T_END};NxRef={NX_REF};order={NEUMANN_ORDER}"
            + (f";tau={RAMP_TAU}" if RAMP_TAU else ""))
