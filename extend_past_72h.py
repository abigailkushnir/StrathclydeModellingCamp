"""
What happens if we run the real ODE system past the real data window?

The real MRI measurements only cover t = 0 to 72 hours, so the fitted
CSF boundary curves (fitted_csf_functions) are only constrained by real
data up to t=72. Past that, calling them is pure extrapolation of a
curve that the paper itself says has "no mechanistic meaning" -- it was
only chosen because it interpolates the 5 real data points well.

This script solves the same real system out past 72 hours three
different ways, to show how much that extrapolation choice matters:

  1) "extrapolated": just keep calling the fitted curves past t=72,
     trusting their smooth decay shape forever.
  2) "zeroed": freeze the CSF input to zero after t=72 -- the minimum
     extreme: no further tracer input at all past the last real
     measurement, only clearance of what's already inside.
  3) "held constant": freeze the CSF input at its LAST real value
     forever -- the maximum extreme: an undiminished source that never
     stops. Note this one isn't physically plausible on its own (a
     one-time bolus injection can't sustain a constant concentration
     forever by conservation of mass), but it's a useful upper bound.

If the three barely differ, the 72h-window conclusions probably
generalize fine. If they diverge a lot, it means the long-horizon
behaviour is mostly an artifact of which extrapolation you trust, not
something the data actually supports.
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp

import ourSolve as base
from src.solver import ode_system

PATCH = 0          # same patch ourSolve.py already looks at
T_END = 264          # one week, instead of the real data's 72 hours
T_DATA_END = 72      # where the real measurements stop

N_NODES = base.N_NODES
params = base.params


def csf_at_t_extrapolated(t):
    """Just keep evaluating the fitted curves, even past real data (t>72)."""
    return np.array([f(t) for f in base.fitted_csf_functions])


def csf_at_t_zeroed(t):
    """Same as above up to t=72, then frozen at zero -- no further tracer input."""
    if t <= T_DATA_END:
        return csf_at_t_extrapolated(t)
    return np.zeros(N_NODES)


# The last real CSF value per patch, used by the "held constant" variant below.
csf_at_72 = csf_at_t_extrapolated(T_DATA_END)


def csf_at_t_held_constant(t):
    """Same up to t=72, then frozen at the LAST real value forever -- the
    opposite extreme from zeroing: an undiminished, never-ending source.
    """
    if t <= T_DATA_END:
        return csf_at_t_extrapolated(t)
    return csf_at_72


def solve_with(csf_at_t_function):
    sol = solve_ivp(
        fun=lambda t, y: ode_system(
            t, y, params, base.M_adv, base.q_S_pos,
            csf_at_t_function(t),
            base.M_laplacian, base.DcLV_mask, base.Vol_vec,
        ),
        t_span=(0, T_END),
        y0=np.zeros(N_NODES * 2),
        t_eval=np.linspace(0, T_END, 400),
        method="Radau",
        rtol=1e-3,
    )
    C_D = sol.y[:N_NODES, :]
    C_L = sol.y[N_NODES:, :]
    return sol.t, C_D[PATCH] + C_L[PATCH]  # total concentration at PATCH over time


print(f"Solving out to {T_END} hours (real data only covers 0-{T_DATA_END}h)...")
t_extrap, total_extrap = solve_with(csf_at_t_extrapolated)
t_zero, total_zero = solve_with(csf_at_t_zeroed)
t_const, total_const = solve_with(csf_at_t_held_constant)

# ----------------------------------------------------------------------
# Plot all three variants, with a marker showing where real data stops
# ----------------------------------------------------------------------

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(t_extrap, total_extrap, label="CSF curve extrapolated past 72h")
ax.plot(t_zero, total_zero, label="CSF input zeroed after 72h")
ax.plot(t_const, total_const, label="CSF held constant at last value")
ax.axvline(T_DATA_END, color="grey", linestyle="--", label="real data ends (72h)")
ax.set(
    xlabel="Time (hours)",
    ylabel=f"Total concentration $C_D + C_L$ (patch {PATCH})",
    title="Extending the real model past the real 72-hour data window",
)
ax.legend()
fig.tight_layout()
fig.savefig("extend_past_72h.png", dpi=150)
print("Wrote extend_past_72h.png")
