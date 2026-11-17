"""
Shared helper: solve the real ODE system once and return per-patch summary
quantities (peak concentration, time-to-half-clearance), for the two
visualization scripts (plot_patch_scatter_3d.py, plot_patch_surface_3d.py)
to paint onto the real geometry.

Extrapolates the fitted CSF curves past the real 72h data window (same
approach as extend_past_72h.py / sensitivity_oat.py) so that every patch
actually has a chance to clear to half its peak -- see those scripts for
the caveat this introduces.
"""

import numpy as np
from scipy.integrate import solve_ivp

import ourSolve as base
from src.solver import ode_system

T_END = 264


def csf_at_t_extrapolated(t):
    return np.array([f(t) for f in base.fitted_csf_functions])


def compute_per_patch_quantities(params=None):
    """Returns (peak_per_patch, time_to_half_per_patch), each length N_NODES,
    in the same patch order as base.df_nodes (ROI_ID 1..150).
    """
    if params is None:
        params = base.params

    sol = solve_ivp(
        fun=lambda t, y: ode_system(
            t, y, params, base.M_adv, base.q_S_pos,
            csf_at_t_extrapolated(t),
            base.M_laplacian, base.DcLV_mask, base.Vol_vec,
        ),
        t_span=(0, T_END),
        y0=np.zeros(base.N_NODES * 2),
        t_eval=np.linspace(0, T_END, 400),
        method="Radau",
        rtol=1e-3,
    )
    C_D = sol.y[:base.N_NODES, :]
    C_L = sol.y[base.N_NODES:, :]
    total = C_D + C_L  # shape (N_NODES, n_times)

    peak_per_patch = total.max(axis=1)

    time_to_half_per_patch = np.full(base.N_NODES, np.nan)
    for i in range(base.N_NODES):
        series = total[i]
        peak_idx = np.argmax(series)
        half = series[peak_idx] / 2
        below_half = np.where(series[peak_idx:] <= half)[0]
        if len(below_half) > 0:
            time_to_half_per_patch[i] = sol.t[peak_idx:][below_half[0]]

    return peak_per_patch, time_to_half_per_patch
