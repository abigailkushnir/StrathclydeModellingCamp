from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.integrate import solve_ivp
# their functions to get CSF curve, L and M matrices, and the ODE system
import src.constants as constants
from src.modeling import fit_CSF_data, setup_matrices
from src.solver import ode_system

DATA_DIR = Path(__file__).parent / "allTheirStuff" / "data" / "paper" / "parameter_estimation"

#### load network ####
df_nodes = pd.read_csv(DATA_DIR / "nodes.csv")
df_edges = pd.read_csv(DATA_DIR / "edges.csv")

Vol_vec = df_nodes["Volume_dm3"].values # from modelling.py and for ode_system
Z_coords = df_nodes["Centroid_Z"].values # for their simulate() function

M_G_dense, M_laplacian = setup_matrices(df_edges, df_nodes) # from modeling.py
N_NODES = M_laplacian.shape[0]


#### get their best fit parameters ####
df_sims = pd.read_csv(DATA_DIR / "simulations.csv")
best_fit = df_sims.sort_values("Weighted_Loss").iloc[0]
print("Best fit parameters:")
print(best_fit[["phi", "P_bot", "D_L", "P_DcLV", "phi_L", "Weighted_Loss"]])

params = (
    best_fit["phi"],     # partition coefficient: dura porosity / lymphatic porosity
    best_fit["P_bot"],   # trans-dural permeability (CSF -> dura)
    best_fit["D_L"],     # diffusion coefficient inside the lymphatic vessels
    best_fit["P_DcLV"],  # clearance rate into the deep cervical lymphatic vessels
    best_fit["phi_L"],   # lymphatic porosity
)


#### part of their simulate() function in solver.py to build stuff for ode_system() ####

z_mid = 200
DcLV_mask = np.where(Z_coords < z_mid, 1.0, 0.0)
_, node_bot_idx = np.argmax(Z_coords), np.argmin(Z_coords)

# Load parameters
phi, _, _, _, phi_L = params
assert 0.1 <= phi * phi_L <= 0.7  # same sanity check simulate() uses

A = (
    constants.K_DARCY_FIXED * M_laplacian
    - sp.eye(N_NODES) * constants.GAMMA_FIXED
)
b = -constants.GAMMA_FIXED * constants.H_D_FIXED * np.ones(N_NODES)
# No-Flux on Top:
A_lil = A.tolil()
# pressure 0 bottom.
A_lil[node_bot_idx, :] = 0
A_lil[node_bot_idx, node_bot_idx] = 1
b[node_bot_idx] = 0
HL = spla.spsolve(A_lil.tocsr(), b)

Delta_H = HL.reshape(1, -1) - HL.reshape(-1, 1)
Q_mat = constants.K_DARCY_FIXED * M_G_dense * Delta_H
M_adv = sp.csr_matrix(
    np.maximum(0, Q_mat) - np.diag(np.sum(np.maximum(0, -Q_mat), axis=1)),
)
q_S_pos = constants.GAMMA_FIXED * np.maximum(0, constants.H_D_FIXED - HL)



#### part of run_qmc_simulations.py file to get full CSF curve through time ####

df_concentration = pd.read_csv(DATA_DIR / "concentration.csv").rename(
    columns={"label": "ROI_ID"},
)
df_dura = df_concentration.query("ROI_ID<1000")
df_csf = df_concentration.query("ROI_ID>1000")
df_csf["ROI_ID"] -= 1000


def clip_invalid(arr):
    arr = np.where(arr < 0, 0, arr)
    arr = np.where(np.isnan(arr), 0, arr)
    return arr


dura_data = {}
for statistic in ["median", "PC25", "PC75"]:
    data = np.zeros((N_NODES, 5))
    data[:, 1:] = df_dura.query(f"statistic=='{statistic}'").pivot(
        values="value",
        index="ROI_ID",
        columns="session",
    )
    dura_data[statistic] = clip_invalid(data)

csf_data_median = np.zeros((N_NODES, 5))
csf_data_median[:, 1:] = clip_invalid(
    df_csf.query("statistic=='median'")
    .pivot(values="value", index="ROI_ID", columns="session")
    .to_numpy(),
)

# Interpolate CSF in time
fitted_csf_functions = fit_CSF_data(csf_data_median, constants.TIME_POINTS)


#### solving ODE system, part of simulate() in solver.py ####

sol = solve_ivp(fun=lambda t, y: ode_system(t, y, params, M_adv, q_S_pos, np.array([f(t) for f in fitted_csf_functions]), M_laplacian, DcLV_mask, Vol_vec),
t_span=(0, 72),
y0 = np.zeros(N_NODES * 2),
t_eval = np.linspace(0, 72, 200),
method = "Radau",
rtol = 1e-3
)
C_D = sol.y[:N_NODES, :]
C_L = sol.y[N_NODES:, :]

#### plot a few nodes ####

fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
for patch in range(0, N_NODES, N_NODES // 6):
    axes[0].plot(sol.t, C_D[patch], label=f"patch {patch}")
    axes[1].plot(sol.t, C_L[patch], label=f"patch {patch}")
axes[0].set(title="Dura concentration $C_D$", xlabel="Time (hours)", ylabel="Concentration")
axes[1].set(title="Lymphatic concentration $C_L$", xlabel="Time (hours)", ylabel="Concentration")
axes[0].legend(fontsize=7)
fig.tight_layout()
fig.savefig("ode_solution.png", dpi=150)