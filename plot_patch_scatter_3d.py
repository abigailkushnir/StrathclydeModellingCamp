"""
Option 1: 3D scatter of the real 150 patch centroids, using their real
positions straight from nodes.csv -- no cortical-surface projection
involved, so nothing is lost or distorted the way it was when projecting
the dura onto an unrelated fsLR cortical mesh.

Three panels, same geometry, three colorings:
  - patch ID (what a "parcellation" picture looks like)
  - peak concentration per patch
  - time-to-half-clearance per patch
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import ourSolve as base
from patch_quantities import compute_per_patch_quantities

peak_per_patch, time_to_half_per_patch = compute_per_patch_quantities()

x = base.df_nodes["Centroid_X"].values
y = base.df_nodes["Centroid_Y"].values
z = base.df_nodes["Centroid_Z"].values
patch_id = base.df_nodes["ROI_ID"].values

panels = [
    ("Patch ID", patch_id, "nipy_spectral"),
    ("Peak concentration", peak_per_patch, "viridis"),
    ("Time to half-clearance (h)", time_to_half_per_patch, "plasma"),
]

fig = plt.figure(figsize=(18, 6))
for i, (title, values, cmap) in enumerate(panels, start=1):
    ax = fig.add_subplot(1, 3, i, projection="3d")
    sc = ax.scatter(x, y, z, c=values, cmap=cmap, s=100)
    ax.set_title(title)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    fig.colorbar(sc, ax=ax, shrink=0.6)

fig.suptitle("Real 150-patch dura network, as patch centroids")
fig.tight_layout()
fig.savefig("patch_scatter_3d.png", dpi=150)
print("Wrote patch_scatter_3d.png")
