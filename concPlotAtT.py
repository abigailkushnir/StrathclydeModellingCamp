from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.ndimage import zoom
from scipy.spatial import cKDTree
from skimage.measure import marching_cubes

import ourSolve as base
from patch_quantities import compute_per_patch_quantities

concType = "total"  # "dura", "lymph", or "total"

DOWNSAMPLE = 4  # shrink the volume by this factor per axis before meshing

SEG_PATH = (
    Path(__file__).parent
    / "allTheirStuff" / "data" / "paper" / "parameter_estimation" / "patch_segmentation.nii.gz"
)

# ----------------------------------------------------------------------
# 1) Load the real parcellation volume and downsample it
# ----------------------------------------------------------------------

labels = nib.load(SEG_PATH).get_fdata()
labels_small = zoom(labels, 1 / DOWNSAMPLE, order=0)  # order=0: nearest-neighbour
dura_mask = (labels_small >= 1) & (labels_small <= 150)
print(f"Downsampled dura mask: {dura_mask.sum()} voxels (from {((labels>=1)&(labels<=150)).sum()})")

# ----------------------------------------------------------------------
# 2) Extract the dura's real surface shape
# ----------------------------------------------------------------------

verts, faces, _, _ = marching_cubes(dura_mask.astype(float), level=0.5)
print(f"Mesh: {len(verts)} vertices, {len(faces)} faces")

# ----------------------------------------------------------------------
# 3) Assign each mesh vertex the patch label of the nearest real dura voxel
# ----------------------------------------------------------------------

dura_voxel_coords = np.argwhere(dura_mask)
dura_voxel_labels = labels_small[dura_mask]
nearest = cKDTree(dura_voxel_coords).query(verts)[1]
vertex_patch_id = dura_voxel_labels[nearest]
face_patch_id = vertex_patch_id[faces[:, 0]].astype(int)  # one patch ID per face


### get ode solution from ourSolve ###

if concType == "dura":
    conc = base.C_D
elif concType == "lymph":
    conc = base.C_L
elif concType == "total":
    conc = base.C_D + base.C_L  # shape (N_NODES, n_times)
sol_times = base.sol.t          # 200 points from 0 to 72 h
T_END = int(sol_times[-1])

# map each face to its row in the solution (ROI_ID 1..150 -> node index)
roi_ids = base.df_nodes["ROI_ID"].values
roi_to_idx = {rid: i for i, rid in enumerate(roi_ids)}
face_node_idx = np.array([roi_to_idx.get(pid, -1) for pid in face_patch_id])
valid = face_node_idx >= 0

# fixed colour scale across ALL frames so the video doesn't flicker
cmap = plt.get_cmap("viridis")
norm = Normalize(vmin=np.nanmin(conc), vmax=np.nanmax(conc))

OUT_DIR = Path(__file__).parent / "frames" / concType
OUT_DIR.mkdir(parents=True, exist_ok=True)

# build the figure once, then just recolour the mesh each frame (much faster)
fig = plt.figure(figsize=(7, 6))
ax = fig.add_subplot(1, 1, 1, projection="3d")
mesh = Poly3DCollection(verts[faces], edgecolor="none")
ax.add_collection3d(mesh)
ax.set_xlim(verts[:, 0].min(), verts[:, 0].max())
ax.set_ylim(verts[:, 1].min(), verts[:, 1].max())
ax.set_zlim(verts[:, 2].min(), verts[:, 2].max())
ax.view_init(elev=20, azim=-60)  # fixed camera
fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), ax=ax, shrink=0.6,
             label=f"{concType} concentration")

for frame, t in enumerate(range(0, T_END + 1, 10)):  ## CHANGE TIME POINTS HERE ##
    k = np.argmin(np.abs(sol_times - t))  # nearest solver time to t

    face_vals = np.zeros(len(faces))
    face_vals[valid] = conc[face_node_idx[valid], k]
    face_colors = cmap(norm(face_vals))
    face_colors[~valid] = (0.7, 0.7, 0.7, 1.0)  # grey for faces with no patch
    mesh.set_facecolor(face_colors)

    ax.set_title(f"{concType} concentration, t = {sol_times[k]:.1f} h")
    fig.savefig(OUT_DIR / f"frame_{frame:04d}.png", dpi=120)
    print(f"Wrote {OUT_DIR / f'frame_{frame:04d}.png'}  (t = {sol_times[k]:.1f} h)")

plt.close(fig)
