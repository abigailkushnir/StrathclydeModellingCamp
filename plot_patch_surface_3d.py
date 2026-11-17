"""
Option 2: a real 3D surface of the dura's own shape (not a cortical
surface) extracted directly from patch_segmentation.nii.gz via marching
cubes, colored by patch ID or by a simulation quantity per patch. This
is anatomically faithful to the dura's actual geometry, unlike
projecting onto an unrelated fsLR cortical mesh.

The dura mask has ~580,000 voxels -- too many for a plain-matplotlib
mesh render -- so we downsample the volume first (nearest-neighbour,
to keep integer labels intact) before running marching cubes.
"""

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

# ----------------------------------------------------------------------
# 4) Look up our own simulation results for each face's patch
# ----------------------------------------------------------------------

peak_per_patch, time_to_half_per_patch = compute_per_patch_quantities()
roi_ids = base.df_nodes["ROI_ID"].values  # 1..150, matching the patch labels
peak_lookup = dict(zip(roi_ids, peak_per_patch))
half_lookup = dict(zip(roi_ids, time_to_half_per_patch))

face_peak = np.array([peak_lookup.get(pid, np.nan) for pid in face_patch_id])
face_half = np.array([half_lookup.get(pid, np.nan) for pid in face_patch_id])

# ----------------------------------------------------------------------
# 5) Render: three panels, same mesh, three colorings
# ----------------------------------------------------------------------

panels = [
    ("Patch ID", face_patch_id.astype(float), "nipy_spectral"),
    ("Peak concentration", face_peak, "viridis"),
    ("Time to half-clearance (h)", face_half, "plasma"),
]

fig = plt.figure(figsize=(18, 6))
for i, (title, values, cmap_name) in enumerate(panels, start=1):
    ax = fig.add_subplot(1, 3, i, projection="3d")
    cmap = plt.get_cmap(cmap_name)
    finite = np.isfinite(values)
    norm = Normalize(vmin=np.nanmin(values[finite]), vmax=np.nanmax(values[finite]))
    face_colors = cmap(norm(np.where(finite, values, np.nanmin(values[finite]))))

    mesh = Poly3DCollection(verts[faces], facecolor=face_colors, edgecolor="none")
    ax.add_collection3d(mesh)
    ax.set_xlim(verts[:, 0].min(), verts[:, 0].max())
    ax.set_ylim(verts[:, 1].min(), verts[:, 1].max())
    ax.set_zlim(verts[:, 2].min(), verts[:, 2].max())
    ax.set_title(title)
    fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), ax=ax, shrink=0.6)

fig.suptitle("Real dura surface (marching cubes), colored per patch")
fig.tight_layout()
fig.savefig("patch_surface_3d.png", dpi=150)
print("Wrote patch_surface_3d.png")
