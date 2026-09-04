import sys, numpy as np, open3d as o3d
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
m = o3d.io.read_triangle_mesh(sys.argv[1])
m = m.simplify_quadric_decimation(20000); m.compute_vertex_normals()
V = np.asarray(m.vertices); T = np.asarray(m.triangles); N = np.asarray(m.triangle_normals)
# watertight / stats
mm = o3d.io.read_triangle_mesh(sys.argv[1])
print(f"watertight={mm.is_watertight()}  verts={len(mm.vertices)}  tris={len(mm.triangles)}  "
      f"bbox_extent={np.round(mm.get_axis_aligned_bounding_box().get_extent(),3)}")
light = np.array([0.4,-0.6,0.7]); light/=np.linalg.norm(light)
shade = np.clip(N@light, 0.15, 1.0)
fig = plt.figure(figsize=(15,5))
for k,(el,az) in enumerate([(20,-60),(70,-90),(10,20)]):
    ax = fig.add_subplot(1,3,k+1, projection="3d")
    tris = V[T]
    pc = Poly3DCollection(tris, linewidths=0)
    pc.set_facecolor(np.c_[shade*0.8, shade*0.75, shade*0.7])
    ax.add_collection3d(pc)
    lim = np.c_[V.min(0), V.max(0)]
    ax.set_xlim(lim[0]); ax.set_ylim(lim[1]); ax.set_zlim(lim[2])
    ax.set_box_aspect(V.max(0)-V.min(0)); ax.view_init(el,az)
    ax.set_title(f"view {k+1}"); ax.set_xticklabels([]); ax.set_yticklabels([]); ax.set_zticklabels([])
fig.suptitle(sys.argv[2]); fig.tight_layout()
fig.savefig(sys.argv[3], dpi=90, bbox_inches="tight")
print("saved", sys.argv[3])
