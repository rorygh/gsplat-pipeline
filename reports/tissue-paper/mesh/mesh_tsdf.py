"""Extract a mesh from a trained gsplat checkpoint by rendering depth from every
training view and TSDF-fusing (Open3D). Apache-only path -- 'something like SuGaR'.
Usage: python mesh_tsdf.py <data_dir> <checkpoint.pt> <out_prefix> [voxel]
"""
import sys, time
import numpy as np, torch, open3d as o3d
from gsplat.rendering import rasterization
from gsplat_pipeline.colmap.dataset import load_scene, train_eval_split, GaussianSplattingDataset
from gsplat_pipeline.io import load_checkpoint
from gsplat_pipeline.model import get_colors, get_opacities, get_scales

data_dir, ckpt_path, out_prefix = sys.argv[1], sys.argv[2], sys.argv[3]
voxel = float(sys.argv[4]) if len(sys.argv) > 4 else 0.015
dev = "cuda"

t0 = time.time()
scene = load_scene(data_dir, align=True)
params, meta = load_checkpoint(ckpt_path, device=dev)
sh = meta.get("sh_degree", 3)
train_idx, _ = train_eval_split(len(scene.image_names))
ds = GaussianSplattingDataset(scene, train_idx)
print(f"[mesh] {len(train_idx)} views, {params['means'].shape[0]} gaussians, voxel={voxel}")

vol = o3d.pipelines.integration.ScalableTSDFVolume(
    voxel_length=voxel, sdf_trunc=voxel * 5,
    color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8)

t_render = 0.0
for k in range(len(ds)):
    it = ds[k]
    c2w = it["camtoworld"].to(dev)
    K = it["K"].to(dev)[None]
    w, h = int(it["width"]), int(it["height"])
    viewmat = torch.linalg.inv(c2w)[None]
    tr = time.time()
    with torch.no_grad():
        out, alpha, _ = rasterization(
            means=params["means"], quats=params["quats"], scales=get_scales(params),
            opacities=get_opacities(params), colors=get_colors(params),
            viewmats=viewmat, Ks=K, width=w, height=h, sh_degree=sh,
            render_mode="RGB+ED")
    torch.cuda.synchronize(); t_render += time.time() - tr
    rgb = (out[0, ..., :3].clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
    depth = out[0, ..., 3].cpu().numpy().astype(np.float32)
    a = alpha[0, ..., 0].cpu().numpy()
    depth[a < 0.5] = 0.0  # trust only well-covered pixels

    rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
        o3d.geometry.Image(np.ascontiguousarray(rgb)),
        o3d.geometry.Image(np.ascontiguousarray(depth)),
        depth_scale=1.0, depth_trunc=float(np.percentile(depth[depth > 0], 99) * 1.5),
        convert_rgb_to_intensity=False)
    Kn = K[0].cpu().numpy()
    intr = o3d.camera.PinholeCameraIntrinsic(w, h, Kn[0, 0], Kn[1, 1], Kn[0, 2], Kn[1, 2])
    vol.integrate(rgbd, intr, np.linalg.inv(c2w.cpu().numpy()))

mesh = vol.extract_triangle_mesh()
mesh.compute_vertex_normals()
# keep the largest connected component (drops stray background blobs)
ti, n_t, _ = mesh.cluster_connected_triangles()
ti = np.asarray(ti); n_t = np.asarray(n_t)
if len(n_t):
    mesh.remove_triangles_by_mask(ti != int(n_t.argmax()))
    mesh.remove_unreferenced_vertices()

o3d.io.write_triangle_mesh(out_prefix + ".ply", mesh)
o3d.io.write_triangle_mesh(out_prefix + ".obj", mesh)
V, T = len(mesh.vertices), len(mesh.triangles)
print(f"[mesh] {V} verts, {T} tris")
print(f"[mesh] total {time.time()-t0:.1f}s  (render {t_render:.1f}s, TSDF+extract {time.time()-t0-t_render:.1f}s)")
