# o-point

GPU surface sampling and multi-view rendering for textured 3D meshes — powered by [nvdiffrast](https://github.com/NVlabs/nvdiffrast), zero C++/CUDA extensions required.

## Overview

`o-point` provides a pure-Python toolkit for loading GLB assets with full PBR material support, sampling surface points with GPU-accelerated texture lookup, and rendering multi-view images. It is built entirely on `nvdiffrast`'s `dr.texture` and rasterization ops, so there are no custom C++ or CUDA kernels to compile.

## Features

- **GLB Loading** — Full glTF 2.0 PBR material parsing (base color, metallic-roughness, normal, occlusion, emissive, alpha mask/blend).
- **GPU Surface Sampling** — Uniform or importance-weighted point sampling on mesh surfaces with differentiable texture interpolation via `dr.texture`.
- **Multi-View Rendering** — Orthographic and perspective camera rendering with depth peeling, producing base color, shading normal, geometric normal, AMR (alpha/metallic/roughness), and occlusion maps.
- **Spatial Queries** *(optional)* — UDF, SDF, and ray-mesh intersection via CUDA BVH (`cubvh`).
- **Importance Analysis** — Per-point texture-gradient estimation via KNN color variance with normal filtering, useful for identifying high-detail surface regions.
- **Export Utilities** — Save sampled point clouds as GLB or PLY with per-point attributes (color, normal, UV, gradients, importance weights).

## Installation

### Prerequisites

- Python ≥ 3.10
- CUDA-capable GPU (for sampling, rendering, and spatial queries)
- PyTorch with CUDA support

### 1. Install PyTorch

Follow the [official guide](https://pytorch.org/get-started/locally/). For example:

```bash
# CUDA 12.1
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

### 2. Install nvdiffrast

```bash
pip install nvdiffrast
```

> **Note:** `nvdiffrast` requires the CUDA driver libraries to be discoverable at runtime. If you encounter `libcuda.so` or `libnvrtc.so` errors, point `LD_LIBRARY_PATH` to your CUDA/conda library directory:
>
> ```bash
> export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
> # or for conda:
> export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
> ```

### 3. Install o-point

```bash
git clone https://github.com/Luh1124/o-point.git
cd o-point
pip install -e .
```

### 4. Optional: spatial queries (UDF / SDF / ray tracing)

```bash
pip install -e ".[spatial]"
```

This installs [`cubvh`](https://github.com/ashawkey/cubvh) and enables distance-field functions. `cubvh` compiles a small CUDA extension on installation; ensure your PyTorch CUDA version matches the system CUDA toolkit.

### Dependencies

- PyTorch (with CUDA)
- nvdiffrast
- trimesh
- numpy
- Pillow
- cubvh *(optional, for spatial queries)*

## Quick Start

### 1. Load a GLB and inspect materials

```python
import o_point

mesh = o_point.load_glb("model.glb", device="cuda")
print(f"{mesh.vertices.shape[0]} vertices, {mesh.faces.shape[0]} faces")
print(f"{len(mesh.materials)} materials, UV sets: {sorted(mesh.uv_sets)}")
```

### 2. Sample surface points

```python
points, attrs = o_point.textured_mesh_to_surface_samples(
    mesh,
    num_samples=1_000_000,
    device="cuda",
)
# points: (N, 3) surface positions
# attrs: dict with base_color, normal, metallic, roughness, uv, etc.
```

### 3. UDF / SDF queries (requires `cubvh`)

```python
mesh = o_point.load_glb("model.glb", device="cuda")

# Unsigned distance field
query_points = torch.randn(100000, 3, device="cuda")
distances, face_ids, uvw = o_point.mesh_udf(
    mesh.vertices, mesh.faces, query_points
)

# Signed distance field (watertight mesh)
sdf, face_ids, uvw = o_point.mesh_sdf(
    mesh.vertices, mesh.faces, query_points, mode="watertight"
)

# Ray-mesh intersection
rays_o = torch.tensor([[0.0, 0.0, 2.0]], device="cuda")
rays_d = torch.tensor([[0.0, 0.0, -1.0]], device="cuda")
hit, face_ids, depth = o_point.ray_trace_mesh(
    mesh.vertices, mesh.faces, rays_o, rays_d
)
```

### 4. Batch processing with rendering

```python
proc = o_point.BatchProcessor(device="cuda", view_size=1024, peel_layers=8)
result = proc.process(
    "model.glb",
    num_samples=1_000_000,
    render=True,
    with_amr=True,
    with_geo_normal=True,
)

result.base_img.save("basecolor.png")
result.normal_img.save("shading_normal.png")
```

### 4. Full demo script

See [`examples/quickdemo.py`](examples/quickdemo.py) for a complete CLI example covering loading, sampling, rendering, and gradient visualization.
A spatial-query demo will be added soon.

```bash
# CPU-only loading smoke test
python examples/quickdemo.py --mesh model.glb --load-only

# Sample 1M points + render 6 views
python examples/quickdemo.py --mesh model.glb --num-samples 1000000 --render

# Render only, no sampling
python examples/quickdemo.py --mesh model.glb --skip-sampling --render
```

## API Modules

| Module | Purpose |
|--------|---------|
| `o_point.materials` | `load_glb`, `NvMesh`, `NvPbrMaterial`, color-space utilities |
| `o_point.sampler` | `MeshTextureSampler`, `BatchProcessor`, `textured_mesh_to_surface_samples` |
| `o_point.render` | `NvMeshRenderer`, `CameraParams`, `render_views`, `render_views_from_glb` |
| `o_point.utils` | Save helpers for GLB/PLY point clouds with per-point attributes |

## Missing-UV Policies

When a textured primitive lacks the required `TEXCOORD_n` attribute, you can choose:

- **`blender`** (default) — Generate local XY UVs for Blender-reference compatibility.
- **`strict`** — Disable texture slots whose required TEXCOORD set is absent.
- **`error`** — Fail fast with a clear message.

## License

TODO
