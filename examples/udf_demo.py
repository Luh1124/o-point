#!/usr/bin/env python3
"""UDF demo: load a GLB, sample volume & near-surface points, query UDF.

Usage:
    cd /home/hongli/code/meshy/texset
    source ./activate
    pixi run python workspace/texvox/o-point/examples/udf_demo.py \
        --mesh "/path/to/model.glb" \
        --num-volume 1000000 \
        --num-near 500000 \
        --out-dir ./viz_udf
"""

from __future__ import annotations

import argparse
import os
import time

import numpy as np
import torch

import o_point


def sample_volume_points(n: int, box_size: float = 1.1, device: str = "cuda") -> torch.Tensor:
    """Uniform random samples in ``[-box_size, box_size]^3``."""
    return torch.rand(n, 3, device=device) * (2 * box_size) - box_size


def sample_near_surface_points(
    mesh: o_point.NvMesh,
    n: int,
    *,
    small_sigma: float = 0.005,
    large_sigma: float = 0.05,
    device: str = "cuda",
) -> torch.Tensor:
    """Sample points near mesh surface via isotropic Gaussian offsets.

    Follows the common "small + large" strategy: half the points get a
    tight offset (narrow band), half get a looser offset (broader band).
    """
    # Uniform surface samples (CPU trimesh → GPU tensor)
    import trimesh

    tm = trimesh.Trimesh(
        vertices=mesh.vertices.cpu().numpy(),
        faces=mesh.faces.cpu().numpy(),
    )
    pts, _ = trimesh.sample.sample_surface(tm, n)
    surface_pts = torch.from_numpy(pts).float().to(device)

    # Random isotropic offsets
    n_half = n // 2
    offsets = torch.randn(n, 3, device=device)
    offsets[:n_half] *= small_sigma
    offsets[n_half:] *= large_sigma

    return surface_pts + offsets


def main() -> None:
    parser = argparse.ArgumentParser(description="o-point UDF sampling demo")
    parser.add_argument("--mesh", type=str, required=True, help="Path to GLB file")
    parser.add_argument("--num-volume", type=int, default=1_000_000, help="Volume sample count")
    parser.add_argument("--num-near", type=int, default=500_000, help="Near-surface sample count")
    parser.add_argument("--box-size", type=float, default=1.1, help="Volume sampling box half-extent")
    parser.add_argument("--small-sigma", type=float, default=0.005, help="Near-surface small offset std")
    parser.add_argument("--large-sigma", type=float, default=0.05, help="Near-surface large offset std")
    parser.add_argument("--out-dir", type=str, default="./viz_udf")
    parser.add_argument("--save-ply", action="store_true", help="Also export PLY point clouds")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this demo")

    os.makedirs(args.out_dir, exist_ok=True)
    device = "cuda"

    # ------------------------------------------------------------------
    # 1. Load mesh (normalized to [-1, 1]^3 by default)
    # ------------------------------------------------------------------
    print(f"Loading {args.mesh} ...")
    t0 = time.time()
    mesh = o_point.load_glb(args.mesh, device=device)
    print(
        f"  Loaded: {mesh.vertices.shape[0]} verts, {mesh.faces.shape[0]} faces, "
        f"bounds [{mesh.vertices.min():.3f}, {mesh.vertices.max():.3f}]  |  "
        f"{time.time() - t0:.2f}s"
    )

    # ------------------------------------------------------------------
    # 2. Sample points
    # ------------------------------------------------------------------
    print(f"\nSampling {args.num_volume} volume points + {args.num_near} near-surface points ...")
    t0 = time.time()

    volume_pts = sample_volume_points(args.num_volume, args.box_size, device)
    near_pts = sample_near_surface_points(
        mesh, args.num_near, small_sigma=args.small_sigma, large_sigma=args.large_sigma, device=device
    )

    print(f"  Done in {time.time() - t0:.2f}s")

    # ------------------------------------------------------------------
    # 3. Query UDF
    # ------------------------------------------------------------------
    print("\nQuerying UDF (volume) ...")
    t0 = time.time()
    volume_udf, volume_face, volume_uvw = o_point.mesh_udf(mesh.vertices, mesh.faces, volume_pts)
    print(f"  Done: udf min={volume_udf.min():.6f} max={volume_udf.max():.6f}  |  {time.time() - t0:.2f}s")

    print("\nQuerying UDF (near-surface) ...")
    t0 = time.time()
    near_udf, near_face, near_uvw = o_point.mesh_udf(mesh.vertices, mesh.faces, near_pts)
    print(f"  Done: udf min={near_udf.min():.6f} max={near_udf.max():.6f}  |  {time.time() - t0:.2f}s")

    # ------------------------------------------------------------------
    # 4. Save results
    # ------------------------------------------------------------------
    stem = os.path.splitext(os.path.basename(args.mesh))[0]

    # Save as compressed .npz for downstream use
    npz_path = os.path.join(args.out_dir, f"{stem}_udf.npz")
    np.savez(
        npz_path,
        volume_pts=volume_pts.cpu().numpy().astype(np.float32),
        volume_udf=volume_udf.cpu().numpy().astype(np.float32),
        near_pts=near_pts.cpu().numpy().astype(np.float32),
        near_udf=near_udf.cpu().numpy().astype(np.float32),
    )
    print(f"\nSaved: {npz_path}")

    # Optional PLY export for visualization
    if args.save_ply:
        from o_point.utils import save_ply_fast

        # Color by UDF: blue (close) → red (far)
        def udf_to_color(udf: torch.Tensor) -> torch.Tensor:
            t = (udf / udf.quantile(0.98)).clamp(0, 1)
            r = (t * 255).round().to(torch.uint8)
            b = ((1 - t) * 255).round().to(torch.uint8)
            g = torch.zeros_like(r)
            return torch.stack([r, g, b], dim=-1)

        vol_color = udf_to_color(volume_udf)
        near_color = udf_to_color(near_udf)

        vol_ply = os.path.join(args.out_dir, f"{stem}_volume.ply")
        near_ply = os.path.join(args.out_dir, f"{stem}_near_surface.ply")

        save_ply_fast(volume_pts.cpu(), vol_ply, vol_color.cpu(), attribute_name="rgb")
        save_ply_fast(near_pts.cpu(), near_ply, near_color.cpu(), attribute_name="rgb")

        print(f"  PLY: {vol_ply}")
        print(f"  PLY: {near_ply}")

    # Summary stats
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"Volume      : {volume_pts.shape[0]} pts  |  udf mean={volume_udf.mean():.4f}  std={volume_udf.std():.4f}")
    print(f"Near-surface: {near_pts.shape[0]} pts  |  udf mean={near_udf.mean():.4f}  std={near_udf.std():.4f}")

    peak = torch.cuda.max_memory_allocated() / (1024.0**3)
    print(f"Peak GPU mem: {peak:.3f} GB")


if __name__ == "__main__":
    main()
