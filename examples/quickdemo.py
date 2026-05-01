#!/usr/bin/env python3
"""Quick demo: load a GLB, sample surface points, export GLB point clouds.

Usage:
    cd <path-to-texset-repo>
    source ./activate
    export PYTHONPATH=workspace/texvox/o-point
    export LD_LIBRARY_PATH=workspace/mvecset/.pixi/envs/default/lib:${LD_LIBRARY_PATH:-}

    # CPU loader smoke test. Works without CUDA.
    workspace/mvecset/.pixi/envs/default/bin/python \
      workspace/texvox/o-point/examples/quickdemo.py \
      --mesh "/home/hongli/demos/demos/ChronographWatch (2).glb" \
             "/home/hongli/demos/demos/chair.glb" \
      --load-only \
      --missing-uv-policy blender

    # Full CUDA sampling + 6-view rendering. Writes GLB point clouds + PNGs.
    workspace/mvecset/.pixi/envs/default/bin/python \
      workspace/texvox/o-point/examples/quickdemo.py \
      --mesh "/home/hongli/demos/demos/ChronographWatch (2).glb" \
             "/home/hongli/demos/demos/chair.glb" \
      --num-samples 1000000 \
      --render \
      --render-size 512 \
      --peel-layers 4 \
      --missing-uv-policy blender

    # Render only, no point sampling.
    workspace/mvecset/.pixi/envs/default/bin/python \
      workspace/texvox/o-point/examples/quickdemo.py \
      --mesh "/home/hongli/demos/demos/chair.glb" \
      --skip-sampling \
      --render \
      --missing-uv-policy blender

    # Missing-UV policies: blender (default), strict, error.
    # blender: use generated local XY UVs for Blender-reference compatibility.
    # strict: disable texture slots whose required TEXCOORD_n is absent.
    # error: fail fast when a textured primitive lacks required TEXCOORD_n.
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import torch
from PIL import Image

import o_point


def _gradient_to_heatmap(grad: torch.Tensor) -> torch.Tensor:
    """Scalar gradient → blue-green-yellow-red heatmap (N, 3) uint8."""
    g = grad.squeeze(-1).float()
    lo = torch.quantile(g, 0.02)
    hi = torch.quantile(g, 0.98)
    if (hi - lo).abs() < 1e-12:
        lo, hi = g.min(), g.max()
    t = ((g - lo) / (hi - lo + 1e-12)).clamp(0.0, 1.0)
    r = (1.5 * t - 0.25).clamp(0.0, 1.0)
    gr = (1.0 - (2.0 * t - 1.0).abs()).clamp(0.0, 1.0)
    b = (1.25 - 1.5 * t).clamp(0.0, 1.0)
    return (torch.stack([r, gr, b], dim=-1) * 255.0).round().to(torch.uint8)


def _export_one_gradient(
    points: torch.Tensor,
    bc_rgb: torch.Tensor,
    grad: torch.Tensor,
    out_dir: str,
    prefix: str,
    label: str,
    top_pct: float = 0.1,
) -> None:
    """Export heatmap + high/mid/low split for one gradient channel."""
    import os
    from o_point.utils import save_glb_fast

    if not 0.0 < top_pct < 0.5:
        raise ValueError(f"top_pct must be in (0, 0.5), got {top_pct}")

    heatmap = _gradient_to_heatmap(grad)
    p = os.path.join(out_dir, f"{prefix}_{label}_heatmap.glb")
    save_glb_fast(points, p, heatmap, attribute_name="rgb")
    print(f"  {p}")

    thresh_high = torch.quantile(grad, 1.0 - top_pct)
    thresh_low = torch.quantile(grad, top_pct)
    mask_high = grad >= thresh_high
    mask_low = grad <= thresh_low
    mask_mid = ~mask_high & ~mask_low

    for tag, mask in [("high", mask_high), ("mid", mask_mid), ("low", mask_low)]:
        count = int(mask.sum())
        if count == 0:
            print(f"  skip {prefix}_{label}_{tag}.glb (0 pts)")
            continue
        p = os.path.join(out_dir, f"{prefix}_{label}_{tag}.glb")
        save_glb_fast(points[mask], p, bc_rgb[mask], attribute_name="rgb")
        print(f"  {p} ({count} pts)")

    print(
        f"  [{label}] min={grad.min():.6f} max={grad.max():.6f} "
        f"mean={grad.mean():.6f}  |  "
        f"high={int(mask_high.sum())} mid={int(mask_mid.sum())} low={int(mask_low.sum())}"
    )


def _export_gradient_viz(
    points: torch.Tensor,
    attrs: dict[str, torch.Tensor],
    out_dir: str,
    prefix: str,
    top_pct: float = 0.25,
) -> None:
    """Export gradient heatmaps + high/mid/low splits for texture & geometry."""
    bc_rgb = (attrs["base_color"].clamp(0.0, 1.0) * 255.0).round().to(torch.uint8)

    print("\n--- Texture gradient ---")
    _export_one_gradient(
        points,
        bc_rgb,
        attrs["texture_gradient"].squeeze(-1),
        out_dir,
        prefix,
        "tex_grad",
        top_pct,
    )

    if "geometry_gradient" in attrs:
        print("\n--- Geometry gradient ---")
        _export_one_gradient(
            points,
            bc_rgb,
            attrs["geometry_gradient"].squeeze(-1),
            out_dir,
            prefix,
            "geo_grad",
            top_pct,
        )

    if "geometry_gradient_nm" in attrs:
        print("\n--- Geometry gradient (with normal map) ---")
        _export_one_gradient(
            points,
            bc_rgb,
            attrs["geometry_gradient_nm"].squeeze(-1),
            out_dir,
            prefix,
            "geo_grad_nm",
            top_pct,
        )

    if "is_geometry_sharp" in attrs:
        import os
        from o_point.utils import save_glb_fast

        sharp = attrs["is_geometry_sharp"].squeeze(-1) > 0.5
        p = os.path.join(out_dir, f"{prefix}_geo_sharp.glb")
        save_glb_fast(points[sharp], p, bc_rgb[sharp], attribute_name="rgb")
        print(f"\n  Sharp geometry: {int(sharp.sum())} pts → {p}")


def _normal_filtered_texture_gradient(
    points: torch.Tensor,
    base_color: torch.Tensor,
    normals: torch.Tensor,
    *,
    knn_k: int,
    normal_cos_thresh: float,
) -> torch.Tensor:
    """Match importance_reweight.py: KNN base-color variance with normal filtering."""
    from scipy.spatial import cKDTree

    pts = np.ascontiguousarray(points.detach().cpu().numpy(), dtype=np.float32)
    rgb = np.ascontiguousarray(
        base_color[:, :3].detach().cpu().numpy(), dtype=np.float32
    )
    nrm = np.ascontiguousarray(normals.detach().cpu().numpy(), dtype=np.float32)
    tree = cKDTree(pts)
    _, knn_idx = tree.query(pts, k=int(knn_k) + 1, workers=-1)
    knn_idx = np.asarray(knn_idx, dtype=np.int64)[:, 1:]

    sq_diff = np.mean((rgb[knn_idx] - rgb[:, None, :]) ** 2, axis=-1)
    cos_sim = np.sum(nrm[knn_idx] * nrm[:, None, :], axis=-1)
    valid = cos_sim >= float(normal_cos_thresh)
    sq_masked = np.where(valid, sq_diff, 0.0)
    valid_count = np.maximum(valid.sum(axis=-1), 1)
    grad = sq_masked.sum(axis=-1) / valid_count
    all_filtered = ~valid.any(axis=-1)
    if np.any(all_filtered):
        grad[all_filtered] = np.mean(sq_diff[all_filtered], axis=-1)
    return torch.from_numpy(grad.astype(np.float32))


def _export_top_texture_gradient_points(
    points: torch.Tensor,
    attrs: dict[str, torch.Tensor],
    out_dir: str,
    prefix: str,
    sizes: list[int],
    knn_k: int,
    normal_cos_thresh: float,
) -> None:
    """Export top-N points by normal-filtered KNN base-color gradient."""
    if not sizes:
        return
    if "base_color" not in attrs or "normal" not in attrs:
        print("  Top texture-gradient export skipped: missing base_color or normal")
        return

    import os
    from o_point.utils import save_glb_fast

    score = _normal_filtered_texture_gradient(
        points,
        attrs["base_color"],
        attrs["normal"],
        knn_k=knn_k,
        normal_cos_thresh=normal_cos_thresh,
    )
    color = (attrs["base_color"].clamp(0.0, 1.0) * 255.0).round().to(torch.uint8)
    order = torch.argsort(score, descending=True)
    for size in sizes:
        n = int(size)
        if n <= 0:
            raise ValueError(f"top gradient size must be > 0, got {size}")
        top_idx = order[: min(n, int(order.numel()))]
        p = os.path.join(out_dir, f"{prefix}_top_texture_nf_{n}.glb")
        save_glb_fast(points[top_idx], p, color[top_idx], attribute_name="rgb")
        min_score = float(score[top_idx].min()) if top_idx.numel() > 0 else float("nan")
        print(
            f"  {p} ({int(top_idx.numel())} pts, "
            f"min normal-filtered texture grad={min_score:.6f})"
        )


def _render_random_views(
    renderer: o_point.NvMeshRenderer,
    mesh: o_point.NvMesh,
    num_views: int,
    out_dir: str,
    stem: str,
    with_amr: bool = False,
    with_geo_normal: bool = False,
    with_occlusion: bool = False,
) -> None:
    """Render random views and save individual PNGs."""
    import os
    from o_point.materials import linear_to_srgb

    os.makedirs(out_dir, exist_ok=True)
    rt: list[str] = ["base_color", "shading_normal"]
    if with_amr:
        rt.append("amr")
    if with_geo_normal:
        rt.append("geo_normal")
    if with_occlusion:
        rt.append("occlusion")
    cameras = renderer.randomize_cameras(mesh, num_views=num_views)
    views = renderer.render_views(mesh, cameras, return_types=tuple(rt))
    for i, vd in enumerate(views):
        bc = linear_to_srgb(vd["base_color"].permute(1, 2, 0)).clamp(0, 1)
        bc_np = (bc.cpu().numpy() * 255.0 + 0.5).astype(np.uint8)
        p = os.path.join(out_dir, f"{stem}_rand{i:03d}_basecolor.png")
        Image.fromarray(bc_np).save(p)

        nrm = vd["shading_normal"].permute(1, 2, 0).clamp(0, 1)
        nrm_np = (nrm.cpu().numpy() * 255.0 + 0.5).astype(np.uint8)
        p_n = os.path.join(out_dir, f"{stem}_rand{i:03d}_shading_normal.png")
        Image.fromarray(nrm_np).save(p_n)

        if with_amr and "amr" in vd:
            amr = vd["amr"].permute(1, 2, 0).clamp(0, 1)
            amr_np = (amr.cpu().numpy() * 255.0 + 0.5).astype(np.uint8)
            p_a = os.path.join(out_dir, f"{stem}_rand{i:03d}_amr.png")
            Image.fromarray(amr_np).save(p_a)

        if with_geo_normal and "geo_normal" in vd:
            gn = vd["geo_normal"].permute(1, 2, 0).clamp(0, 1)
            gn_np = (gn.cpu().numpy() * 255.0 + 0.5).astype(np.uint8)
            p_g = os.path.join(out_dir, f"{stem}_rand{i:03d}_geo_normal.png")
            Image.fromarray(gn_np).save(p_g)

        if with_occlusion and "occlusion" in vd:
            occ = (
                vd["occlusion"]
                .unsqueeze(0)
                .expand(3, -1, -1)
                .permute(1, 2, 0)
                .clamp(0, 1)
            )
            occ_np = (occ.cpu().numpy() * 255.0 + 0.5).astype(np.uint8)
            p_o = os.path.join(out_dir, f"{stem}_rand{i:03d}_occlusion.png")
            Image.fromarray(occ_np).save(p_o)

    print(f"  Saved {num_views} random views to {out_dir}")


def _process_one(
    proc: o_point.BatchProcessor,
    glb_path: str,
    args: argparse.Namespace,
) -> None:
    """Process a single GLB file using the shared BatchProcessor."""
    import os

    stem = glb_path.rsplit("/", 1)[-1].rsplit(".", 1)[0]
    print(f"\n{'=' * 60}")
    print(f"Processing: {glb_path}")

    t0 = time.time()
    if args.load_only:
        mesh = o_point.load_glb(
            glb_path,
            device=args.load_device,
            missing_uv_policy=args.missing_uv_policy,
        )
        print(
            f"  load-only ok: {mesh.vertices.shape[0]} verts, "
            f"{mesh.faces.shape[0]} faces, {len(mesh.materials)} materials, "
            f"uv_sets={sorted(mesh.uv_sets)}  |  {time.time() - t0:.2f}s"
        )
        return

    if args.skip_sampling:
        mesh = o_point.load_glb(
            glb_path,
            device="cuda",
            missing_uv_policy=args.missing_uv_policy,
            use_transmission=args.transmission,
        )
        print(
            f"  render-only load ok: {mesh.vertices.shape[0]} verts, "
            f"{mesh.faces.shape[0]} faces, {len(mesh.materials)} materials, "
            f"uv_sets={sorted(mesh.uv_sets)}"
        )
        if args.render:
            imgs = proc._do_render(
                mesh,
                with_amr=args.amr,
                with_geo_normal=args.geo_normal,
                with_occlusion=args.occlusion,
            )
            os.makedirs(args.render_out, exist_ok=True)
            base_path = os.path.join(args.render_out, f"{stem}_basecolor.png")
            normal_path = os.path.join(args.render_out, f"{stem}_shading_normal.png")
            imgs["base_color"].save(base_path)
            imgs["shading_normal"].save(normal_path)
            print(f"  Rendered: {base_path}")
            if args.amr and "amr" in imgs:
                amr_path = os.path.join(args.render_out, f"{stem}_amr.png")
                imgs["amr"].save(amr_path)
                print(f"  AMR: {amr_path}")
            if args.geo_normal and "geo_normal" in imgs:
                gn_path = os.path.join(args.render_out, f"{stem}_geo_normal.png")
                imgs["geo_normal"].save(gn_path)
                print(f"  Geo normal: {gn_path}")
            if args.occlusion and "occlusion" in imgs:
                occ_path = os.path.join(args.render_out, f"{stem}_occlusion.png")
                imgs["occlusion"].save(occ_path)
                print(f"  Occlusion: {occ_path}")
        if args.random_views > 0:
            _render_random_views(
                proc.renderer,
                mesh,
                args.random_views,
                args.render_out,
                stem,
                with_amr=args.amr,
                with_geo_normal=args.geo_normal,
                with_occlusion=args.occlusion,
            )
        torch.cuda.synchronize()
        print(f"  render-only done  |  {time.time() - t0:.2f}s")
        return

    result = proc.process(
        glb_path,
        num_samples=args.num_samples,
        tonemap_method="srgb",
        compute_importance=args.compute_importance,
        knn_k=args.knn_k,
        render=args.render,
        with_amr=args.amr,
        with_geo_normal=args.geo_normal,
        with_occlusion=args.occlusion,
    )
    torch.cuda.synchronize()
    print(
        f"  {result.mesh.vertices.shape[0]} verts, "
        f"{result.mesh.faces.shape[0]} faces, "
        f"{len(result.mesh.materials)} materials  |  "
        f"{result.points.shape[0]} samples  |  {time.time() - t0:.2f}s"
    )

    points_cpu = result.points.cpu()
    attrs_cpu = {k: v.cpu() for k, v in result.attrs.items()}

    if not args.skip_glb:
        _skip_keys = {
            "importance_weight",
            "texture_gradient",
            "geometry_gradient",
            "geometry_gradient_nm",
            "is_geometry_sharp",
            "normal",  # alias for shading_normal
            "normal_interp",  # alias for geo_normal
        }
        viz_attrs = {k: v for k, v in attrs_cpu.items() if k not in _skip_keys}
        outputs = o_point.save_all_glb_visualizations(
            points_cpu, viz_attrs, args.out_dir, prefix=f"{stem}_cuda"
        )
        print(f"  Saved {len(outputs)} GLB point clouds to {args.out_dir}")

        if args.compute_importance and "texture_gradient" in attrs_cpu:
            _export_gradient_viz(
                points_cpu,
                attrs_cpu,
                args.out_dir,
                prefix=f"{stem}_cuda",
            )

    if args.compute_importance:
        print("\n--- Top normal-filtered texture-gradient point clouds ---")
        _export_top_texture_gradient_points(
            points_cpu,
            attrs_cpu,
            args.out_dir,
            prefix=f"{stem}_cuda",
            sizes=args.top_gradient_sizes,
            knn_k=args.knn_k,
            normal_cos_thresh=args.texture_gradient_normal_cos_thresh,
        )

    if args.render and result.base_img is not None and result.normal_img is not None:
        os.makedirs(args.render_out, exist_ok=True)
        base_path = os.path.join(args.render_out, f"{stem}_basecolor.png")
        normal_path = os.path.join(args.render_out, f"{stem}_shading_normal.png")
        result.base_img.save(base_path)
        result.normal_img.save(normal_path)
        print(f"  Rendered: {base_path}")
        if result.amr_img is not None:
            amr_path = os.path.join(args.render_out, f"{stem}_amr.png")
            result.amr_img.save(amr_path)
            print(f"  AMR: {amr_path}")
        if result.geo_normal_img is not None:
            gn_path = os.path.join(args.render_out, f"{stem}_geo_normal.png")
            result.geo_normal_img.save(gn_path)
            print(f"  Geo normal: {gn_path}")
        if result.occlusion_img is not None:
            occ_path = os.path.join(args.render_out, f"{stem}_occlusion.png")
            result.occlusion_img.save(occ_path)
            print(f"  Occlusion: {occ_path}")

    if args.random_views > 0:
        _render_random_views(
            proc.renderer,
            result.mesh,
            args.random_views,
            args.render_out,
            stem,
            with_amr=args.amr,
            with_geo_normal=args.geo_normal,
            with_occlusion=args.occlusion,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="o-point GPU quick demo")
    parser.add_argument(
        "--mesh",
        type=str,
        nargs="+",
        default=None,
        help="Path(s) to GLB file(s)",
    )
    parser.add_argument(
        "--mesh-dir",
        type=str,
        default=None,
        help="Directory of GLB files (batch mode)",
    )
    parser.add_argument("--num-samples", type=int, default=1_000_000)
    parser.add_argument("--out-dir", type=str, default="./viz_glb")
    parser.add_argument(
        "--skip-sampling",
        action="store_true",
        help="Only load the mesh and optionally render; do not sample points",
    )
    parser.add_argument(
        "--skip-glb",
        action="store_true",
        help="Run sampling but skip writing GLB point-cloud visualizations",
    )
    parser.add_argument(
        "--load-only",
        action="store_true",
        help="Only load GLBs and print mesh/material stats; works on CPU",
    )
    parser.add_argument(
        "--load-device",
        choices=("cpu", "cuda"),
        default="cpu",
        help="Device for --load-only (default: cpu)",
    )
    parser.add_argument(
        "--render", action="store_true", help="Also render 6-view images"
    )
    parser.add_argument("--render-size", type=int, default=1024)
    parser.add_argument("--render-out", type=str, default="./viz_render")
    parser.add_argument("--peel-layers", type=int, default=8)
    parser.add_argument(
        "--missing-uv-policy",
        choices=("blender", "strict", "error"),
        default="blender",
        help="Policy for textured primitives missing TEXCOORD_n",
    )
    parser.add_argument(
        "--compute-importance",
        action="store_true",
        help="Compute per-point importance weight via KNN color gradient",
    )
    parser.add_argument(
        "--knn-k",
        type=int,
        default=16,
        help="KNN neighbors for importance weight (default 16)",
    )
    parser.add_argument(
        "--top-gradient-sizes",
        type=int,
        nargs="*",
        default=[32768, 65536],
        help=(
            "Export top-N point clouds by normal-filtered KNN base-color "
            "gradient when --compute-importance is enabled (default: 32768 "
            "65536; pass no values to disable)"
        ),
    )
    parser.add_argument(
        "--texture-gradient-normal-cos-thresh",
        type=float,
        default=0.5,
        help=(
            "Normal cosine threshold for top texture-gradient export, matching "
            "importance_reweight.py default (0.5)"
        ),
    )
    parser.add_argument(
        "--random-views",
        type=int,
        default=0,
        help="Number of random-camera views to render (0 = none)",
    )
    parser.add_argument(
        "--amr",
        action="store_true",
        help="Also render AMR (alpha, metallic, roughness) images",
    )
    parser.add_argument(
        "--geo-normal",
        action="store_true",
        dest="geo_normal",
        help="Also render geometric (interpolated vertex) normal images",
    )
    parser.add_argument(
        "--occlusion",
        action="store_true",
        help="Also render occlusion images",
    )
    parser.add_argument(
        "--transmission",
        action="store_true",
        help="Approximate KHR_materials_transmission as alpha transparency (off by default for data alignment)",
    )
    args = parser.parse_args()

    if args.skip_sampling and not args.render:
        args.load_only = True

    if not args.load_only and not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available; this demo requires GPU")
    if args.load_only and args.load_device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available; use --load-device cpu")

    # --- Collect GLB paths ---
    import glob as glob_mod
    import os

    glb_paths: list[str] = []
    if args.mesh:
        glb_paths.extend(args.mesh)
    if args.mesh_dir:
        glb_paths.extend(sorted(glob_mod.glob(os.path.join(args.mesh_dir, "*.glb"))))
    if not glb_paths:
        parser.error("Provide --mesh or --mesh-dir")

    print(f"Found {len(glb_paths)} GLB file(s)")

    proc = None
    if not args.load_only:
        proc = o_point.BatchProcessor(
            device="cuda",
            view_size=args.render_size,
            peel_layers=args.peel_layers,
            missing_uv_policy=args.missing_uv_policy,
            use_transmission=args.transmission,
        )

    t_total = time.time()
    for glb_path in glb_paths:
        try:
            if proc is None:
                mesh = o_point.load_glb(
                    glb_path,
                    device=args.load_device,
                    missing_uv_policy=args.missing_uv_policy,
                )
                print(f"\n{'=' * 60}")
                print(f"Processing: {glb_path}")
                print(
                    f"  load-only ok: {mesh.vertices.shape[0]} verts, "
                    f"{mesh.faces.shape[0]} faces, {len(mesh.materials)} materials, "
                    f"uv_sets={sorted(mesh.uv_sets)}"
                )
            else:
                _process_one(proc, glb_path, args)
        except Exception as e:
            print(f"  ERROR on {glb_path}: {e}")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    elapsed = time.time() - t_total
    gpu_peak = (
        torch.cuda.max_memory_allocated() / (1024.0**3)
        if torch.cuda.is_available()
        else 0.0
    )
    print(f"\n{'=' * 60}")
    print(
        f"Done: {len(glb_paths)} file(s) in {elapsed:.1f}s  |  "
        f"Peak GPU: {gpu_peak:.3f} GB"
    )


if __name__ == "__main__":
    main()
