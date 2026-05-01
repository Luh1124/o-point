"""GLB export and visualization utilities."""

from __future__ import annotations

import os

import numpy as np
import torch
import trimesh


def _as_points_np(points: torch.Tensor) -> np.ndarray:
    points_np = points.detach().cpu().numpy().astype(np.float32, copy=False)
    if points_np.ndim != 2 or points_np.shape[1] != 3:
        raise ValueError("points must be (N, 3)")
    return points_np


def _as_rgba_np(attribute: torch.Tensor | None, num_points: int) -> np.ndarray | None:
    if attribute is None:
        return None
    attr_np = attribute.detach().cpu().numpy()
    if attr_np.ndim == 1:
        attr_np = attr_np[:, None]
    if attr_np.shape[0] != num_points:
        raise ValueError("attribute length must match points")
    if attr_np.shape[1] == 1:
        rgb = np.repeat(attr_np, 3, axis=1)
    else:
        rgb = attr_np[:, :3]
    if np.issubdtype(rgb.dtype, np.floating):
        rgb = np.clip(rgb, 0.0, 1.0) * 255.0
    rgb = np.clip(rgb, 0, 255).round().astype(np.uint8, copy=False)
    alpha = np.full((num_points, 1), 255, dtype=np.uint8)
    return np.concatenate([rgb, alpha], axis=1)


def save_glb_fast(
    points: torch.Tensor,
    output_path: str,
    attribute: torch.Tensor | None = None,
    attribute_name: str = "attr",
) -> None:
    """Write a binary GLB point cloud with optional RGB/RGBA colors."""
    del attribute_name
    points_np = _as_points_np(points)
    colors = _as_rgba_np(attribute, points_np.shape[0])
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    cloud = trimesh.points.PointCloud(points_np, colors=colors)
    trimesh.Scene(cloud).export(output_path, file_type="glb")


def save_ply_fast(
    points: torch.Tensor,
    output_path: str,
    attribute: torch.Tensor | None = None,
    attribute_name: str = "attr",
) -> None:
    """Backward-compatible alias that now writes GLB via trimesh."""
    del attribute_name
    save_glb_fast(points, _with_glb_suffix(output_path), attribute)


def _with_glb_suffix(path: str) -> str:
    root, ext = os.path.splitext(path)
    return path if ext.lower() == ".glb" else f"{root}.glb"


def to_ply_visual_attr(attr: torch.Tensor, key: str) -> torch.Tensor:
    """Convert an attribute tensor to uint8 RGB for GLB visualization."""
    if attr.ndim == 1:
        attr = attr[:, None]
    if key == "normal":
        vis = (attr + 1.0) * 0.5
    else:
        vis = attr
    vis = vis.clamp(0.0, 1.0)
    if vis.shape[1] == 1:
        vis = vis.repeat(1, 3)
    elif vis.shape[1] >= 3:
        vis = vis[:, :3]
    return (vis * 255.0).round().to(torch.uint8)


def save_ply_visualization(
    points: torch.Tensor,
    attrs: dict[str, torch.Tensor],
    out_dir: str,
    attr_key: str,
    save_raw: bool = False,
) -> str:
    """Save a single attribute as a colored GLB point cloud."""
    os.makedirs(out_dir, exist_ok=True)
    output_path = os.path.join(out_dir, f"{attr_key}.glb")
    if attr_key not in attrs:
        raise ValueError(f"Attribute {attr_key} not found in outputs")
    vis_attr = to_ply_visual_attr(attrs[attr_key], attr_key)
    save_glb_fast(points, output_path, vis_attr, attribute_name="rgb")
    if save_raw:
        raw_output = os.path.join(out_dir, f"{attr_key}_raw.glb")
        save_glb_fast(points, raw_output, attrs[attr_key], attribute_name=attr_key)
    return output_path


def save_all_ply_visualizations(
    points: torch.Tensor,
    attrs: dict[str, torch.Tensor],
    out_dir: str,
    prefix: str = "",
    save_raw: bool = False,
) -> list[str]:
    """Save all attributes as colored GLB point clouds."""
    os.makedirs(out_dir, exist_ok=True)
    outputs: list[str] = []
    for key in sorted(attrs.keys()):
        fname = f"{prefix}_{key}.glb" if prefix else f"{key}.glb"
        output_path = os.path.join(out_dir, fname)
        vis_attr = to_ply_visual_attr(attrs[key], key)
        save_glb_fast(points, output_path, vis_attr, attribute_name="rgb")
        outputs.append(output_path)
        if save_raw:
            raw_fname = f"{prefix}_{key}_raw.glb" if prefix else f"{key}_raw.glb"
            raw_output = os.path.join(out_dir, raw_fname)
            save_glb_fast(points, raw_output, attrs[key], attribute_name=key)
    return outputs


to_glb_visual_attr = to_ply_visual_attr
save_glb_visualization = save_ply_visualization
save_all_glb_visualizations = save_all_ply_visualizations
