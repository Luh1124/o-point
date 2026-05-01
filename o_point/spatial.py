"""Spatial queries via BVH — UDF, SDF, and ray tracing.

Requires the ``spatial`` extra (cubvh) to be installed::

    pip install o-point[spatial]
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import torch


def _ensure_cubvh():
    try:
        import cubvh
    except ImportError as exc:
        raise ImportError(
            "cubvh is required for spatial queries. "
            "Install it with: pip install o-point[spatial]"
        ) from exc
    return cubvh


def mesh_bvh(
    vertices: np.ndarray | torch.Tensor,
    faces: np.ndarray | torch.Tensor,
):
    """Build a CUDA BVH from a triangle mesh.

    Args:
        vertices: (V, 3) vertex positions.
        faces: (F, 3) triangle indices.

    Returns:
        A ``cubvh.cuBVH`` instance.
    """
    cubvh = _ensure_cubvh()
    return cubvh.cuBVH(vertices, faces)


def mesh_udf(
    vertices: np.ndarray | torch.Tensor,
    faces: np.ndarray | torch.Tensor,
    query_points: torch.Tensor,
    return_uvw: bool = True,
) -> tuple[torch.Tensor, ...]:
    """Unsigned distance field query on a mesh.

    Args:
        vertices: (V, 3) mesh vertices.
        faces: (F, 3) triangle indices.
        query_points: (N, 3) query points. Must be on CUDA.
        return_uvw: If True, also return barycentric coordinates of the
            closest point on the nearest triangle.

    Returns:
        - distances: (N,) unsigned distances (always >= 0).
        - face_ids: (N,) index of the nearest triangle.
        - uvw: (N, 3) barycentric coordinates on the nearest triangle
          (only if *return_uvw* is True).
    """
    bvh = mesh_bvh(vertices, faces)
    return bvh.unsigned_distance(query_points, return_uvw=return_uvw)


def mesh_sdf(
    vertices: np.ndarray | torch.Tensor,
    faces: np.ndarray | torch.Tensor,
    query_points: torch.Tensor,
    *,
    mode: Literal["watertight", "raystab"] = "watertight",
    return_uvw: bool = True,
) -> tuple[torch.Tensor, ...]:
    """Signed distance field query on a mesh.

    Negative values indicate points inside the mesh.

    Args:
        vertices: (V, 3) mesh vertices.
        faces: (F, 3) triangle indices.
        query_points: (N, 3) query points. Must be on CUDA.
        mode: Sign computation strategy. ``watertight`` requires a
            closed mesh; ``raystab`` uses ray stabbing and is more
            robust for open meshes.
        return_uvw: If True, also return barycentric coordinates.

    Returns:
        - distances: (N,) signed distances.
        - face_ids: (N,) index of the nearest triangle.
        - uvw: (N, 3) barycentric coordinates (only if *return_uvw*).
    """
    bvh = mesh_bvh(vertices, faces)
    return bvh.signed_distance(query_points, return_uvw=return_uvw, mode=mode)


def ray_trace_mesh(
    vertices: np.ndarray | torch.Tensor,
    faces: np.ndarray | torch.Tensor,
    rays_o: torch.Tensor,
    rays_d: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Ray-mesh intersection query.

    Args:
        vertices: (V, 3) mesh vertices.
        faces: (F, 3) triangle indices.
        rays_o: (N, 3) ray origins. Must be on CUDA.
        rays_d: (N, 3) ray directions. Must be on CUDA.

    Returns:
        - intersections: (N, 3) hit points (NaN for misses).
        - face_ids: (N,) hit triangle index (-1 for misses).
        - depth: (N,) ray parameter t at hit (inf for misses).
    """
    bvh = mesh_bvh(vertices, faces)
    return bvh.ray_trace(rays_o, rays_d)
