"""o-point: GPU surface sampling with nvdiffrast dr.texture.

Optional spatial queries (UDF/SDF/ray tracing) are available when
``cubvh`` is installed::

    pip install o-point[spatial]
"""

from __future__ import annotations

from .materials import (
    ALPHA_BLEND,
    ALPHA_MASK,
    ALPHA_OPAQUE,
    NvMesh,
    NvPbrMaterial,
    linear_to_srgb,
    load_glb,
    srgb_to_linear,
)
from .spatial import (
    mesh_bvh,
    mesh_sdf,
    mesh_udf,
    ray_trace_mesh,
)
from .utils import (
    save_all_glb_visualizations,
    save_all_ply_visualizations,
    save_glb_fast,
    save_glb_visualization,
    save_ply_fast,
    save_ply_visualization,
    to_glb_visual_attr,
    to_ply_visual_attr,
)

__version__ = "4.0.0"

_LAZY_EXPORTS = {
    "BatchProcessor": (".sampler", "BatchProcessor"),
    "CameraParams": (".render", "CameraParams"),
    "MeshTextureSampler": (".sampler", "MeshTextureSampler"),
    "mesh_bvh": (".spatial", "mesh_bvh"),
    "mesh_sdf": (".spatial", "mesh_sdf"),
    "mesh_udf": (".spatial", "mesh_udf"),
    "NvMeshRenderer": (".render", "NvMeshRenderer"),
    "ray_trace_mesh": (".spatial", "ray_trace_mesh"),
    "render_views": (".render", "render_views"),
    "render_views_from_glb": (".render", "render_views_from_glb"),
    "textured_mesh_to_surface_samples": (
        ".sampler",
        "textured_mesh_to_surface_samples",
    ),
}

__all__ = [
    "ALPHA_BLEND",
    "ALPHA_MASK",
    "ALPHA_OPAQUE",
    "BatchProcessor",
    "CameraParams",
    "MeshTextureSampler",
    "NvMesh",
    "NvMeshRenderer",
    "NvPbrMaterial",
    "linear_to_srgb",
    "load_glb",
    "mesh_bvh",
    "mesh_sdf",
    "mesh_udf",
    "render_views",
    "render_views_from_glb",
    "save_all_glb_visualizations",
    "save_all_ply_visualizations",
    "save_glb_fast",
    "save_glb_visualization",
    "save_ply_fast",
    "save_ply_visualization",
    "srgb_to_linear",
    "textured_mesh_to_surface_samples",
    "ray_trace_mesh",
    "to_glb_visual_attr",
    "to_ply_visual_attr",
]


def __getattr__(name: str) -> object:
    if name not in _LAZY_EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from importlib import import_module

    module_name, attr_name = _LAZY_EXPORTS[name]
    value = getattr(import_module(module_name, __name__), attr_name)
    globals()[name] = value
    return value
