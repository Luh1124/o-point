"""NvMesh renderer with DepthPeeler transparency, random cameras, and PBR output.

Provides :class:`NvMeshRenderer` — a stateful renderer that holds a shared
``dr.RasterizeCudaContext`` and supports:

* ``render()`` — single-view rendering returning a ``dict[str, Tensor]``
* ``render_grid()`` — multi-view rendering tiled into PIL Images
* ``randomize_cameras()`` — random camera generation (à la TRELLIS pbr_vae)
* ``build_6_views()`` — deterministic 6-view camera layout

Legacy functions :func:`render_views` and :func:`render_views_from_glb` are
kept as thin wrappers for backward compatibility.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Union

import numpy as np
import nvdiffrast.torch as dr
import torch
import torch.nn.functional as F
from PIL import Image

from .materials import (
    ALPHA_BLEND,
    ALPHA_MASK,
    ALPHA_OPAQUE,
    NvMesh,
    _prepare_texture_uv,
    linear_to_srgb,
    load_glb,
    srgb_to_linear,
)

__all__ = [
    "CameraParams",
    "NvMeshRenderer",
    "render_views",
    "render_views_from_glb",
]

VIEW_SIZE = 1024
DEFAULT_PEEL_LAYERS = 8

ALL_RETURN_TYPES = frozenset(
    {
        "base_color",
        "metallic",
        "roughness",
        "alpha",
        "emissive",
        "normal",
        "shading_normal",
        "geo_normal",
        "occlusion",
        "depth",
        "mask",
        "position",
        "amr",
        "layer_position",
        "layer_alpha",
        "layer_weight",
        "layer_mask",
    }
)


# ---------------------------------------------------------------------------
# Camera data
# ---------------------------------------------------------------------------


@dataclass
class CameraParams:
    """Camera parameters for one or more views.

    All tensors have a leading *N* (num-views) dimension.
    """

    extrinsics: torch.Tensor  # (N, 4, 4) world-to-camera
    intrinsics: torch.Tensor  # (N, 3, 3) OpenCV-style
    near: List[float]  # per-view near plane
    far: List[float]  # per-view far plane
    names: List[str] = field(default_factory=list)

    def __len__(self) -> int:
        return self.extrinsics.shape[0]


# ---------------------------------------------------------------------------
# Camera utilities
# ---------------------------------------------------------------------------


def _extrinsics_look_at(
    eye: np.ndarray,
    target: np.ndarray,
    up: np.ndarray,
) -> np.ndarray:
    """Build OpenCV-style world-to-camera 4x4 matrix.

    Matches ``utils3d.torch.extrinsics_look_at`` used by TRELLIS:
      z = target - eye  (camera looks along +z in camera space)
      x = cross(-up, z) (right)
      y = cross(z, x)   (down in screen space)
    """
    z = target - eye
    z = z / np.linalg.norm(z)
    x = np.cross(-up, z)
    x = x / np.linalg.norm(x)
    y = np.cross(z, x)
    y = y / np.linalg.norm(y)
    R = np.stack([x, y, z], axis=0).astype(np.float32)
    t = -R @ eye.astype(np.float32)
    mat = np.eye(4, dtype=np.float32)
    mat[:3, :3] = R
    mat[:3, 3] = t
    return mat


def _intrinsics_to_projection(
    intrinsics: torch.Tensor,
    near: float,
    far: float,
) -> torch.Tensor:
    """OpenCV intrinsics (3x3) → OpenGL perspective matrix (4x4)."""
    fx, fy = intrinsics[0, 0], intrinsics[1, 1]
    cx, cy = intrinsics[0, 2], intrinsics[1, 2]
    ret = torch.zeros((4, 4), dtype=intrinsics.dtype, device=intrinsics.device)
    ret[0, 0] = 2 * fx
    ret[1, 1] = 2 * fy
    ret[0, 2] = 2 * cx - 1
    ret[1, 2] = -2 * cy + 1
    ret[2, 2] = (far + near) / (far - near)
    ret[2, 3] = 2 * near * far / (near - far)
    ret[3, 2] = 1.0
    return ret


def _mesh_aabb(mesh: NvMesh) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Return (aabb_min, aabb_max, center, extent_max)."""
    aabb_min = mesh.vertices.min(dim=0).values.cpu().numpy()
    aabb_max = mesh.vertices.max(dim=0).values.cpu().numpy()
    center = (aabb_min + aabb_max) / 2.0
    extent = float(np.max(aabb_max - aabb_min))
    return aabb_min, aabb_max, center, extent


# ---------------------------------------------------------------------------
# Renderer
# ---------------------------------------------------------------------------


class NvMeshRenderer:
    """GPU PBR renderer backed by nvdiffrast DepthPeeler.

    Holds a shared ``dr.RasterizeCudaContext`` that is created once and
    reused across all ``render`` / ``render_grid`` calls.

    Example::

        renderer = NvMeshRenderer(device="cuda")
        mesh = load_glb("model.glb", device="cuda")

        # Single random view → tensor dict
        cam = renderer.randomize_cameras(mesh, num_views=1)
        out = renderer.render(mesh, cam.extrinsics[0], cam.intrinsics[0],
                              cam.near[0], cam.far[0])
        base_color = out["base_color"]  # (3, H, W)

        # 6-view grid → dict of PIL Images
        grids = renderer.render_grid(mesh)  # keys: "base_color", "shading_normal"
    """

    def __init__(
        self,
        device: str = "cuda",
        resolution: int = VIEW_SIZE,
        peel_layers: int = DEFAULT_PEEL_LAYERS,
    ) -> None:
        self.device = device
        self.resolution = resolution
        self.peel_layers = peel_layers
        self._glctx = dr.RasterizeCudaContext(device=device)

    @property
    def glctx(self) -> object:
        return self._glctx

    # ------------------------------------------------------------------
    # Camera generators
    # ------------------------------------------------------------------

    def build_6_views(self, mesh: NvMesh, fov: float = 45.0) -> CameraParams:
        """Deterministic 6-view layout (5 azimuth + top)."""
        device = mesh.vertices.device
        aabb_min, aabb_max, center, _ = _mesh_aabb(mesh)
        half_size = float(np.max(aabb_max - aabb_min)) / 2.0
        dist_cam = (
            half_size / math.tan(math.radians(fov / 2.0)) if half_size > 1e-8 else 1.0
        )

        w2c_list: list[np.ndarray] = []
        names: list[str] = []
        for deg in [0.0, 45.0, 135.0, 225.0, 315.0]:
            a = math.radians(deg)
            eye = (
                center
                + np.array([math.sin(a), 0, math.cos(a)], dtype=np.float32) * dist_cam
            )
            w2c_list.append(
                _extrinsics_look_at(eye, center, np.array([0, 1, 0], dtype=np.float32))
            )
            names.append(f"az{int(deg):03d}")
        eye_top = center + np.array([0, dist_cam, 0], dtype=np.float32)
        w2c_list.append(
            _extrinsics_look_at(eye_top, center, np.array([0, 0, -1], dtype=np.float32))
        )
        names.append("top")

        extrinsics = torch.from_numpy(np.stack(w2c_list)).to(device)

        focal = 0.5 * self.resolution / math.tan(math.radians(fov / 2.0))
        intr = torch.tensor(
            [
                [focal / self.resolution, 0.0, 0.5],
                [0.0, focal / self.resolution, 0.5],
                [0.0, 0.0, 1.0],
            ],
            dtype=torch.float32,
            device=device,
        )
        intrinsics = intr.unsqueeze(0).expand(len(w2c_list), -1, -1)

        extent = float(np.max(aabb_max - aabb_min))
        near = [max(dist_cam - extent, 0.01)] * len(w2c_list)
        far = [dist_cam + extent] * len(w2c_list)

        return CameraParams(
            extrinsics=extrinsics,
            intrinsics=intrinsics,
            near=near,
            far=far,
            names=names,
        )

    def randomize_cameras(
        self,
        mesh: NvMesh,
        num_views: int = 1,
        radius_range: Optional[tuple[float, float]] = None,
        fov: float = 45.0,
    ) -> CameraParams:
        """Generate random cameras looking at the mesh center.

        Radius is sampled with inverse-square distribution (uniform in
        projected size) following TRELLIS ``pbr_vae._randomize_camera``.

        Args:
            mesh: target mesh (used for AABB-based defaults).
            num_views: number of random cameras.
            radius_range: ``(r_min, r_max)``; if *None*, derived from AABB.
            fov: vertical field-of-view in degrees.
        """
        device = mesh.vertices.device
        _, _, center_np, extent = _mesh_aabb(mesh)
        center = torch.from_numpy(center_np).to(device)

        if radius_range is None:
            half = extent / 2.0
            r_base = half / math.tan(math.radians(fov / 2.0)) if half > 1e-8 else 1.0
            radius_range = (r_base * 0.8, r_base * 3.0)

        r_min, r_max = radius_range
        k_min = 1.0 / r_max**2
        k_max = 1.0 / r_min**2
        ks = torch.rand(num_views, device=device) * (k_max - k_min) + k_min
        radius = 1.0 / torch.sqrt(ks)

        direction = F.normalize(torch.randn(num_views, 3, device=device), dim=-1)
        origin = center.unsqueeze(0) + radius.unsqueeze(-1) * direction

        up = torch.tensor([0.0, 1.0, 0.0], device=device).expand(num_views, -1)
        parallel = (F.normalize(direction, dim=-1) * F.normalize(up[:1], dim=-1)).sum(
            -1
        ).abs() > 0.99
        if parallel.any():
            alt_up = torch.tensor([0.0, 0.0, 1.0], device=device).expand(
                parallel.sum(), -1
            )
            up = up.clone()
            up[parallel] = alt_up

        w2c_list: list[np.ndarray] = []
        for i in range(num_views):
            eye = origin[i].cpu().numpy()
            tgt = center_np.copy()
            u = up[i].cpu().numpy()
            w2c_list.append(_extrinsics_look_at(eye, tgt, u))

        extrinsics = torch.from_numpy(np.stack(w2c_list)).to(device)

        focal = 0.5 * self.resolution / math.tan(math.radians(fov / 2.0))
        intr = torch.tensor(
            [
                [focal / self.resolution, 0.0, 0.5],
                [0.0, focal / self.resolution, 0.5],
                [0.0, 0.0, 1.0],
            ],
            dtype=torch.float32,
            device=device,
        )
        intrinsics = intr.unsqueeze(0).expand(num_views, -1, -1)

        near = [max(float(r) - extent, 0.01) for r in radius.tolist()]
        far = [float(r) + extent for r in radius.tolist()]
        names = [f"rand_{i:03d}" for i in range(num_views)]

        return CameraParams(
            extrinsics=extrinsics,
            intrinsics=intrinsics,
            near=near,
            far=far,
            names=names,
        )

    # ------------------------------------------------------------------
    # Core single-view render
    # ------------------------------------------------------------------

    def render(
        self,
        mesh: NvMesh,
        extrinsics: torch.Tensor,
        intrinsics: torch.Tensor,
        near: float,
        far: float,
        return_types: Sequence[str] = ("base_color", "normal"),
        resolution: Optional[int] = None,
    ) -> Dict[str, torch.Tensor]:
        """Render a single view of *mesh*.

        Args:
            mesh: loaded :class:`NvMesh`.
            extrinsics: (4, 4) world-to-camera matrix.
            intrinsics: (3, 3) OpenCV intrinsics.
            near / far: clipping planes.
            return_types: subset of ``ALL_RETURN_TYPES``.
                ``"depth"`` is camera-space Z (positive, increasing away from
                camera, OpenCV convention) — not Euclidean distance.
                ``"normal"`` / ``"shading_normal"`` is the normal-map-perturbed
                shading normal encoded to ``[0, 1]`` (``n * 0.5 + 0.5``).
                ``"geo_normal"`` is the interpolated vertex normal (same encoding).
                ``"amr"`` packs alpha, metallic, roughness into ``(3, H, W)``.
                ``"layer_position"``, ``"layer_alpha"``, ``"layer_weight"``,
                and ``"layer_mask"`` expose per-depth-peel layer buffers.
            resolution: override ``self.resolution`` for this call.

        Returns:
            ``dict[str, Tensor]`` — each value is ``(C, H, W)`` or ``(H, W)``.
            All color outputs are in **linear** space.
        """
        res = resolution or self.resolution
        device = mesh.vertices.device
        height = width = res
        return_set = set(return_types)

        want_amr = "amr" in return_set
        if want_amr:
            return_set.discard("amr")
            return_set |= {"alpha", "metallic", "roughness"}

        perspective = _intrinsics_to_projection(intrinsics, near, far)
        full_proj = (perspective @ extrinsics).unsqueeze(0)

        vertices_b = mesh.vertices.unsqueeze(0)
        vertices_homo = torch.cat(
            [vertices_b, torch.ones_like(vertices_b[..., :1])], dim=-1
        )
        vertices_clip = torch.bmm(vertices_homo, full_proj.transpose(-1, -2))

        # Camera position for backface detection
        w2c_np = extrinsics.cpu().numpy()
        R = w2c_np[:3, :3]
        t = w2c_np[:3, 3]
        cam_pos = torch.from_numpy((-R.T @ t).astype(np.float32)).to(device)

        # Vertex normals (smooth shading) — fallback to face normals if absent
        if mesh.normals is not None and mesh.normals.numel() > 0:
            vertex_normal_flat = mesh.normals.reshape(1, -1, 3).float().contiguous()
            normal_face_idx = (
                torch.arange(mesh.faces.shape[0] * 3, dtype=torch.int32, device=device)
                .reshape(-1, 3)
                .contiguous()
            )
        else:
            v0 = mesh.vertices[mesh.faces[:, 0]]
            v1 = mesh.vertices[mesh.faces[:, 1]]
            v2 = mesh.vertices[mesh.faces[:, 2]]
            face_normal = F.normalize(torch.cross(v1 - v0, v2 - v0, dim=1), dim=1)
            vertex_normal_flat = face_normal.unsqueeze(0).float().contiguous()
            normal_face_idx = (
                torch.arange(mesh.faces.shape[0], dtype=torch.int32, device=device)
                .unsqueeze(1)
                .repeat(1, 3)
                .contiguous()
            )

        mesh_uv_sets = mesh.uv_sets if mesh.uv_sets else {0: mesh.uv_coords}
        if 0 not in mesh_uv_sets:
            mesh_uv_sets = {**mesh_uv_sets, 0: mesh.uv_coords}
        uv_flat_by_set = {
            texcoord: uv.reshape(1, -1, 2) for texcoord, uv in mesh_uv_sets.items()
        }
        uv_face_idx_by_set = {
            texcoord: torch.arange(
                uv.shape[0] * 3, dtype=torch.int32, device=device
            ).reshape(-1, 3)
            for texcoord, uv in mesh_uv_sets.items()
        }
        tangent_flat = mesh.tangents.reshape(1, -1, 4)
        tangent_face_idx = torch.arange(
            mesh.tangents.shape[0] * 3, dtype=torch.int32, device=device
        ).reshape(-1, 3)

        has_vc = (
            mesh.face_vertex_colors is not None and mesh.vertex_color_mask is not None
        )
        if has_vc:
            vc_flat = mesh.face_vertex_colors.reshape(1, -1, 4)
            vc_face_idx = torch.arange(
                mesh.face_vertex_colors.shape[0] * 3,
                dtype=torch.int32,
                device=device,
            ).reshape(-1, 3)

        need_pbr = return_set & {
            "base_color",
            "metallic",
            "roughness",
            "alpha",
            "emissive",
        }

        accum: dict[str, torch.Tensor] = {}
        if "base_color" in return_set:
            accum["base_color"] = torch.zeros(
                (height, width, 3), dtype=torch.float32, device=device
            )
        if "metallic" in return_set:
            accum["metallic"] = torch.zeros(
                (height, width, 1), dtype=torch.float32, device=device
            )
        if "roughness" in return_set:
            accum["roughness"] = torch.zeros(
                (height, width, 1), dtype=torch.float32, device=device
            )
        if "emissive" in return_set:
            accum["emissive"] = torch.zeros(
                (height, width, 3), dtype=torch.float32, device=device
            )
        if "normal" in return_set:
            accum["normal"] = torch.zeros(
                (height, width, 3), dtype=torch.float32, device=device
            )
        if "geo_normal" in return_set:
            accum["geo_normal"] = torch.zeros(
                (height, width, 3), dtype=torch.float32, device=device
            )
        if "shading_normal" in return_set:
            accum["shading_normal"] = torch.zeros(
                (height, width, 3), dtype=torch.float32, device=device
            )
        if "occlusion" in return_set:
            accum["occlusion"] = torch.ones(
                (height, width, 1), dtype=torch.float32, device=device
            )
        if "depth" in return_set:
            accum["depth"] = torch.zeros(
                (height, width, 1), dtype=torch.float32, device=device
            )
        if "position" in return_set:
            accum["position"] = torch.zeros(
                (height, width, 3), dtype=torch.float32, device=device
            )

        accum_alpha_total = torch.zeros(
            (height, width, 1), dtype=torch.float32, device=device
        )
        accum_mask = torch.zeros((height, width, 1), dtype=torch.float32, device=device)
        want_layer_buffers = bool(
            return_set & {"layer_position", "layer_alpha", "layer_weight", "layer_mask"}
        )
        layer_positions: list[torch.Tensor] = []
        layer_alphas: list[torch.Tensor] = []
        layer_weights: list[torch.Tensor] = []
        layer_masks: list[torch.Tensor] = []

        with dr.DepthPeeler(
            self._glctx, vertices_clip, mesh.faces, (height, width)
        ) as peeler:
            for _ in range(self.peel_layers):
                rast, rast_db = peeler.rasterize_next_layer()
                tri_id = rast[0, :, :, -1:]
                mask = (tri_id > 0).float()
                if mask.sum() == 0:
                    break

                pos = dr.interpolate(vertices_b, rast, mesh.faces)[0][0]
                gb_normal = dr.interpolate(
                    vertex_normal_flat,
                    rast,
                    normal_face_idx,
                )[0][0]

                view_dir = pos - cam_pos.reshape(1, 1, 3)
                is_backface = (gb_normal * view_dir).sum(dim=-1, keepdim=True) > 0
                gb_normal = torch.where(is_backface, -gb_normal, gb_normal)

                tex_cache = {}

                def raster_uv(texcoord: int) -> tuple[torch.Tensor, torch.Tensor]:
                    if texcoord not in tex_cache:
                        if texcoord not in uv_flat_by_set:
                            raise RuntimeError(
                                f"Mesh does not contain TEXCOORD_{texcoord}"
                            )
                        texc_i, texd_i = dr.interpolate(
                            uv_flat_by_set[texcoord],
                            rast,
                            uv_face_idx_by_set[texcoord],
                            rast_db=rast_db,
                            diff_attrs="all",
                        )
                        texc_i = torch.nan_to_num(
                            texc_i, nan=0.0, posinf=1e3, neginf=-1e3
                        ).clamp(-1e3, 1e3)
                        texd_i = torch.nan_to_num(
                            texd_i, nan=0.0, posinf=1e3, neginf=-1e3
                        ).clamp(-1e3, 1e3)
                        tex_cache[texcoord] = (texc_i, texd_i)
                    return tex_cache[texcoord]

                texc, texd = raster_uv(0)

                mid = mesh.material_ids[(tri_id.long() - 1).squeeze(-1)]
                tangent4 = dr.interpolate(tangent_flat, rast, tangent_face_idx)[0][0]

                gb_basecolor = torch.zeros(
                    (height, width, 3), dtype=torch.float32, device=device
                )
                gb_metallic = torch.zeros(
                    (height, width, 1), dtype=torch.float32, device=device
                )
                gb_roughness = torch.zeros(
                    (height, width, 1), dtype=torch.float32, device=device
                )
                gb_emissive = torch.zeros(
                    (height, width, 3), dtype=torch.float32, device=device
                )
                gb_alpha = torch.zeros(
                    (height, width, 1), dtype=torch.float32, device=device
                )
                gb_occlusion = torch.ones(
                    (height, width, 1), dtype=torch.float32, device=device
                )

                vc_interp = None
                vc_pixel_mask = None
                if has_vc:
                    vc_interp = dr.interpolate(vc_flat, rast, vc_face_idx)[0][0]
                    face_ids_hw = (tri_id.long() - 1).squeeze(-1)
                    vc_pixel_mask = (
                        (
                            (tri_id.squeeze(-1) > 0)
                            & mesh.vertex_color_mask[face_ids_hw.clamp(min=0)]
                        )
                        .unsqueeze(-1)
                        .float()
                    )

                for mat_idx, mat in enumerate(mesh.materials):
                    mat_mask = ((mid == mat_idx).float() * mask.squeeze(-1)).unsqueeze(
                        -1
                    )
                    if not mat.double_sided:
                        mat_mask = mat_mask * (~is_backface).float()

                    # Base color
                    if need_pbr or mat.is_unlit:
                        factor = mat.base_color_factor.to(device)
                        if mat.base_color_texture is not None:
                            mat_texc, mat_texd = raster_uv(mat.base_color_texcoord)
                            mat_texc = mat_texc * mat_mask
                            mat_texd = mat_texd * mat_mask
                            bc_texc, bc_boundary = _prepare_texture_uv(
                                mat_texc,
                                mat.base_color_boundary_mode,
                                offset=mat.base_color_uv_offset,
                                scale=mat.base_color_uv_scale,
                                rotation=mat.base_color_uv_rotation,
                            )
                            bc = dr.texture(
                                mat.base_color_texture.unsqueeze(0),
                                bc_texc,
                                mat_texd,
                                filter_mode=mat.base_color_filter_mode,
                                boundary_mode=bc_boundary,
                            )[0]
                            bc_result = srgb_to_linear(bc) * factor[:3]
                        else:
                            bc_result = factor[:3].unsqueeze(0).unsqueeze(0)

                        if vc_interp is not None and vc_pixel_mask is not None:
                            vc_rgb = torch.lerp(
                                torch.ones_like(vc_interp[..., :3]),
                                vc_interp[..., :3],
                                vc_pixel_mask,
                            )
                        else:
                            vc_rgb = 1.0
                        gb_basecolor += bc_result * vc_rgb * mat_mask

                        # Metallic
                        if mat.metallic_texture is not None:
                            mat_texc, mat_texd = raster_uv(mat.metallic_texcoord)
                            mat_texc = mat_texc * mat_mask
                            mat_texd = mat_texd * mat_mask
                            m_texc, m_boundary = _prepare_texture_uv(
                                mat_texc,
                                mat.metallic_boundary_mode,
                                offset=mat.metallic_uv_offset,
                                scale=mat.metallic_uv_scale,
                                rotation=mat.metallic_uv_rotation,
                            )
                            m_tex = dr.texture(
                                mat.metallic_texture.unsqueeze(0),
                                m_texc,
                                mat_texd,
                                filter_mode=mat.metallic_filter_mode,
                                boundary_mode=m_boundary,
                            )[0]
                            gb_metallic += m_tex * mat.metallic_factor * mat_mask
                        else:
                            gb_metallic += mat.metallic_factor * mat_mask

                        # Roughness
                        if mat.roughness_texture is not None:
                            mat_texc, mat_texd = raster_uv(mat.roughness_texcoord)
                            mat_texc = mat_texc * mat_mask
                            mat_texd = mat_texd * mat_mask
                            r_texc, r_boundary = _prepare_texture_uv(
                                mat_texc,
                                mat.roughness_boundary_mode,
                                offset=mat.roughness_uv_offset,
                                scale=mat.roughness_uv_scale,
                                rotation=mat.roughness_uv_rotation,
                            )
                            r_tex = dr.texture(
                                mat.roughness_texture.unsqueeze(0),
                                r_texc,
                                mat_texd,
                                filter_mode=mat.roughness_filter_mode,
                                boundary_mode=r_boundary,
                            )[0]
                            gb_roughness += r_tex * mat.roughness_factor * mat_mask
                        else:
                            gb_roughness += mat.roughness_factor * mat_mask

                        # Emissive
                        if mat.emissive_texture is not None:
                            mat_texc, mat_texd = raster_uv(mat.emissive_texcoord)
                            mat_texc = mat_texc * mat_mask
                            mat_texd = mat_texd * mat_mask
                            em_texc, em_boundary = _prepare_texture_uv(
                                mat_texc,
                                mat.emissive_boundary_mode,
                                offset=mat.emissive_uv_offset,
                                scale=mat.emissive_uv_scale,
                                rotation=mat.emissive_uv_rotation,
                            )
                            em = dr.texture(
                                mat.emissive_texture.unsqueeze(0),
                                em_texc,
                                mat_texd,
                                filter_mode=mat.emissive_filter_mode,
                                boundary_mode=em_boundary,
                            )[0]
                            gb_emissive += (
                                srgb_to_linear(em)
                                * mat.emissive_factor.to(device)
                                * mat_mask
                            )
                        else:
                            gb_emissive += (
                                mat.emissive_factor.to(device).unsqueeze(0).unsqueeze(0)
                                * mat_mask
                            )

                        # KHR_materials_unlit: zero metallic, full roughness, no specular
                        if mat.is_unlit:
                            gb_metallic = gb_metallic * (1.0 - mat_mask)
                            gb_roughness = gb_roughness * (1.0 - mat_mask) + mat_mask

                        # Occlusion (R channel, linear, not sRGB)
                        if mat.occlusion_texture is not None:
                            mat_texc, mat_texd = raster_uv(mat.occlusion_texcoord)
                            mat_texc = mat_texc * mat_mask
                            mat_texd = mat_texd * mat_mask
                            occ_texc, occ_boundary = _prepare_texture_uv(
                                mat_texc,
                                mat.occlusion_boundary_mode,
                                offset=mat.occlusion_uv_offset,
                                scale=mat.occlusion_uv_scale,
                                rotation=mat.occlusion_uv_rotation,
                            )
                            occ = dr.texture(
                                mat.occlusion_texture.unsqueeze(0),
                                occ_texc,
                                mat_texd,
                                filter_mode=mat.occlusion_filter_mode,
                                boundary_mode=occ_boundary,
                            )[0]
                            # strength lerps between no-occlusion (1.0) and full occlusion
                            occ_val = 1.0 + mat.occlusion_strength * (occ - 1.0)
                            gb_occlusion = (
                                gb_occlusion * (1.0 - mat_mask) + occ_val * mat_mask
                            )

                    # Alpha (always needed for compositing). glTF combines
                    # baseColorFactor.a, baseColorTexture.a, and COLOR_0.a
                    # before applying alphaMode.
                    factor = mat.base_color_factor.to(device)
                    alpha_raw = factor[3] * mat_mask
                    if mat.alpha_texture is not None:
                        mat_texc, mat_texd = raster_uv(mat.alpha_texcoord)
                        mat_texc = mat_texc * mat_mask
                        mat_texd = mat_texd * mat_mask
                        a_texc, a_boundary = _prepare_texture_uv(
                            mat_texc,
                            mat.alpha_boundary_mode,
                            offset=mat.alpha_uv_offset,
                            scale=mat.alpha_uv_scale,
                            rotation=mat.alpha_uv_rotation,
                        )
                        a = dr.texture(
                            mat.alpha_texture.unsqueeze(0),
                            a_texc,
                            mat_texd,
                            filter_mode=mat.alpha_filter_mode,
                            boundary_mode=a_boundary,
                        )[0]
                        alpha_raw = alpha_raw * a
                    if vc_interp is not None and vc_pixel_mask is not None:
                        vc_alpha = torch.lerp(
                            torch.ones_like(vc_interp[..., 3:4]),
                            vc_interp[..., 3:4],
                            vc_pixel_mask,
                        )
                        alpha_raw = alpha_raw * vc_alpha

                    if mat.alpha_mode == ALPHA_OPAQUE:
                        effective = 1.0 * mat_mask
                    elif mat.alpha_mode == ALPHA_MASK:
                        effective = (alpha_raw >= mat.alpha_cutoff).float() * mat_mask
                    else:  # ALPHA_BLEND
                        effective = alpha_raw
                    if mat.transmission_factor > 0.0:
                        # KHR_materials_transmission: attenuate alpha by transmission
                        effective = effective * (1.0 - mat.transmission_factor)
                    gb_alpha += effective

                # Capture geometric (interpolated vertex) normal before perturbation.
                gb_geo_normal = F.normalize(gb_normal, dim=-1)

                # Normal-map perturbation with UV-derived tangent frame.
                if "normal" in return_set or "shading_normal" in return_set:
                    n_geo = F.normalize(gb_normal, dim=-1)
                    T = F.normalize(
                        tangent4[..., :3]
                        - n_geo * (tangent4[..., :3] * n_geo).sum(dim=-1, keepdim=True),
                        dim=-1,
                    )
                    tangent_sign = torch.where(
                        tangent4[..., 3:4] < 0.0,
                        torch.full_like(tangent4[..., 3:4], -1.0),
                        torch.ones_like(tangent4[..., 3:4]),
                    )
                    B = F.normalize(
                        torch.cross(n_geo, T, dim=-1) * tangent_sign, dim=-1
                    )

                    for mat_idx2, mat2 in enumerate(mesh.materials):
                        if mat2.normal_texture is None or mat2.is_unlit:
                            continue
                        mat_mask2 = (
                            (mid == mat_idx2).float() * mask.squeeze(-1)
                        ).unsqueeze(-1)
                        if mat_mask2.sum() == 0:
                            continue
                        nm_texc0, nm_texd0 = raster_uv(mat2.normal_texcoord)
                        nm_texc, nm_boundary = _prepare_texture_uv(
                            nm_texc0 * mat_mask2,
                            mat2.normal_boundary_mode,
                            offset=mat2.normal_uv_offset,
                            scale=mat2.normal_uv_scale,
                            rotation=mat2.normal_uv_rotation,
                        )
                        nm = dr.texture(
                            mat2.normal_texture.unsqueeze(0),
                            nm_texc,
                            nm_texd0 * mat_mask2,
                            filter_mode=mat2.normal_filter_mode,
                            boundary_mode=nm_boundary,
                        )[0]
                        nm_ts = nm * 2.0 - 1.0
                        if mat2.normal_scale != 1.0:
                            nm_ts = torch.cat(
                                [nm_ts[..., :2] * mat2.normal_scale, nm_ts[..., 2:3]],
                                dim=-1,
                            )
                        perturbed = F.normalize(
                            T * nm_ts[..., 0:1]
                            + B * nm_ts[..., 1:2]
                            + n_geo * nm_ts[..., 2:3],
                            dim=-1,
                        )
                        has_perturbation = (nm_ts[..., :2].abs() > 0.01).any(
                            dim=-1, keepdim=True
                        )
                        gb_normal = torch.where(
                            (mat_mask2 > 0) & has_perturbation,
                            perturbed,
                            gb_normal,
                        )

                # Front-to-back alpha compositing
                w = (1.0 - accum_alpha_total) * gb_alpha
                if want_layer_buffers:
                    layer_positions.append(pos.permute(2, 0, 1))
                    layer_alphas.append(gb_alpha.squeeze(-1))
                    layer_weights.append(w.squeeze(-1))
                    layer_masks.append(mask.squeeze(-1))
                if "base_color" in accum:
                    accum["base_color"] += w * gb_basecolor
                if "metallic" in accum:
                    accum["metallic"] += w * gb_metallic
                if "roughness" in accum:
                    accum["roughness"] += w * gb_roughness
                if "emissive" in accum:
                    accum["emissive"] += w * gb_emissive
                normal_enc = gb_normal * 0.5 + 0.5
                geo_normal_enc = gb_geo_normal * 0.5 + 0.5
                if "normal" in accum:
                    accum["normal"] += w * normal_enc
                if "shading_normal" in accum:
                    accum["shading_normal"] += w * normal_enc
                if "geo_normal" in accum:
                    accum["geo_normal"] += w * geo_normal_enc
                if "occlusion" in accum:
                    accum["occlusion"] = (
                        accum["occlusion"] * (1.0 - w) + w * gb_occlusion
                    )
                if "depth" in accum:
                    vertices_camera = torch.bmm(
                        vertices_homo,
                        extrinsics.unsqueeze(0).transpose(-1, -2),
                    )
                    depth_interp = dr.interpolate(
                        vertices_camera[..., 2:3],
                        rast,
                        mesh.faces,
                    )[0][0]
                    accum["depth"] += w * depth_interp
                if "position" in accum:
                    accum["position"] += w * pos

                accum_alpha_total += w
                accum_mask = torch.maximum(accum_mask, mask)

        # Assemble output dict — (H, W, C) → (C, H, W)
        out: Dict[str, torch.Tensor] = {}
        for key in return_set:
            if key == "mask":
                out["mask"] = accum_mask.squeeze(-1)  # (H, W)
            elif key == "alpha":
                out["alpha"] = accum_alpha_total.squeeze(-1)  # (H, W)
            elif key in accum:
                t = accum[key]
                if t.shape[-1] == 1:
                    out[key] = t.squeeze(-1)  # (H, W)
                else:
                    out[key] = t.permute(2, 0, 1)  # (C, H, W)

        if "layer_position" in return_set:
            if layer_positions:
                out["layer_position"] = torch.stack(layer_positions, dim=0)
            else:
                out["layer_position"] = torch.empty(
                    0, 3, height, width, dtype=torch.float32, device=device
                )
        if "layer_alpha" in return_set:
            if layer_alphas:
                out["layer_alpha"] = torch.stack(layer_alphas, dim=0)
            else:
                out["layer_alpha"] = torch.empty(
                    0, height, width, dtype=torch.float32, device=device
                )
        if "layer_weight" in return_set:
            if layer_weights:
                out["layer_weight"] = torch.stack(layer_weights, dim=0)
            else:
                out["layer_weight"] = torch.empty(
                    0, height, width, dtype=torch.float32, device=device
                )
        if "layer_mask" in return_set:
            if layer_masks:
                out["layer_mask"] = torch.stack(layer_masks, dim=0)
            else:
                out["layer_mask"] = torch.empty(
                    0, height, width, dtype=torch.float32, device=device
                )

        if want_amr:
            a = out.get("alpha", accum_alpha_total.squeeze(-1))
            m = out.get("metallic", torch.zeros(height, width, device=device))
            r = out.get("roughness", torch.zeros(height, width, device=device))
            out["amr"] = torch.stack(
                [a, r, m], dim=0
            )  # (3, H, W) — match mesh_dist [alpha, roughness, metallic]

        return out

    # ------------------------------------------------------------------
    # Multi-view render → tensor list
    # ------------------------------------------------------------------

    def render_views(
        self,
        mesh: NvMesh,
        cameras: CameraParams,
        return_types: Sequence[str] = ("base_color", "normal"),
        resolution: Optional[int] = None,
    ) -> List[Dict[str, torch.Tensor]]:
        """Render multiple views, returning a list of per-view dicts."""
        results: list[dict[str, torch.Tensor]] = []
        for i in range(len(cameras)):
            out = self.render(
                mesh,
                cameras.extrinsics[i],
                cameras.intrinsics[i],
                cameras.near[i],
                cameras.far[i],
                return_types=return_types,
                resolution=resolution,
            )
            results.append(out)
        return results

    # ------------------------------------------------------------------
    # Grid render → PIL Images
    # ------------------------------------------------------------------

    def render_grid(
        self,
        mesh: NvMesh,
        cameras: Optional[CameraParams] = None,
        view_size: Optional[int] = None,
        cols: int = 3,
        with_amr: bool = False,
        with_geo_normal: bool = False,
        with_occlusion: bool = False,
    ) -> Dict[str, Image.Image]:
        """Render views and tile into grid images per channel.

        Args:
            mesh: loaded :class:`NvMesh`.
            cameras: camera parameters; defaults to 6-view layout.
            view_size: per-view resolution; defaults to ``self.resolution``.
            cols: number of columns in the grid.
            with_amr: include ``"amr"`` grid (alpha/metallic/roughness).
            with_geo_normal: include ``"geo_normal"`` grid.
            with_occlusion: include ``"occlusion"`` grid.

        Returns:
            Dict with keys ``"base_color"`` and ``"shading_normal"`` always
            present, plus ``"amr"``, ``"geo_normal"``, ``"occlusion"`` when
            the corresponding flag is set. All images are sRGB uint8 PIL Images
            (base_color is tonemapped; others are linear-encoded to [0, 255]).
        """
        res = view_size or self.resolution
        if cameras is None:
            old_res = self.resolution
            self.resolution = res
            cameras = self.build_6_views(mesh)
            self.resolution = old_res

        rt: list[str] = ["base_color", "shading_normal"]
        if with_amr:
            rt.append("amr")
        if with_geo_normal:
            rt.append("geo_normal")
        if with_occlusion:
            rt.append("occlusion")

        view_dicts = self.render_views(
            mesh,
            cameras,
            return_types=tuple(rt),
            resolution=res,
        )

        n = len(view_dicts)
        rows = math.ceil(n / cols)

        def _blank() -> Image.Image:
            return Image.new("RGB", (res * cols, res * rows), (0, 0, 0))

        canvases: Dict[str, Image.Image] = {
            "base_color": _blank(),
            "shading_normal": _blank(),
        }
        if with_amr:
            canvases["amr"] = _blank()
        if with_geo_normal:
            canvases["geo_normal"] = _blank()
        if with_occlusion:
            canvases["occlusion"] = _blank()

        for i, vd in enumerate(view_dicts):
            r, c = i // cols, i % cols
            x, y = c * res, r * res

            bc = linear_to_srgb(vd["base_color"].permute(1, 2, 0)).clamp(0, 1)
            canvases["base_color"].paste(
                Image.fromarray((bc.cpu().numpy() * 255.0 + 0.5).astype(np.uint8)),
                (x, y),
            )
            nrm = vd["shading_normal"].permute(1, 2, 0).clamp(0, 1)
            canvases["shading_normal"].paste(
                Image.fromarray((nrm.cpu().numpy() * 255.0 + 0.5).astype(np.uint8)),
                (x, y),
            )
            if with_amr:
                amr = vd["amr"].permute(1, 2, 0).clamp(0, 1)
                canvases["amr"].paste(
                    Image.fromarray((amr.cpu().numpy() * 255.0 + 0.5).astype(np.uint8)),
                    (x, y),
                )
            if with_geo_normal:
                gn = vd["geo_normal"].permute(1, 2, 0).clamp(0, 1)
                canvases["geo_normal"].paste(
                    Image.fromarray((gn.cpu().numpy() * 255.0 + 0.5).astype(np.uint8)),
                    (x, y),
                )
            if with_occlusion:
                occ = (
                    vd["occlusion"]
                    .unsqueeze(0)
                    .expand(3, -1, -1)
                    .permute(1, 2, 0)
                    .clamp(0, 1)
                )
                canvases["occlusion"].paste(
                    Image.fromarray((occ.cpu().numpy() * 255.0 + 0.5).astype(np.uint8)),
                    (x, y),
                )

        return canvases


# ---------------------------------------------------------------------------
# Backward-compatible free functions
# ---------------------------------------------------------------------------


def render_views(
    mesh: NvMesh,
    view_size: int = VIEW_SIZE,
    peel_layers: int = DEFAULT_PEEL_LAYERS,
    glctx: Optional[object] = None,
) -> tuple[Image.Image, Image.Image]:
    """Render 6-view basecolor and shading_normal grid images (legacy API).

    Prefer :class:`NvMeshRenderer` for new code.
    """
    renderer = NvMeshRenderer(
        device=str(mesh.vertices.device),
        resolution=view_size,
        peel_layers=peel_layers,
    )
    if glctx is not None:
        renderer._glctx = glctx  # type: ignore[assignment]
    imgs = renderer.render_grid(mesh, view_size=view_size)
    return imgs["base_color"], imgs["shading_normal"]


def render_views_from_glb(
    glb_path: Union[str, Path],
    view_size: int = VIEW_SIZE,
    peel_layers: int = DEFAULT_PEEL_LAYERS,
    device: str = "cuda",
    glctx: Optional[object] = None,
    missing_uv_policy: str = "blender",
) -> tuple[Image.Image, Image.Image]:
    """Load a GLB and render 6-view basecolor + shading_normal (legacy API)."""
    mesh = load_glb(glb_path, device=device, missing_uv_policy=missing_uv_policy)
    return render_views(mesh, view_size=view_size, peel_layers=peel_layers, glctx=glctx)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(
        description="Render views from a GLB using nvdiffrast.",
    )
    parser.add_argument("--glb", type=str, required=True, help="Path to GLB file")
    parser.add_argument("--view-size", type=int, default=VIEW_SIZE)
    parser.add_argument("--out", type=str, default=None, help="Output directory")
    parser.add_argument("--peel-layers", type=int, default=DEFAULT_PEEL_LAYERS)
    parser.add_argument(
        "--missing-uv-policy",
        choices=("blender", "strict", "error"),
        default="blender",
        help="How to handle textured primitives missing required TEXCOORD_n.",
    )
    parser.add_argument(
        "--random-views",
        type=int,
        default=0,
        help="Number of random views (0 = use 6-view grid)",
    )
    args = parser.parse_args()

    glb_path = Path(args.glb)
    if not glb_path.exists():
        raise FileNotFoundError(f"Model not found: {glb_path}")

    out_dir = Path(args.out) if args.out else Path(".")
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = "".join(c if c.isalnum() or c in "._-" else "_" for c in glb_path.stem)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available")

    print(f"Loading GLB: {glb_path}")
    mesh = load_glb(
        glb_path,
        device="cuda",
        missing_uv_policy=args.missing_uv_policy,
    )
    print(
        f"  vertices: {mesh.vertices.shape[0]}, "
        f"faces: {mesh.faces.shape[0]}, "
        f"materials: {len(mesh.materials)}"
    )

    renderer = NvMeshRenderer(
        device="cuda",
        resolution=args.view_size,
        peel_layers=args.peel_layers,
    )

    if args.random_views > 0:
        cameras = renderer.randomize_cameras(mesh, num_views=args.random_views)
        print(
            f"Rendering {args.random_views} random views at {args.view_size}x{args.view_size}..."
        )
        views = renderer.render_views(
            mesh, cameras, return_types=("base_color", "normal")
        )
        for i, vd in enumerate(views):
            bc = linear_to_srgb(vd["base_color"].permute(1, 2, 0)).clamp(0, 1)
            bc_np = (bc.cpu().numpy() * 255.0 + 0.5).astype(np.uint8)
            p = out_dir / f"{stem}_rand{i:03d}_basecolor.png"
            Image.fromarray(bc_np).save(p)
            print(f"  Saved: {p}")
    else:
        print(
            f"Rendering 6 views at {args.view_size}x{args.view_size} "
            f"with {args.peel_layers} peel layers..."
        )
        imgs = renderer.render_grid(mesh, view_size=args.view_size)
        out_views = out_dir / f"{stem}_views_6_basecolor.png"
        out_normals = out_dir / f"{stem}_views_6_normals.png"
        imgs["base_color"].save(out_views)
        imgs["shading_normal"].save(out_normals)
        print(f"Saved: {out_views}")
        print(f"Saved: {out_normals}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
