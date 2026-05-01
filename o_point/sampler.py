"""Surface sampler using pure PyTorch + nvdiffrast dr.texture."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, Literal, Optional, Tuple, Union

if TYPE_CHECKING:
    from PIL import Image

import numpy as np
import nvdiffrast.torch as dr
import torch
import torch.nn.functional as F
import trimesh

from .materials import (
    ALPHA_BLEND,
    ALPHA_MASK,
    ALPHA_OPAQUE,
    NvMesh,
    NvPbrMaterial,
    _prepare_texture_uv,
    linear_to_srgb,
    load_glb,
    srgb_to_linear,
)

__all__ = [
    "BatchProcessor",
    "MeshTextureSampler",
    "textured_mesh_to_surface_samples",
]


# ---------------------------------------------------------------------------
# dr.texture wrapper
# ---------------------------------------------------------------------------


def _query_texture(
    tex: torch.Tensor,
    uv: torch.Tensor,
    filter_mode: str = "linear-mipmap-linear",
    boundary_mode: tuple[str, str] | str = ("wrap", "wrap"),
    uv_offset: tuple[float, float] = (0.0, 0.0),
    uv_scale: tuple[float, float] = (1.0, 1.0),
    uv_rotation: float = 0.0,
) -> torch.Tensor:
    """Sample a texture at arbitrary UV coordinates using nvdiffrast.

    Args:
        tex: (H, W, C) float32 texture on GPU.
        uv: (K, 2) float32 UV coordinates in [0, 1].
        filter_mode: nvdiffrast filter mode.

    Returns:
        (K, C) sampled values.
    """
    uv, native_boundary_mode = _prepare_texture_uv(
        uv,
        boundary_mode,
        offset=uv_offset,
        scale=uv_scale,
        rotation=uv_rotation,
    )
    uv_4d = uv.reshape(1, 1, -1, 2)
    tex_4d = tex.unsqueeze(0)  # (1, H, W, C)
    if "mipmap" in filter_mode:
        # Point queries have no screen-space UV derivatives; use mip_level_bias=0
        # to sample the finest mip level.
        mip_bias = torch.zeros(1, 1, uv.shape[0], device=uv.device, dtype=uv.dtype)
        sampled = dr.texture(
            tex_4d,
            uv_4d,
            mip_level_bias=mip_bias,
            filter_mode=filter_mode,
            boundary_mode=native_boundary_mode,
        )
    else:
        sampled = dr.texture(
            tex_4d,
            uv_4d,
            filter_mode=filter_mode,
            boundary_mode=native_boundary_mode,
        )
    return sampled[0, 0]  # (K, C)


# ---------------------------------------------------------------------------
# Sampler
# ---------------------------------------------------------------------------


class MeshTextureSampler:
    """GPU surface sampler backed by ``nvdiffrast.dr.texture``.

    Loads a GLB (or accepts a pre-loaded :class:`NvMesh`), provides:

    * ``sample(num_samples)`` — uniform surface sampling + attribute query
    * ``sample_from_face_bary(face_ids, bary)`` — attribute query only
    """

    def __init__(
        self,
        mesh: Union[str, trimesh.Scene, trimesh.Trimesh, NvMesh],
        device: Optional[Union[torch.device, str]] = None,
        missing_uv_policy: str = "blender",
    ) -> None:
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        if isinstance(mesh, NvMesh):
            self._mesh = mesh
        elif isinstance(mesh, str):
            self._mesh = load_glb(
                mesh,
                device=str(self.device),
                missing_uv_policy=missing_uv_policy,
            )
        elif isinstance(mesh, (trimesh.Scene, trimesh.Trimesh)):
            self._mesh = self._from_trimesh(mesh, missing_uv_policy=missing_uv_policy)
        else:
            raise TypeError(f"Unsupported mesh type: {type(mesh)}")

        self._face_areas: Optional[torch.Tensor] = None
        self._cdf_cache: dict[tuple, torch.Tensor] = {}

    # ------------------------------------------------------------------
    # trimesh → NvMesh conversion (for API compatibility)
    # ------------------------------------------------------------------

    def _from_trimesh(
        self,
        mesh: Union[trimesh.Scene, trimesh.Trimesh],
        missing_uv_policy: str = "blender",
    ) -> NvMesh:
        """Convert a trimesh object to NvMesh via a temporary GLB roundtrip."""
        import tempfile

        with tempfile.NamedTemporaryFile(suffix=".glb", delete=True) as tmp:
            if isinstance(mesh, trimesh.Scene):
                mesh.export(tmp.name, file_type="glb")
            else:
                trimesh.Scene(mesh).export(tmp.name, file_type="glb")
            return load_glb(
                tmp.name,
                device=str(self.device),
                missing_uv_policy=missing_uv_policy,
            )

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def triangles(self) -> torch.Tensor:
        return self._mesh.triangles

    @property
    def normals(self) -> torch.Tensor:
        return self._mesh.normals

    @property
    def tangents(self) -> torch.Tensor:
        return self._mesh.tangents

    @property
    def uvs(self) -> torch.Tensor:
        return self._mesh.uv_coords

    @property
    def uv_sets(self) -> dict[int, torch.Tensor]:
        return self._mesh.uv_sets

    @property
    def material_ids(self) -> torch.Tensor:
        return self._mesh.material_ids

    @property
    def materials(self) -> list[NvPbrMaterial]:
        return self._mesh.materials

    @property
    def mesh(self) -> NvMesh:
        return self._mesh

    # ------------------------------------------------------------------
    # GPU uniform sampling
    # ------------------------------------------------------------------

    def _ensure_face_areas(self) -> torch.Tensor:
        if self._face_areas is not None:
            return self._face_areas
        tris = self.triangles  # (F, 3, 3)
        e1 = tris[:, 1] - tris[:, 0]
        e2 = tris[:, 2] - tris[:, 0]
        self._face_areas = 0.5 * torch.cross(e1, e2, dim=1).norm(dim=1)
        return self._face_areas

    def _build_face_area_cdf(self, skip_non_double_sided: bool = False) -> torch.Tensor:
        """Build area-weighted CDF for uniform surface sampling."""
        cache_key = ("area", skip_non_double_sided)
        if cache_key in self._cdf_cache:
            return self._cdf_cache[cache_key]

        areas = self._ensure_face_areas().clone()
        if skip_non_double_sided:
            keep = torch.tensor(
                [mat.double_sided for mat in self.materials],
                dtype=torch.bool,
                device=self.device,
            )
            areas = torch.where(keep[self.material_ids.long()], areas, 0.0)

        cdf = torch.cumsum(areas, dim=0)
        self._cdf_cache[cache_key] = cdf
        return cdf

    def _sample_on_cdf(
        self,
        num_samples: int,
        cdf: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Draw *num_samples* points from a pre-built face CDF."""
        total = cdf[-1]
        if total <= 0:
            raise ValueError("Mesh has zero total area (or all faces filtered).")

        r = torch.rand(num_samples, device=self.device) * total
        face_idx = torch.searchsorted(cdf, r, right=False).to(torch.int32)

        r1 = torch.rand(num_samples, device=self.device)
        r2 = torch.rand(num_samples, device=self.device)
        sqrt_r1 = torch.sqrt(r1)
        w0 = 1.0 - sqrt_r1
        w1 = sqrt_r1 * (1.0 - r2)
        w2 = sqrt_r1 * r2
        bary = torch.stack([w0, w1, w2], dim=-1).to(torch.float32)
        bary = bary.clamp(0.0, 1.0)
        bary = bary / bary.sum(dim=-1, keepdim=True).clamp_min(1e-8)

        tris = self.triangles[face_idx]  # (N, 3, 3)
        points = (
            tris[:, 0] * w0[:, None]
            + tris[:, 1] * w1[:, None]
            + tris[:, 2] * w2[:, None]
        ).to(torch.float32)

        return points, face_idx, bary

    # ------------------------------------------------------------------
    # Two-stage texture importance sampling
    # ------------------------------------------------------------------

    @staticmethod
    def _build_knn_index(points: torch.Tensor, k: int) -> torch.Tensor:
        """Build KNN index via scipy cKDTree. Returns (N, k) int64 on device."""
        from scipy.spatial import cKDTree

        pts_np = points.detach().cpu().numpy().astype(np.float32)
        tree = cKDTree(pts_np)
        _, knn_idx = tree.query(pts_np, k=k + 1, workers=-1)
        knn_idx = knn_idx[:, 1:]  # (N, k) exclude self
        return torch.from_numpy(knn_idx.astype(np.int64)).to(points.device)

    @staticmethod
    def _knn_attr_variance(
        attrs: torch.Tensor,
        knn_idx: torch.Tensor,
    ) -> torch.Tensor:
        """Mean squared difference to KNN neighbors for any (N, C) attribute."""
        nbr = attrs[knn_idx]  # (N, k, C)
        center = attrs.unsqueeze(1)  # (N, 1, C)
        return (nbr - center).pow(2).mean(dim=(1, 2))

    def compute_importance_weights(
        self,
        points: torch.Tensor,
        colors: torch.Tensor,
        normals: torch.Tensor,
        knn_k: int = 16,
        strength: float = 1.0,
        pct_lo: float = 2.0,
        pct_hi: float = 98.0,
    ) -> Dict[str, torch.Tensor]:
        """Compute per-point importance weights and gradient diagnostics.

        Builds a single KNN index, then computes:
        - **texture gradient**: KNN color variance
        - **geometry gradient**: KNN normal variance (sharp edges/corners)
        - **importance weight**: combined, mean-normalized to 1.0

        Args:
            points: (N, 3) world-space positions.
            colors: (N, 3) base_color (linear recommended).
            normals: (N, 3) surface normals.
            knn_k: number of neighbors.
            strength: scaling factor for weight range.
            pct_lo: lower percentile clip.
            pct_hi: upper percentile clip.

        Returns:
            dict with keys:
            - ``importance_weight`` (N,) — combined weight, mean=1.0
            - ``texture_gradient`` (N,) — raw color KNN variance
            - ``geometry_gradient`` (N,) — raw normal KNN variance
            - ``is_geometry_sharp`` (N,) — bool, True if in top 10% normal gradient
        """
        knn_idx = self._build_knn_index(points, knn_k)

        tex_grad = self._knn_attr_variance(colors, knn_idx)
        geo_grad = self._knn_attr_variance(normals, knn_idx)

        def _normalize(g: torch.Tensor) -> torch.Tensor:
            lo = torch.quantile(g, pct_lo / 100.0)
            hi = torch.quantile(g, pct_hi / 100.0)
            clipped = g.clamp(lo, hi)
            return (clipped - lo) / (hi - lo).clamp(min=1e-12)

        tex_norm = _normalize(tex_grad)
        geo_norm = _normalize(geo_grad)
        combined = 1.0 + strength * (tex_norm + geo_norm) / 2.0
        combined = combined / combined.mean()

        geo_thresh = torch.quantile(geo_grad, 0.9)
        is_sharp = geo_grad >= geo_thresh

        return {
            "importance_weight": combined,
            "texture_gradient": tex_grad,
            "geometry_gradient": geo_grad,
            "is_geometry_sharp": is_sharp,
            "_knn_idx": knn_idx,
        }

    # ------------------------------------------------------------------
    # Attribute query via dr.texture
    # ------------------------------------------------------------------

    def sample_from_face_bary(
        self,
        points: torch.Tensor,
        face_ids: torch.Tensor,
        bary: torch.Tensor,
        tonemap_method: Literal["srgb", "linear"] = "srgb",
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Query all PBR attributes for given surface points.

        Args:
            points: (N, 3) world-space positions (returned as-is).
            face_ids: (N,) int32 face indices.
            bary: (N, 3) barycentric coordinates.
            tonemap_method: ``"srgb"`` applies linear-to-sRGB on base_color
                and emissive; ``"linear"`` keeps linear space.

        Returns:
            points: (N, 3)
            attrs: dict with keys ``base_color``, ``metallic``, ``roughness``,
                ``emissive``, ``alpha``, ``normal`` — all (N, C) float32.
        """
        face_ids_long = face_ids.long()
        N = face_ids_long.shape[0]
        dev = self.device

        # --- UV interpolation ---
        face_uvs = self.uvs[face_ids_long]  # (N, 3, 2)
        uv = (bary.unsqueeze(-1) * face_uvs).sum(dim=1)  # (N, 2)
        uv_cache = {0: uv}

        def uv_for(texcoord: int) -> torch.Tensor:
            if texcoord not in uv_cache:
                face_uv_set = self.uv_sets.get(texcoord)
                if face_uv_set is None:
                    raise RuntimeError(f"Mesh does not contain TEXCOORD_{texcoord}")
                uv_cache[texcoord] = (
                    bary.unsqueeze(-1) * face_uv_set[face_ids_long]
                ).sum(dim=1)
            return uv_cache[texcoord]

        # --- Normal interpolation ---
        face_n = self.normals[face_ids_long]  # (N, 3, 3)
        normal = (bary.unsqueeze(-1) * face_n).sum(dim=1)  # (N, 3)
        normal = F.normalize(normal, dim=-1)
        normal_interp = normal.clone()  # before normal-map perturbation
        face_t = self.tangents[face_ids_long]  # (N, 3, 4)
        tangent4 = (bary.unsqueeze(-1) * face_t).sum(dim=1)
        tangent = tangent4[:, :3]
        tangent = F.normalize(
            tangent - normal * (tangent * normal).sum(dim=-1, keepdim=True),
            dim=-1,
        )
        tangent_sign = torch.where(
            tangent4[:, 3:4] < 0.0,
            torch.full_like(tangent4[:, 3:4], -1.0),
            torch.ones_like(tangent4[:, 3:4]),
        )
        bitangent = F.normalize(
            torch.cross(normal, tangent, dim=-1) * tangent_sign,
            dim=-1,
        )

        # --- Per-material texture sampling ---
        mid = self.material_ids[face_ids_long]  # (N,)

        base_color = torch.zeros(N, 3, device=dev)
        metallic = torch.zeros(N, 1, device=dev)
        roughness = torch.zeros(N, 1, device=dev)
        emissive = torch.zeros(N, 3, device=dev)
        occlusion = torch.ones(N, 1, device=dev)
        alpha = torch.ones(N, 1, device=dev)
        alpha_raw_all = torch.ones(N, 1, device=dev)

        for mat_idx, mat in enumerate(self.materials):
            sel = mid == mat_idx
            if not sel.any():
                continue
            factor = mat.base_color_factor.to(dev)

            # Base color — glTF: texture is sRGB, factor/COLOR_0 are linear.
            if mat.base_color_texture is not None:
                uv_sel = uv_for(mat.base_color_texcoord)[sel]
                bc = srgb_to_linear(
                    _query_texture(
                        mat.base_color_texture,
                        uv_sel,
                        filter_mode=mat.base_color_filter_mode,
                        boundary_mode=mat.base_color_boundary_mode,
                        uv_offset=mat.base_color_uv_offset,
                        uv_scale=mat.base_color_uv_scale,
                        uv_rotation=mat.base_color_uv_rotation,
                    )
                )
                base_color[sel] = bc * factor[:3]
            else:
                base_color[sel] = factor[:3].unsqueeze(0)

            # Metallic
            if mat.metallic_texture is not None:
                uv_sel = uv_for(mat.metallic_texcoord)[sel]
                m = _query_texture(
                    mat.metallic_texture,
                    uv_sel,
                    filter_mode=mat.metallic_filter_mode,
                    boundary_mode=mat.metallic_boundary_mode,
                    uv_offset=mat.metallic_uv_offset,
                    uv_scale=mat.metallic_uv_scale,
                    uv_rotation=mat.metallic_uv_rotation,
                )
                metallic[sel] = m * mat.metallic_factor
            else:
                metallic[sel] = mat.metallic_factor

            # Roughness
            if mat.roughness_texture is not None:
                uv_sel = uv_for(mat.roughness_texcoord)[sel]
                r = _query_texture(
                    mat.roughness_texture,
                    uv_sel,
                    filter_mode=mat.roughness_filter_mode,
                    boundary_mode=mat.roughness_boundary_mode,
                    uv_offset=mat.roughness_uv_offset,
                    uv_scale=mat.roughness_uv_scale,
                    uv_rotation=mat.roughness_uv_rotation,
                )
                roughness[sel] = r * mat.roughness_factor
            else:
                roughness[sel] = mat.roughness_factor

            # Emissive — texture is sRGB, same treatment as base color.
            if mat.emissive_texture is not None:
                uv_sel = uv_for(mat.emissive_texcoord)[sel]
                em = srgb_to_linear(
                    _query_texture(
                        mat.emissive_texture,
                        uv_sel,
                        filter_mode=mat.emissive_filter_mode,
                        boundary_mode=mat.emissive_boundary_mode,
                        uv_offset=mat.emissive_uv_offset,
                        uv_scale=mat.emissive_uv_scale,
                        uv_rotation=mat.emissive_uv_rotation,
                    )
                )
                emissive[sel] = em * mat.emissive_factor.to(dev)
            else:
                emissive[sel] = mat.emissive_factor.to(dev).unsqueeze(0)

            # KHR_materials_unlit: zero metallic, full roughness
            if mat.is_unlit:
                metallic[sel] = 0.0
                roughness[sel] = 1.0

            # Occlusion (R channel, linear — glTF §5.19.3)
            if mat.occlusion_texture is not None:
                uv_sel = uv_for(mat.occlusion_texcoord)[sel]
                occ = _query_texture(
                    mat.occlusion_texture,
                    uv_sel,
                    filter_mode=mat.occlusion_filter_mode,
                    boundary_mode=mat.occlusion_boundary_mode,
                    uv_offset=mat.occlusion_uv_offset,
                    uv_scale=mat.occlusion_uv_scale,
                    uv_rotation=mat.occlusion_uv_rotation,
                )
                occlusion[sel] = 1.0 + mat.occlusion_strength * (occ - 1.0)

            # Alpha starts from baseColorFactor.a and baseColorTexture.a.
            alpha_raw = torch.full((int(sel.sum()), 1), factor[3].item(), device=dev)
            if mat.alpha_texture is not None:
                uv_sel = uv_for(mat.alpha_texcoord)[sel]
                a = _query_texture(
                    mat.alpha_texture,
                    uv_sel,
                    filter_mode=mat.alpha_filter_mode,
                    boundary_mode=mat.alpha_boundary_mode,
                    uv_offset=mat.alpha_uv_offset,
                    uv_scale=mat.alpha_uv_scale,
                    uv_rotation=mat.alpha_uv_rotation,
                )
                alpha_raw = alpha_raw * a
            alpha_raw_all[sel] = alpha_raw

            # Normal map perturbation: tangent-space -> world-space.
            if mat.normal_texture is not None and not mat.is_unlit:
                uv_sel = uv_for(mat.normal_texcoord)[sel]
                nm = _query_texture(
                    mat.normal_texture,
                    uv_sel,
                    filter_mode=mat.normal_filter_mode,
                    boundary_mode=mat.normal_boundary_mode,
                    uv_offset=mat.normal_uv_offset,
                    uv_scale=mat.normal_uv_scale,
                    uv_rotation=mat.normal_uv_rotation,
                )
                nm_tangent = nm * 2.0 - 1.0  # [0,1] -> [-1,1]
                if mat.normal_scale != 1.0:
                    nm_tangent = torch.cat(
                        [nm_tangent[..., :2] * mat.normal_scale, nm_tangent[..., 2:3]],
                        dim=-1,
                    )

                n_sel = normal[sel]  # (K, 3) interpolated world-space normal

                t_sel = tangent[sel]
                b_sel = bitangent[sel]
                perturbed = F.normalize(
                    t_sel * nm_tangent[..., 0:1]
                    + b_sel * nm_tangent[..., 1:2]
                    + n_sel * nm_tangent[..., 2:3],
                    dim=-1,
                )

                has_perturbation = (nm_tangent[..., :2].abs() > 0.01).any(dim=-1)
                normal[sel] = torch.where(
                    has_perturbation.unsqueeze(-1), perturbed, n_sel
                )

        # glTF COLOR_0 multiplies baseColorFactor/baseColorTexture.
        if (
            self._mesh.face_vertex_colors is not None
            and self._mesh.vertex_color_mask is not None
        ):
            vc_sel = self._mesh.vertex_color_mask[face_ids_long]
            if vc_sel.any():
                fvc = self._mesh.face_vertex_colors[face_ids_long[vc_sel]]  # (K, 3, 4)
                bary_vc = bary[vc_sel]  # (K, 3)
                interp_vc = (fvc * bary_vc.unsqueeze(-1)).sum(dim=1).clamp(0.0, 1.0)
                base_color[vc_sel] = base_color[vc_sel] * interp_vc[:, :3]
                alpha_raw_all[vc_sel] = alpha_raw_all[vc_sel] * interp_vc[:, 3:4]

        for mat_idx, mat in enumerate(self.materials):
            sel = mid == mat_idx
            if not sel.any():
                continue
            if mat.alpha_mode == ALPHA_OPAQUE:
                a_val = torch.ones(int(sel.sum()), 1, device=dev)
            elif mat.alpha_mode == ALPHA_MASK:
                a_val = (alpha_raw_all[sel] >= mat.alpha_cutoff).float()
            else:
                a_val = alpha_raw_all[sel]
            if mat.transmission_factor > 0.0:
                a_val = a_val * (1.0 - mat.transmission_factor)
            alpha[sel] = a_val

        # Tonemap
        if tonemap_method == "srgb":
            base_color = linear_to_srgb(base_color)
            emissive = linear_to_srgb(emissive)

        shading_n = normal.clamp(-1.0, 1.0)
        geo_n = normal_interp.clamp(-1.0, 1.0)
        attrs = {
            "base_color": base_color.clamp(0.0, 1.0),
            "metallic": metallic.clamp(0.0, 1.0),
            "roughness": roughness.clamp(0.0, 1.0),
            "emissive": emissive.clamp(0.0, 1.0),
            "occlusion": occlusion.clamp(0.0, 1.0),
            "alpha": alpha.clamp(0.0, 1.0),
            "shading_normal": shading_n,
            "geo_normal": geo_n,
            "normal": shading_n,  # backward-compat alias for shading_normal
            "normal_interp": geo_n,  # backward-compat alias for geo_normal
        }
        return points, attrs

    # ------------------------------------------------------------------
    # High-level API
    # ------------------------------------------------------------------

    def sample(
        self,
        num_samples: int = 100_000,
        tonemap_method: Literal["srgb", "linear"] = "srgb",
        skip_non_double_sided: bool = False,
        compute_importance: bool = False,
        knn_k: int = 16,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Sample points uniformly and query all PBR attributes.

        Args:
            num_samples: number of surface points to sample.
            tonemap_method: color space for base_color / emissive output.
            compute_importance: if True, compute per-point importance
                weight via KNN color gradient and include it in attrs
                as ``"importance_weight"`` (N, 1).
            knn_k: neighbors for gradient estimation (only used when
                *compute_importance* is True).

        Returns:
            points: (N, 3) world-space positions.
            attrs: dict of attribute tensors.  When *compute_importance*
                is True, includes ``"importance_weight"`` (N, 1).
        """
        cdf = self._build_face_area_cdf(skip_non_double_sided=skip_non_double_sided)
        points, face_ids, bary = self._sample_on_cdf(num_samples, cdf)
        points, attrs = self.sample_from_face_bary(
            points, face_ids, bary, tonemap_method=tonemap_method
        )
        if compute_importance:
            imp = self.compute_importance_weights(
                points,
                attrs["base_color"],
                attrs["normal_interp"],
                knn_k=knn_k,
            )
            attrs["importance_weight"] = imp["importance_weight"].unsqueeze(-1)
            attrs["texture_gradient"] = imp["texture_gradient"].unsqueeze(-1)
            attrs["geometry_gradient"] = imp["geometry_gradient"].unsqueeze(-1)
            attrs["is_geometry_sharp"] = imp["is_geometry_sharp"].unsqueeze(-1).float()
            attrs["geometry_gradient_nm"] = self._knn_attr_variance(
                attrs["normal"],
                imp["_knn_idx"],
            ).unsqueeze(-1)
        return points, attrs

    def sample_points_with_bary(
        self,
        num_samples: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample surface points and return raw (points, face_ids, bary)."""
        cdf = self._build_face_area_cdf()
        return self._sample_on_cdf(num_samples, cdf)


# ---------------------------------------------------------------------------
# Convenience function
# ---------------------------------------------------------------------------


def textured_mesh_to_surface_samples(
    mesh: Union[str, trimesh.Scene, trimesh.Trimesh, NvMesh],
    num_samples: int = 100_000,
    tonemap_method: Literal["srgb", "linear"] = "srgb",
    device: Optional[Union[torch.device, str]] = None,
    skip_non_double_sided: bool = False,
    compute_importance: bool = False,
    knn_k: int = 16,
    missing_uv_policy: str = "blender",
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """One-shot: load mesh, sample surface, return (points, attrs).

    This is the main entry point replacing the old
    ``o_point.textured_mesh_to_surface_samples_gpu``.
    """
    sampler = MeshTextureSampler(
        mesh,
        device=device,
        missing_uv_policy=missing_uv_policy,
    )
    return sampler.sample(
        num_samples=num_samples,
        tonemap_method=tonemap_method,
        skip_non_double_sided=skip_non_double_sided,
        compute_importance=compute_importance,
        knn_k=knn_k,
    )


# ---------------------------------------------------------------------------
# Batch processor — reuses expensive state across many GLBs
# ---------------------------------------------------------------------------


class BatchProcessor:
    """Process many GLB files efficiently by reusing GPU context.

    Expensive one-time costs (nvdiffrast ``glctx`` creation, CUDA context
    warm-up) are paid once at ``__init__`` and amortised across all
    subsequent :meth:`process` calls.

    Example::

        proc = BatchProcessor(device="cuda")
        for glb in glob.glob("data/*.glb"):
            result = proc.process(glb, num_samples=500_000, render=True)
            # result.points, result.attrs, result.base_img, result.normal_img
    """

    @dataclass
    class Result:
        """Output of a single :meth:`process` call."""

        mesh: NvMesh
        points: torch.Tensor
        attrs: Dict[str, torch.Tensor]
        base_img: Optional["Image.Image"] = None
        normal_img: Optional["Image.Image"] = None
        amr_img: Optional["Image.Image"] = None
        geo_normal_img: Optional["Image.Image"] = None
        occlusion_img: Optional["Image.Image"] = None

    def __init__(
        self,
        device: str = "cuda",
        view_size: int = 1024,
        peel_layers: int = 8,
        missing_uv_policy: str = "blender",
        use_transmission: bool = False,
    ) -> None:
        from .render import NvMeshRenderer

        self.device = device
        self.view_size = view_size
        self.peel_layers = peel_layers
        self.missing_uv_policy = missing_uv_policy
        self.use_transmission = use_transmission
        self._renderer = NvMeshRenderer(
            device=device,
            resolution=view_size,
            peel_layers=peel_layers,
        )

    @property
    def renderer(self):  # -> NvMeshRenderer (lazy import)
        """The shared :class:`NvMeshRenderer`."""
        return self._renderer

    @property
    def glctx(self) -> object:
        """The shared nvdiffrast rasterization context."""
        return self._renderer.glctx

    def _do_render(
        self,
        mesh: NvMesh,
        with_amr: bool = False,
        with_geo_normal: bool = False,
        with_occlusion: bool = False,
    ) -> Dict[str, "Image.Image"]:
        return self._renderer.render_grid(
            mesh,
            view_size=self.view_size,
            with_amr=with_amr,
            with_geo_normal=with_geo_normal,
            with_occlusion=with_occlusion,
        )

    def process(
        self,
        glb_path: str,
        num_samples: int = 500_000,
        tonemap_method: Literal["srgb", "linear"] = "srgb",
        compute_importance: bool = False,
        knn_k: int = 16,
        render: bool = False,
        with_amr: bool = False,
        with_geo_normal: bool = False,
        with_occlusion: bool = False,
    ) -> "BatchProcessor.Result":
        """Load a GLB, sample surface, and optionally render views.

        All calls share the same renderer / ``glctx`` created at init time.
        """
        mesh = load_glb(
            glb_path,
            device=self.device,
            missing_uv_policy=self.missing_uv_policy,
            use_transmission=self.use_transmission,
        )
        sampler = MeshTextureSampler(mesh, device=self.device)
        points, attrs = sampler.sample(
            num_samples=num_samples,
            tonemap_method=tonemap_method,
            compute_importance=compute_importance,
            knn_k=knn_k,
        )

        base_img = None
        normal_img = None
        amr_img = None
        geo_normal_img = None
        occlusion_img = None
        if render:
            imgs = self._do_render(
                mesh,
                with_amr=with_amr,
                with_geo_normal=with_geo_normal,
                with_occlusion=with_occlusion,
            )
            base_img = imgs["base_color"]
            normal_img = imgs["shading_normal"]
            amr_img = imgs.get("amr")
            geo_normal_img = imgs.get("geo_normal")
            occlusion_img = imgs.get("occlusion")

        return BatchProcessor.Result(
            mesh=mesh,
            points=points,
            attrs=attrs,
            base_img=base_img,
            normal_img=normal_img,
            amr_img=amr_img,
            geo_normal_img=geo_normal_img,
            occlusion_img=occlusion_img,
        )

    def process_mesh(
        self,
        mesh: NvMesh,
        num_samples: int = 500_000,
        tonemap_method: Literal["srgb", "linear"] = "srgb",
        compute_importance: bool = False,
        knn_k: int = 16,
        render: bool = False,
        with_amr: bool = False,
        with_geo_normal: bool = False,
        with_occlusion: bool = False,
    ) -> "BatchProcessor.Result":
        """Same as :meth:`process` but accepts a pre-loaded :class:`NvMesh`."""
        sampler = MeshTextureSampler(mesh, device=self.device)
        points, attrs = sampler.sample(
            num_samples=num_samples,
            tonemap_method=tonemap_method,
            compute_importance=compute_importance,
            knn_k=knn_k,
        )

        base_img = None
        normal_img = None
        amr_img = None
        geo_normal_img = None
        occlusion_img = None
        if render:
            imgs = self._do_render(
                mesh,
                with_amr=with_amr,
                with_geo_normal=with_geo_normal,
                with_occlusion=with_occlusion,
            )
            base_img = imgs["base_color"]
            normal_img = imgs["shading_normal"]
            amr_img = imgs.get("amr")
            geo_normal_img = imgs.get("geo_normal")
            occlusion_img = imgs.get("occlusion")

        return BatchProcessor.Result(
            mesh=mesh,
            points=points,
            attrs=attrs,
            base_img=base_img,
            normal_img=normal_img,
            amr_img=amr_img,
            geo_normal_img=geo_normal_img,
            occlusion_img=occlusion_img,
        )
