from __future__ import annotations

import torch
import torch.nn.functional as F

import o_point.render as render


def _per_pixel_face_normal_enc(
    vertices: torch.Tensor,
    faces: torch.Tensor,
    tri_id: torch.Tensor,
    geometry_normal: torch.Tensor,
) -> torch.Tensor:
    """Pure-CPU mirror of the per-pixel face_normal block in ``render``.

    ``tri_id`` is the 1-based rasterized triangle id per pixel (H, W, 1);
    ``geometry_normal`` is the interpolated smooth normal per pixel (H, W, 3).
    Returns the ``n * 0.5 + 0.5`` encoded flat face normal (H, W, 3).
    """
    fv0 = vertices[faces[:, 0]]
    fv1 = vertices[faces[:, 1]]
    fv2 = vertices[faces[:, 2]]
    per_face_normal = F.normalize(torch.cross(fv1 - fv0, fv2 - fv0, dim=1), dim=1)

    face_ids = (tri_id.long() - 1).squeeze(-1).clamp(min=0)
    gb_face_normal = per_face_normal[face_ids]
    flip = (gb_face_normal * geometry_normal).sum(dim=-1, keepdim=True) < 0
    gb_face_normal = torch.where(flip, -gb_face_normal, gb_face_normal)
    return gb_face_normal * 0.5 + 0.5


def test_per_pixel_face_normal_gather_and_encode() -> None:
    # Two triangles in the z=0 plane; right-hand-rule normal is +z.
    vertices = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
        ]
    )
    faces = torch.tensor([[0, 1, 2], [1, 3, 2]], dtype=torch.int64)
    # One pixel hits tri 1, another hits tri 2 (1-based ids).
    tri_id = torch.tensor([[[1], [2]]], dtype=torch.float32)  # (1, 2, 1)
    geometry_normal = torch.tensor([[[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]]])

    enc = _per_pixel_face_normal_enc(vertices, faces, tri_id, geometry_normal)

    # +z encoded as 0.5*z+0.5 -> (0.5, 0.5, 1.0)
    expected = torch.tensor([[[0.5, 0.5, 1.0], [0.5, 0.5, 1.0]]])
    assert torch.allclose(enc, expected, atol=1e-6), enc


def test_per_pixel_face_normal_oriented_to_geometry_normal() -> None:
    vertices = torch.tensor(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
    )
    faces = torch.tensor([[0, 1, 2]], dtype=torch.int64)
    tri_id = torch.tensor([[[1]]], dtype=torch.float32)
    # Geometry normal points -z, so the flat face normal must flip to -z.
    geometry_normal = torch.tensor([[[0.0, 0.0, -1.0]]])

    enc = _per_pixel_face_normal_enc(vertices, faces, tri_id, geometry_normal)
    decoded = enc * 2.0 - 1.0
    assert torch.allclose(decoded, torch.tensor([[[0.0, 0.0, -1.0]]]), atol=1e-6)


def test_return_types_plumbing() -> None:
    # New aligned attribute names are registered; old names are gone.
    rt = render.ALL_RETURN_TYPES
    assert {"geometry_normal", "face_normal", "emission"} <= rt
    assert "geo_normal" not in rt
    assert "emissive" not in rt
