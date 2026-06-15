from __future__ import annotations

import torch
import torch.nn.functional as F


def _face_normal(tris: torch.Tensor, geometry_normal: torch.Tensor) -> torch.Tensor:
    """Flat triangle face normal oriented to agree with the smooth normal.

    Mirrors the computation in
    ``MeshTextureSampler.sample_from_face_bary`` so it can run on CPU
    without the GPU ``dr.texture`` material path.
    """
    e1 = tris[:, 1] - tris[:, 0]
    e2 = tris[:, 2] - tris[:, 0]
    face_n = F.normalize(torch.cross(e1, e2, dim=-1), dim=-1)
    sign = torch.sign((face_n * geometry_normal).sum(-1, keepdim=True))
    sign = torch.where(sign == 0, torch.ones_like(sign), sign)
    return face_n * sign


def test_face_normal_matches_analytic_cross_product() -> None:
    # Triangle in the z=0 plane; geometric normal is +z by right-hand rule.
    tris = torch.tensor([[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]])
    geometry_normal = torch.tensor([[0.0, 0.0, 1.0]])

    face_n = _face_normal(tris, geometry_normal)

    expected = torch.tensor([[0.0, 0.0, 1.0]])
    assert torch.allclose(face_n, expected, atol=1e-6), face_n
    assert torch.allclose(face_n.norm(dim=-1), torch.ones(1), atol=1e-6)


def test_face_normal_oriented_to_geometry_normal() -> None:
    # Same triangle, but the smooth/geometry normal points -z, so the flat
    # face normal must be flipped to agree with it.
    tris = torch.tensor([[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]])
    geometry_normal = torch.tensor([[0.0, 0.0, -1.0]])

    face_n = _face_normal(tris, geometry_normal)

    expected = torch.tensor([[0.0, 0.0, -1.0]])
    assert torch.allclose(face_n, expected, atol=1e-6), face_n
    # Agreement with the geometry normal: dot product must be positive.
    dot = (face_n * geometry_normal).sum(-1)
    assert (dot > 0).all(), dot
