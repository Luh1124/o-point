"""render's raw_back_faces on a single-sided unit quad, drawn from both sides.

The az000 camera sees one winding of the quad from the front and the reversed one from
the back. By default the back view is culled to nothing, as glTF specifies for a
single-sided material: alpha 0, though the raster mask still covers it. With
raw_back_faces both views cover the quad and each keeps its own winding's normal, so the
two views' normals are opposite: the back view's points away from the camera.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
import trimesh

from o_point.materials import load_glb
from o_point.render import NvMeshRenderer

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")

WINDINGS = [[[0, 1, 2], [0, 2, 3]], [[0, 2, 1], [0, 3, 2]]]


def _drawn(path: Path, faces, raw_back_faces: bool) -> tuple[float, torch.Tensor]:
    """The az000 view's coverage and its mean decoded shading normal."""
    quad = trimesh.Trimesh(
        vertices=[[-0.5, -0.5, 0], [0.5, -0.5, 0], [0.5, 0.5, 0], [-0.5, 0.5, 0]],
        faces=faces,
        process=False,
    )
    quad.visual = trimesh.visual.TextureVisuals(
        material=trimesh.visual.material.PBRMaterial(doubleSided=False)
    )
    quad.export(path)
    mesh = load_glb(path, device="cuda", scale_factor=None)
    assert not mesh.materials[0].double_sided
    renderer = NvMeshRenderer(device="cuda", resolution=16)
    cameras = renderer.build_6_views(mesh)
    out = renderer.render(
        mesh,
        cameras.extrinsics[0],
        cameras.intrinsics[0],
        cameras.near[0],
        cameras.far[0],
        return_types=("shading_normal", "alpha"),
        raw_back_faces=raw_back_faces,
    )
    covered = out["alpha"] > 0
    normal = out["shading_normal"].permute(1, 2, 0)[covered] * 2.0 - 1.0
    return float(covered.float().mean()), normal.mean(0)


def test_raw_back_faces_draws_the_culled_side_unturned(tmp_path: Path) -> None:
    default = [_drawn(tmp_path / f"d{i}.glb", w, False) for i, w in enumerate(WINDINGS)]
    raw = [_drawn(tmp_path / f"r{i}.glb", w, True) for i, w in enumerate(WINDINGS)]

    culled, kept = sorted(coverage for coverage, _ in default)
    assert culled == 0.0
    assert kept > 0.1
    assert min(coverage for coverage, _ in raw) > 0.1
    torch.testing.assert_close(raw[0][1], -raw[1][1], atol=1e-3, rtol=0.0)
