"""render's two normal-map knobs on a unit quad facing the az000 camera.

normal_filter_mode: the map alternates tangent-space x = +-0.25 in 32-texel columns and
the quad repeats it twice across 8 pixels, so each pixel spans one whole period. Mip
filtering (the material's default) averages the columns away; linear sampling reads one
column per pixel (the UVs start 0.1 in, keeping every pixel centre off a column edge).

normal_perturbation_min: a flat map's 8-bit code 128 decodes to x = y = 0.0039, under the
default 0.01, so the normal is left as is; with 0.0 the texel tilts it.
"""

from __future__ import annotations

import json
import struct
from pathlib import Path

import numpy as np
import pytest
import torch
import trimesh
from PIL import Image

from o_point.materials import load_glb
from o_point.render import NvMeshRenderer

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


def _quad(path: Path, normal_codes: np.ndarray) -> None:
    quad = trimesh.Trimesh(
        vertices=[[-0.5, -0.5, 0], [0.5, -0.5, 0], [0.5, 0.5, 0], [-0.5, 0.5, 0]],
        faces=[[0, 1, 2], [0, 2, 3]],
        process=False,
    )
    quad.visual = trimesh.visual.TextureVisuals(
        uv=[[0.1, 0.0], [2.1, 0.0], [2.1, 2.0], [0.1, 2.0]],
        material=trimesh.visual.material.PBRMaterial(
            normalTexture=Image.fromarray(normal_codes), doubleSided=True
        ),
    )
    quad.export(path)
    # trimesh writes no sampler (LINEAR); Meshy's GLBs ask for mip filtering in theirs.
    data = path.read_bytes()
    (json_length,) = struct.unpack_from("<I", data, 12)
    gltf = json.loads(data[20 : 20 + json_length])
    gltf["samplers"] = [{"magFilter": 9729, "minFilter": 9987}]  # LINEAR_MIPMAP_LINEAR
    for texture in gltf["textures"]:
        texture["sampler"] = 0
    body = json.dumps(gltf).encode()
    body += b" " * (-len(body) % 4)
    rest = data[20 + json_length :]
    path.write_bytes(
        struct.pack("<III", 0x46546C67, 2, 20 + len(body) + len(rest))
        + struct.pack("<II", len(body), 0x4E4F534A)
        + body
        + rest
    )


def _shading_normals(path: Path, resolution: int, **knobs) -> torch.Tensor:
    """The az000 view's decoded shading normals at its covered pixels, [n, 3]."""
    mesh = load_glb(path, device="cuda", scale_factor=None)
    assert mesh.materials[0].normal_filter_mode == "linear-mipmap-linear"
    renderer = NvMeshRenderer(device="cuda", resolution=resolution)
    cameras = renderer.build_6_views(mesh)
    out = renderer.render(
        mesh,
        cameras.extrinsics[0],
        cameras.intrinsics[0],
        cameras.near[0],
        cameras.far[0],
        return_types=("shading_normal", "mask"),
        **knobs,
    )
    covered = out["mask"] > 0
    assert covered.sum() > 0
    return out["shading_normal"].permute(1, 2, 0)[covered] * 2.0 - 1.0


def test_linear_filter_keeps_detail_mip_averages(tmp_path: Path) -> None:
    columns = np.where(np.arange(256) % 64 < 32, 160, 96).astype(np.uint8)
    codes = np.empty((256, 256, 3), np.uint8)
    codes[..., 0] = columns[None, :]
    codes[..., 1] = 128
    codes[..., 2] = 255
    path = tmp_path / "columns.glb"
    _quad(path, codes)

    mipped = _shading_normals(path, 8)
    linear = _shading_normals(path, 8, normal_filter_mode="linear")

    assert float(mipped[:, 0].abs().mean()) < 0.05
    assert float(linear[:, 0].abs().mean()) > 0.2


def test_perturbation_min_applies_flat_code_texels(tmp_path: Path) -> None:
    path = tmp_path / "flat.glb"
    _quad(path, np.full((4, 4, 3), (128, 128, 255), np.uint8))

    gated = _shading_normals(path, 16)
    applied = _shading_normals(path, 16, normal_perturbation_min=0.0)

    torch.testing.assert_close(
        gated, torch.tensor([0.0, 0.0, 1.0], device="cuda").expand_as(gated)
    )
    assert float((applied - gated).abs().amax()) > 1e-3
