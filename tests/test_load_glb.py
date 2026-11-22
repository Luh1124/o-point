"""load_glb places a primitive by its node's matrix: positions by the matrix, normals by
its inverse-transpose, TANGENT by the matrix with its handedness flipped under a mirror,
and every per-corner attribute stays on the corner of its own vertex when a mirror
reverses the winding.

Each case writes a one-triangle GLB by hand, with a distinct TANGENT per vertex so a
corner that picked up another vertex's tangent is visible, and compares load_glb's
output with the transform applied analytically.
"""

from __future__ import annotations

import json
import struct
from pathlib import Path

import numpy as np
import pytest
import torch

from o_point.materials import load_glb

POSITIONS = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float32)
NORMALS = np.array([[0, 0, 1]] * 3, dtype=np.float32)
UVS = np.array([[0, 0], [1, 0], [0, 1]], dtype=np.float32)
TANGENTS = np.array([[1, 0, 0, 1], [0.8, 0.6, 0, 1], [0.6, 0.8, 0, -1]], dtype=np.float32)
INDICES = np.array([0, 1, 2], dtype=np.uint16)


def _write_glb(path: Path, matrix: np.ndarray) -> None:
    arrays = [POSITIONS, NORMALS, UVS, TANGENTS, INDICES]
    kinds = ["VEC3", "VEC3", "VEC2", "VEC4", "SCALAR"]
    blob = b""
    views, accessors = [], []
    for array, kind in zip(arrays, kinds):
        blob += b"\0" * (-len(blob) % 4)
        views.append({"buffer": 0, "byteOffset": len(blob), "byteLength": array.nbytes})
        accessor = {
            "bufferView": len(views) - 1,
            "componentType": 5123 if array.dtype == np.uint16 else 5126,
            "count": len(array),
            "type": kind,
        }
        if kind == "VEC3" and not accessors:
            accessor |= {"min": array.min(0).tolist(), "max": array.max(0).tolist()}
        accessors.append(accessor)
        blob += array.tobytes()
    blob += b"\0" * (-len(blob) % 4)
    gltf = {
        "asset": {"version": "2.0"},
        "scene": 0,
        "scenes": [{"nodes": [0]}],
        "nodes": [{"mesh": 0, "matrix": matrix.T.reshape(-1).tolist()}],
        "meshes": [
            {
                "name": "tri",
                "primitives": [
                    {
                        "attributes": {
                            "POSITION": 0,
                            "NORMAL": 1,
                            "TEXCOORD_0": 2,
                            "TANGENT": 3,
                        },
                        "indices": 4,
                    }
                ],
            }
        ],
        "buffers": [{"byteLength": len(blob)}],
        "bufferViews": views,
        "accessors": accessors,
    }
    body = json.dumps(gltf).encode()
    body += b" " * (-len(body) % 4)
    total = 12 + 8 + len(body) + 8 + len(blob)
    path.write_bytes(
        struct.pack("<III", 0x46546C67, 2, total)
        + struct.pack("<II", len(body), 0x4E4F534A)
        + body
        + struct.pack("<II", len(blob), 0x004E4942)
        + blob
    )


ROTATE_Y_90 = np.array(
    [[0, 0, 1, 0.5], [0, 1, 0, -0.2], [-1, 0, 0, 0.3], [0, 0, 0, 1]], dtype=np.float64
)
MIRROR_X = np.diag([-1.0, 1.0, 1.0, 1.0])


@pytest.mark.parametrize("matrix", [ROTATE_Y_90, MIRROR_X], ids=["rotate", "mirror"])
def test_node_matrix_places_every_attribute(tmp_path: Path, matrix: np.ndarray) -> None:
    path = tmp_path / "tri.glb"
    _write_glb(path, matrix)

    mesh = load_glb(path, device="cpu", scale_factor=None)

    rotation = matrix[:3, :3]
    mirrored = np.linalg.det(rotation) < 0
    faces = mesh.faces.numpy()
    assert faces.shape == (1, 3)
    if mirrored:
        np.testing.assert_array_equal(faces, [[2, 1, 0]])
    else:
        np.testing.assert_array_equal(faces, [[0, 1, 2]])

    corners = faces[0]
    expected_positions = POSITIONS[corners] @ rotation.T + matrix[:3, 3]
    np.testing.assert_allclose(mesh.triangles[0].numpy(), expected_positions, atol=1e-6)

    normal = np.linalg.inv(rotation).T @ NORMALS[0]
    normal /= np.linalg.norm(normal)
    np.testing.assert_allclose(mesh.normals[0].numpy(), [normal] * 3, atol=1e-6)
    # The winding the drawn triangle has agrees with its normals.
    edges = expected_positions[1:] - expected_positions[0]
    assert np.cross(edges[0], edges[1]) @ normal > 0

    tangent_xyz = TANGENTS[corners, :3] @ rotation.T
    tangent_xyz /= np.linalg.norm(tangent_xyz, axis=1, keepdims=True)
    handedness = TANGENTS[corners, 3:] * (-1.0 if mirrored else 1.0)
    np.testing.assert_allclose(
        mesh.tangents[0].numpy(),
        np.concatenate([tangent_xyz, handedness], axis=1),
        atol=1e-6,
    )

    expected_uvs = UVS[corners] * [1, -1] + [0, 1]
    np.testing.assert_allclose(mesh.uv_coords[0].numpy(), expected_uvs, atol=1e-6)


def test_identity_node_keeps_the_file_vertices(tmp_path: Path) -> None:
    path = tmp_path / "tri.glb"
    _write_glb(path, np.eye(4))

    mesh = load_glb(path, device="cpu", scale_factor=None)

    torch.testing.assert_close(mesh.vertices, torch.from_numpy(POSITIONS))
    assert mesh.normals.dtype == torch.float32
    torch.testing.assert_close(mesh.tangents[0], torch.from_numpy(TANGENTS))
