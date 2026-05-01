from __future__ import annotations

import numpy as np
import pytest
import torch

from o_point.materials import ALPHA_BLEND, ALPHA_MASK, NvMesh, NvPbrMaterial


class _Visual:
    def __init__(self, colors: np.ndarray, material: object | None = None) -> None:
        self.kind = "vertex"
        self.vertex_colors = colors
        self.material = material


class _TextureVisual:
    def __init__(self, colors: np.ndarray) -> None:
        self.kind = "texture"
        self.vertex_attributes = {"color": colors}


class _Geom:
    def __init__(
        self,
        colors: np.ndarray,
        *,
        textured: bool = False,
        material: object | None = None,
    ) -> None:
        self.vertices = np.zeros((3, 3), dtype=np.float32)
        self.faces = np.array([[0, 1, 2]], dtype=np.int32)
        self.visual = _TextureVisual(colors) if textured else _Visual(colors, material)


class _Material:
    def __init__(self, name: str, base_texture: object | None = None) -> None:
        self.name = name
        self.baseColorTexture = base_texture


class _Image:
    def convert(self, _mode: str) -> "_Image":
        return self


def test_vertex_colors_are_kept_as_linear_rgba_with_uvs() -> None:
    from o_point.materials import _try_get_face_vertex_colors

    colors = np.array(
        [
            [255, 128, 0, 64],
            [0, 255, 128, 128],
            [128, 0, 255, 255],
        ],
        dtype=np.uint8,
    )

    face_colors = _try_get_face_vertex_colors(
        _Geom(colors),
        np.array([[0, 1, 2]], dtype=np.int32),
    )

    assert face_colors is not None
    assert face_colors.shape == (1, 3, 4)
    np.testing.assert_allclose(face_colors[0, 0], [1.0, 128 / 255, 0.0, 64 / 255])


def test_texture_visual_color_attribute_is_kept() -> None:
    from o_point.materials import _try_get_face_vertex_colors

    colors = np.array(
        [
            [0.2, 0.4, 0.6],
            [0.3, 0.5, 0.7],
            [0.4, 0.6, 0.8],
        ],
        dtype=np.float32,
    )

    face_colors = _try_get_face_vertex_colors(
        _Geom(colors, textured=True),
        np.array([[0, 1, 2]], dtype=np.int32),
    )

    assert face_colors is not None
    assert face_colors.shape == (1, 3, 4)
    np.testing.assert_allclose(face_colors[0, 1], [0.3, 0.5, 0.7, 1.0])


def test_material_key_does_not_merge_same_named_materials() -> None:
    from o_point.materials import _material_key

    first = _Geom(np.ones((3, 3), dtype=np.float32), material=_Material("shared"))
    second = _Geom(np.ones((3, 3), dtype=np.float32), material=_Material("shared"))

    assert _material_key(first) != _material_key(second)


def test_material_texture_detection_for_missing_uv_guard() -> None:
    from o_point.materials import _material_has_texture

    geom = _Geom(
        np.ones((3, 3), dtype=np.float32),
        material=_Material("textured", base_texture=_Image()),
    )

    assert _material_has_texture(geom)


def test_material_disables_texture_slots_with_missing_texcoords() -> None:
    from o_point.materials import _disable_texture_slots_without_uv

    material = NvPbrMaterial(
        base_color_texture=torch.ones((1, 1, 3)),
        base_color_texcoord=0,
        normal_texture=torch.ones((1, 1, 3)),
        normal_texcoord=1,
    )

    patched = _disable_texture_slots_without_uv(material, {1})

    assert patched.base_color_texture is None
    assert patched.normal_texture is material.normal_texture
    assert material.base_color_texture is not None


def test_generated_face_uvs_use_local_xy_bounds() -> None:
    from o_point.materials import _generated_face_uvs

    vertices = np.array(
        [[2.0, 4.0, 0.0], [6.0, 4.0, 1.0], [2.0, 8.0, 2.0]],
        dtype=np.float32,
    )
    faces = np.array([[0, 1, 2]], dtype=np.int32)

    np.testing.assert_allclose(
        _generated_face_uvs(vertices, faces),
        np.array([[[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]], dtype=np.float32),
    )


def test_gltf_texture_sampler_info_maps_supported_modes() -> None:
    from o_point.materials import _texture_sampler_info

    gltf = {
        "textures": [{"sampler": 0}],
        "samplers": [
            {"wrapS": 33071, "wrapT": 33071, "minFilter": 9987, "magFilter": 9729}
        ],
    }

    assert _texture_sampler_info({"index": 0}, gltf) == (
        "linear-mipmap-linear",
        ("clamp", "clamp"),
        0,
        (0.0, 0.0),
        (1.0, 1.0),
        0.0,
    )


def test_gltf_texture_sampler_info_reads_texcoord_override() -> None:
    from o_point.materials import _texture_sampler_info

    gltf = {"textures": [{"sampler": 0}], "samplers": [{}]}

    assert _texture_sampler_info({"index": 0, "texCoord": 1}, gltf) == (
        "linear",
        ("wrap", "wrap"),
        1,
        (0.0, 0.0),
        (1.0, 1.0),
        0.0,
    )


def test_gltf_texture_sampler_info_reads_khr_texture_transform() -> None:
    from o_point.materials import _texture_sampler_info

    gltf = {"textures": [{"sampler": 0}], "samplers": [{}]}

    assert _texture_sampler_info(
        {
            "index": 0,
            "texCoord": 0,
            "extensions": {
                "KHR_texture_transform": {
                    "texCoord": 1,
                    "offset": [0.25, 0.5],
                    "scale": [2.0, 3.0],
                    "rotation": 1.0,
                }
            },
        },
        gltf,
    ) == ("linear", ("wrap", "wrap"), 1, (0.25, 0.5), (2.0, 3.0), 1.0)


def test_gltf_texture_sampler_info_supports_mixed_and_mirrored_wrap() -> None:
    from o_point.materials import _texture_sampler_info

    gltf = {
        "textures": [{"sampler": 0}],
        "samplers": [{"wrapS": 33648, "wrapT": 33071}],
    }

    assert _texture_sampler_info({"index": 0}, gltf) == (
        "linear",
        ("mirror", "clamp"),
        0,
        (0.0, 0.0),
        (1.0, 1.0),
        0.0,
    )


def test_prepare_texture_uv_handles_per_axis_wrap_modes() -> None:
    from o_point.materials import _prepare_texture_uv

    uv = torch.tensor([[-0.25, 1.5], [1.25, -0.5]], dtype=torch.float32)

    prepared, native_boundary = _prepare_texture_uv(uv, ("mirror", "wrap"))

    assert native_boundary == "clamp"
    torch.testing.assert_close(
        prepared,
        torch.tensor([[0.25, 0.5], [0.75, 0.5]], dtype=torch.float32),
    )


def test_prepare_texture_uv_flips_runtime_v_for_texture_rows() -> None:
    from o_point.materials import _prepare_texture_uv

    uv = torch.tensor([[0.25, 0.25]], dtype=torch.float32)

    prepared, native_boundary = _prepare_texture_uv(uv, ("clamp", "clamp"))

    assert native_boundary == "clamp"
    torch.testing.assert_close(
        prepared,
        torch.tensor([[0.25, 0.75]], dtype=torch.float32),
    )


def test_prepare_texture_uv_applies_khr_transform_in_gltf_space() -> None:
    from o_point.materials import _prepare_texture_uv

    uv = torch.tensor([[0.25, 0.25]], dtype=torch.float32)

    prepared, native_boundary = _prepare_texture_uv(
        uv,
        ("clamp", "clamp"),
        offset=(0.25, 0.25),
        scale=(2.0, 0.5),
    )

    assert native_boundary == "clamp"
    torch.testing.assert_close(
        prepared,
        torch.tensor([[0.75, 0.625]], dtype=torch.float32),
    )


def test_collect_gltf_primitive_uv_sets_reads_texcoord_1() -> None:
    from o_point.materials import _collect_gltf_primitive_uv_sets

    positions = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        dtype=np.float32,
    )
    indices = np.array([0, 1, 2], dtype=np.uint16)
    uv1 = np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]], dtype=np.float32)
    chunks = [positions.tobytes(), indices.tobytes(), uv1.tobytes()]
    offsets = np.cumsum([0] + [len(chunk) for chunk in chunks[:-1]]).tolist()
    gltf = {
        "bufferViews": [
            {"buffer": 0, "byteOffset": offsets[0], "byteLength": len(chunks[0])},
            {"buffer": 0, "byteOffset": offsets[1], "byteLength": len(chunks[1])},
            {"buffer": 0, "byteOffset": offsets[2], "byteLength": len(chunks[2])},
        ],
        "accessors": [
            {"bufferView": 0, "componentType": 5126, "count": 3, "type": "VEC3"},
            {"bufferView": 1, "componentType": 5123, "count": 3, "type": "SCALAR"},
            {"bufferView": 2, "componentType": 5126, "count": 3, "type": "VEC2"},
        ],
        "meshes": [
            {
                "name": "Plane",
                "primitives": [
                    {
                        "attributes": {"POSITION": 0, "TEXCOORD_1": 2},
                        "indices": 1,
                    }
                ],
            }
        ],
    }

    uv_sets = _collect_gltf_primitive_uv_sets(gltf, b"".join(chunks))

    np.testing.assert_allclose(
        uv_sets["Plane"][1],
        np.array(
            [[[0.1, 0.8], [0.3, 0.6], [0.5, 0.4]]],
            dtype=np.float32,
        ),
    )


def test_face_tangents_follow_uv_basis() -> None:
    from o_point.materials import _compute_face_tangents

    triangles = np.array(
        [[[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 3.0, 0.0]]],
        dtype=np.float32,
    )
    normals = np.array(
        [[[0.0, 0.0, 1.0], [0.0, 0.0, 1.0], [0.0, 0.0, 1.0]]],
        dtype=np.float32,
    )
    uvs = np.array([[[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]], dtype=np.float32)

    tangents = _compute_face_tangents(triangles, normals, uvs)

    assert tangents.shape == (1, 3, 4)
    np.testing.assert_allclose(tangents[0, 0], [1.0, 0.0, 0.0, 1.0], atol=1e-6)


@pytest.mark.parametrize(
    ("alpha_mode", "expected_alpha"),
    [
        (ALPHA_BLEND, 0.25),
        (ALPHA_MASK, 0.0),
    ],
)
def test_sampler_combines_factor_and_vertex_color_without_texture(
    alpha_mode: int,
    expected_alpha: float,
) -> None:
    pytest.importorskip("nvdiffrast.torch")
    from o_point.sampler import MeshTextureSampler

    mat = NvPbrMaterial(
        base_color_factor=torch.tensor([0.5, 0.25, 0.75, 0.5]),
        alpha_factor=0.5,
        alpha_mode=alpha_mode,
        alpha_cutoff=0.4,
    )
    mesh = NvMesh(
        vertices=torch.tensor(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            dtype=torch.float32,
        ),
        faces=torch.tensor([[0, 1, 2]], dtype=torch.int32),
        triangles=torch.tensor(
            [[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]],
            dtype=torch.float32,
        ),
        normals=torch.tensor(
            [[[0.0, 0.0, 1.0], [0.0, 0.0, 1.0], [0.0, 0.0, 1.0]]],
            dtype=torch.float32,
        ),
        tangents=torch.tensor(
            [[[1.0, 0.0, 0.0, 1.0], [1.0, 0.0, 0.0, 1.0], [1.0, 0.0, 0.0, 1.0]]],
            dtype=torch.float32,
        ),
        uv_coords=torch.zeros((1, 3, 2), dtype=torch.float32),
        material_ids=torch.tensor([0], dtype=torch.int32),
        materials=[mat],
        face_vertex_colors=torch.tensor(
            [[[0.2, 0.4, 0.8, 0.5], [0.2, 0.4, 0.8, 0.5], [0.2, 0.4, 0.8, 0.5]]],
            dtype=torch.float32,
        ),
        vertex_color_mask=torch.tensor([True]),
    )

    sampler = MeshTextureSampler(mesh, device="cpu")
    _, attrs = sampler.sample_from_face_bary(
        torch.tensor([[0.25, 0.25, 0.0]], dtype=torch.float32),
        torch.tensor([0], dtype=torch.int32),
        torch.tensor([[0.5, 0.25, 0.25]], dtype=torch.float32),
        tonemap_method="linear",
    )

    torch.testing.assert_close(
        attrs["base_color"],
        torch.tensor([[0.1, 0.1, 0.6]], dtype=torch.float32),
    )
    torch.testing.assert_close(
        attrs["alpha"],
        torch.tensor([[expected_alpha]], dtype=torch.float32),
    )


def test_skip_non_double_sided_filters_sampling_cdf() -> None:
    pytest.importorskip("nvdiffrast.torch")
    from o_point.sampler import MeshTextureSampler

    mesh = NvMesh(
        vertices=torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
                [1.0, 0.0, 1.0],
                [0.0, 1.0, 1.0],
            ],
            dtype=torch.float32,
        ),
        faces=torch.tensor([[0, 1, 2], [3, 4, 5]], dtype=torch.int32),
        triangles=torch.tensor(
            [
                [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
                [[0.0, 0.0, 1.0], [1.0, 0.0, 1.0], [0.0, 1.0, 1.0]],
            ],
            dtype=torch.float32,
        ),
        normals=torch.tensor(
            [
                [[0.0, 0.0, 1.0], [0.0, 0.0, 1.0], [0.0, 0.0, 1.0]],
                [[0.0, 0.0, 1.0], [0.0, 0.0, 1.0], [0.0, 0.0, 1.0]],
            ],
            dtype=torch.float32,
        ),
        tangents=torch.tensor(
            [
                [[1.0, 0.0, 0.0, 1.0], [1.0, 0.0, 0.0, 1.0], [1.0, 0.0, 0.0, 1.0]],
                [[1.0, 0.0, 0.0, 1.0], [1.0, 0.0, 0.0, 1.0], [1.0, 0.0, 0.0, 1.0]],
            ],
            dtype=torch.float32,
        ),
        uv_coords=torch.zeros((2, 3, 2), dtype=torch.float32),
        material_ids=torch.tensor([0, 1], dtype=torch.int32),
        materials=[NvPbrMaterial(double_sided=False), NvPbrMaterial(double_sided=True)],
    )

    sampler = MeshTextureSampler(mesh, device="cpu")
    cdf = sampler._build_face_area_cdf(skip_non_double_sided=True)

    torch.testing.assert_close(cdf, torch.tensor([0.0, 0.5]))
