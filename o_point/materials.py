"""PBR material data structures and GLB loading for nvdiffrast-based sampling."""

from __future__ import annotations

import json
import math
import struct
import warnings
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Optional, Union

import numpy as np
import torch
import trimesh
from PIL import Image

Image.MAX_IMAGE_PIXELS = None

# ---------------------------------------------------------------------------
# Alpha mode constants
# ---------------------------------------------------------------------------

ALPHA_OPAQUE = 0
ALPHA_MASK = 1
ALPHA_BLEND = 2

_ALPHA_MODE_MAP = {"OPAQUE": ALPHA_OPAQUE, "MASK": ALPHA_MASK, "BLEND": ALPHA_BLEND}

GLTF_REPEAT = 10497
GLTF_CLAMP_TO_EDGE = 33071
GLTF_MIRRORED_REPEAT = 33648
GLTF_NEAREST = 9728
GLTF_LINEAR = 9729
GLTF_TRIANGLES = 4
GLTF_TRIANGLE_STRIP = 5


# ---------------------------------------------------------------------------
# Color-space helpers
# ---------------------------------------------------------------------------


def srgb_to_linear(x: torch.Tensor) -> torch.Tensor:
    """sRGB → linear (standard IEC 61966-2-1), works on any shape."""
    return torch.where(x > 0.04045, torch.pow((x + 0.055) / 1.055, 2.4), x / 12.92)


def linear_to_srgb(x: torch.Tensor) -> torch.Tensor:
    """Linear → sRGB (standard IEC 61966-2-1), works on any shape."""
    x = torch.clamp(x, min=0.0)
    return torch.where(
        x > 0.0031308,
        1.055 * torch.pow(x, 1.0 / 2.4) - 0.055,
        12.92 * x,
    )


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class NvPbrMaterial:
    """Full PBR material with textures stored as float32 tensors on GPU,
    ready for ``dr.texture`` sampling."""

    base_color_texture: Optional[torch.Tensor] = None  # (H, W, 3)
    base_color_filter_mode: str = "linear-mipmap-linear"
    base_color_boundary_mode: tuple[str, str] = ("wrap", "wrap")
    base_color_texcoord: int = 0
    base_color_uv_offset: tuple[float, float] = (0.0, 0.0)
    base_color_uv_scale: tuple[float, float] = (1.0, 1.0)
    base_color_uv_rotation: float = 0.0
    base_color_factor: torch.Tensor = field(
        default_factory=lambda: torch.ones(4, dtype=torch.float32)
    )

    metallic_texture: Optional[torch.Tensor] = None  # (H, W, 1)
    metallic_filter_mode: str = "linear-mipmap-linear"
    metallic_boundary_mode: tuple[str, str] = ("wrap", "wrap")
    metallic_texcoord: int = 0
    metallic_uv_offset: tuple[float, float] = (0.0, 0.0)
    metallic_uv_scale: tuple[float, float] = (1.0, 1.0)
    metallic_uv_rotation: float = 0.0
    metallic_factor: float = 1.0

    roughness_texture: Optional[torch.Tensor] = None  # (H, W, 1)
    roughness_filter_mode: str = "linear-mipmap-linear"
    roughness_boundary_mode: tuple[str, str] = ("wrap", "wrap")
    roughness_texcoord: int = 0
    roughness_uv_offset: tuple[float, float] = (0.0, 0.0)
    roughness_uv_scale: tuple[float, float] = (1.0, 1.0)
    roughness_uv_rotation: float = 0.0
    roughness_factor: float = 1.0

    emissive_texture: Optional[torch.Tensor] = None  # (H, W, 3)
    emissive_filter_mode: str = "linear-mipmap-linear"
    emissive_boundary_mode: tuple[str, str] = ("wrap", "wrap")
    emissive_texcoord: int = 0
    emissive_uv_offset: tuple[float, float] = (0.0, 0.0)
    emissive_uv_scale: tuple[float, float] = (1.0, 1.0)
    emissive_uv_rotation: float = 0.0
    emissive_factor: torch.Tensor = field(
        default_factory=lambda: torch.zeros(3, dtype=torch.float32)
    )

    normal_texture: Optional[torch.Tensor] = None  # (H, W, 3)
    normal_filter_mode: str = "linear-mipmap-linear"
    normal_boundary_mode: tuple[str, str] = ("wrap", "wrap")
    normal_texcoord: int = 0
    normal_uv_offset: tuple[float, float] = (0.0, 0.0)
    normal_uv_scale: tuple[float, float] = (1.0, 1.0)
    normal_uv_rotation: float = 0.0
    # glTF §5.22: scales XY of the decoded tangent-space normal before normalizing
    normal_scale: float = 1.0

    occlusion_texture: Optional[torch.Tensor] = None  # (H, W, 1) — R channel
    occlusion_filter_mode: str = "linear-mipmap-linear"
    occlusion_boundary_mode: tuple[str, str] = ("wrap", "wrap")
    occlusion_texcoord: int = 0
    occlusion_uv_offset: tuple[float, float] = (0.0, 0.0)
    occlusion_uv_scale: tuple[float, float] = (1.0, 1.0)
    occlusion_uv_rotation: float = 0.0
    occlusion_strength: float = 1.0

    alpha_texture: Optional[torch.Tensor] = None  # (H, W, 1)
    alpha_filter_mode: str = "nearest"
    alpha_boundary_mode: tuple[str, str] = ("wrap", "wrap")
    alpha_texcoord: int = 0
    alpha_uv_offset: tuple[float, float] = (0.0, 0.0)
    alpha_uv_scale: tuple[float, float] = (1.0, 1.0)
    alpha_uv_rotation: float = 0.0
    alpha_factor: float = 1.0
    alpha_mode: int = ALPHA_OPAQUE
    alpha_cutoff: float = 0.5

    double_sided: bool = False
    # KHR_materials_unlit: render as emissive surface (no PBR metallic/roughness/normal)
    is_unlit: bool = False

    # KHR_materials_transmission: approximated as alpha = 1 - transmission_factor
    transmission_factor: float = 0.0
    transmission_texture: Optional[torch.Tensor] = None  # (H, W, 1) — R channel
    transmission_filter_mode: str = "linear-mipmap-linear"
    transmission_boundary_mode: tuple[str, str] = ("wrap", "wrap")
    transmission_texcoord: int = 0
    transmission_uv_offset: tuple[float, float] = (0.0, 0.0)
    transmission_uv_scale: tuple[float, float] = (1.0, 1.0)
    transmission_uv_rotation: float = 0.0


@dataclass
class NvMesh:
    """Concatenated mesh geometry with per-face material assignment."""

    vertices: torch.Tensor  # (V, 3) float32
    faces: torch.Tensor  # (F, 3) int32
    triangles: torch.Tensor  # (F, 3, 3) float32 — vertex positions per face
    normals: torch.Tensor  # (F, 3, 3) float32 — vertex normals per face
    tangents: torch.Tensor  # (F, 3, 4) float32 — xyz tangent plus handedness
    uv_coords: torch.Tensor  # (F, 3, 2) float32 — per-corner UV
    material_ids: torch.Tensor  # (F,) int32
    uv_sets: dict[int, torch.Tensor] = field(default_factory=dict)
    materials: list[NvPbrMaterial] = field(default_factory=list)
    face_vertex_colors: Optional[torch.Tensor] = None  # (F, 3, 4) linear RGBA
    vertex_color_mask: Optional[torch.Tensor] = (
        None  # (F,) bool — True if face has COLOR_0 data
    )


# ---------------------------------------------------------------------------
# Texture helpers
# ---------------------------------------------------------------------------


def _get_texture_image(mat: object, attr_name: str) -> Optional[Image.Image]:
    tex = getattr(mat, attr_name, None)
    if tex is None:
        return None
    if hasattr(tex, "convert"):
        return tex
    if hasattr(tex, "image"):
        return tex.image
    return None


def _read_glb_chunks(glb_path: Path) -> tuple[dict, bytes]:
    """Read JSON and BIN chunks from a binary .glb file."""
    with glb_path.open("rb") as f:
        header = f.read(12)
        if len(header) != 12:
            raise RuntimeError(f"Invalid GLB header in {glb_path}")
        magic, version, _length = struct.unpack("<III", header)
        if magic != 0x46546C67 or version != 2:
            raise RuntimeError(f"Expected glTF 2.0 GLB file: {glb_path}")
        # glTF §4: first chunk MUST be JSON, second (optional) MUST be BIN
        chunk_header = f.read(8)
        if len(chunk_header) != 8:
            raise RuntimeError(f"Invalid GLB chunk header in {glb_path}")
        chunk_length, chunk_type = struct.unpack("<II", chunk_header)
        if chunk_type != 0x4E4F534A:
            raise RuntimeError(f"First GLB chunk is not JSON in {glb_path}")
        chunk = f.read(chunk_length)
        if len(chunk) != chunk_length:
            raise RuntimeError(f"Truncated GLB JSON chunk in {glb_path}")
        gltf = json.loads(chunk.decode("utf-8"))

        bin_blob = b""
        if f.tell() < _length:
            chunk_header = f.read(8)
            if len(chunk_header) == 8:
                chunk_length, chunk_type = struct.unpack("<II", chunk_header)
                if chunk_type == 0x004E4942:
                    bin_blob = f.read(chunk_length)

        return gltf, bin_blob


def _read_glb_json(glb_path: Path) -> dict:
    """Read the JSON chunk from a binary .glb file."""
    return _read_glb_chunks(glb_path)[0]


_ACCESSOR_DTYPES = {
    5120: np.int8,
    5121: np.uint8,
    5122: np.int16,
    5123: np.uint16,
    5125: np.uint32,
    5126: np.float32,
}

_ACCESSOR_COMPONENTS = {
    "SCALAR": 1,
    "VEC2": 2,
    "VEC3": 3,
    "VEC4": 4,
    "MAT2": 4,
    "MAT3": 9,
    "MAT4": 16,
}


def _accessor_array(gltf: dict, bin_blob: bytes, accessor_idx: int) -> np.ndarray:
    """Read a dense accessor from the GLB BIN chunk."""
    accessor = gltf["accessors"][int(accessor_idx)]
    if "sparse" in accessor:
        raise RuntimeError("Sparse glTF accessors are not supported")
    dtype = np.dtype(_ACCESSOR_DTYPES[int(accessor["componentType"])])
    n_comp = _ACCESSOR_COMPONENTS[str(accessor["type"])]
    count = int(accessor["count"])
    # glTF §5.14: accessor with no bufferView and no sparse is all zeros
    if "bufferView" not in accessor:
        return np.zeros((count, n_comp), dtype=dtype)
    view = gltf["bufferViews"][int(accessor["bufferView"])]
    if int(view.get("buffer", 0)) != 0:
        raise RuntimeError("Only single-buffer GLB files are supported")

    byte_offset = int(view.get("byteOffset", 0)) + int(accessor.get("byteOffset", 0))
    byte_stride = int(view.get("byteStride", dtype.itemsize * n_comp))
    item_size = dtype.itemsize * n_comp
    raw = memoryview(bin_blob)
    if byte_stride == item_size:
        array = np.frombuffer(
            raw[byte_offset : byte_offset + count * item_size], dtype=dtype
        )
        array = array.reshape(count, n_comp).copy()
    else:
        array = np.empty((count, n_comp), dtype=dtype)
        for i in range(count):
            start = byte_offset + i * byte_stride
            array[i] = np.frombuffer(
                raw[start : start + item_size], dtype=dtype, count=n_comp
            )

    if accessor.get("normalized", False) and dtype != np.float32:
        array = array.astype(np.float32)
        if np.issubdtype(dtype, np.unsignedinteger):
            array /= float(np.iinfo(dtype).max)
        else:
            info = np.iinfo(dtype)
            array = np.maximum(array / float(info.max), -1.0)
    return array


def _unique_gltf_name(start: Optional[str], contains: dict, counts: dict) -> str:
    """Mirror trimesh.util.unique_name for glTF primitive names."""
    if start is not None and len(start) > 0 and start not in contains:
        return start
    increment = counts.get(start, 0)
    if start is not None and len(start) > 0:
        formatter = start + "_{}"
        split = start.rsplit("_", 1)
        if len(split) == 2 and increment == 0:
            try:
                increment = int(split[1])
                formatter = split[0] + "_{}"
            except ValueError:
                pass
    else:
        formatter = "geometry_{}"
    for i in range(increment + 1, 2 + increment + len(contains)):
        check = formatter.format(i)
        if check not in contains:
            counts[start] = i
            return check
    raise RuntimeError("Unable to establish unique glTF primitive name")


def _triangle_strip_to_faces(strip: np.ndarray) -> np.ndarray:
    faces = []
    for i in range(len(strip) - 2):
        if i % 2 == 0:
            face = [strip[i], strip[i + 1], strip[i + 2]]
        else:
            face = [strip[i + 1], strip[i], strip[i + 2]]
        if len(set(int(v) for v in face)) == 3:
            faces.append(face)
    return np.asarray(faces, dtype=np.int32)


def _collect_gltf_primitive_data(
    gltf: dict,
    bin_blob: bytes,
) -> tuple[dict[str, dict[int, np.ndarray]], dict[str, Optional[np.ndarray]]]:
    """Collect per-face-corner TEXCOORD_n and per-vertex TANGENT arrays.

    Returns:
        primitive_uvs: geometry_name -> {texcoord_index -> face_uv (F, 3, 2)}
        primitive_tangents: geometry_name -> per-vertex tangents (V, 4) or None
    """
    meshes_by_name: dict[str, object] = {}
    name_counts: dict[Optional[str], int] = {}
    primitive_uvs: dict[str, dict[int, np.ndarray]] = {}
    primitive_tangents: dict[str, Optional[np.ndarray]] = {}

    for mesh in gltf.get("meshes", []):
        mesh_name = mesh.get("name", "GLTF")
        for primitive in mesh.get("primitives", []):
            mode = int(primitive.get("mode", GLTF_TRIANGLES))
            if mode not in (GLTF_TRIANGLES, GLTF_TRIANGLE_STRIP):
                warnings.warn(
                    f"Skipping glTF primitive with unsupported mode {mode} "
                    f"(only TRIANGLES=4 and TRIANGLE_STRIP=5 are supported)",
                    RuntimeWarning,
                    stacklevel=2,
                )
                continue
            name = _unique_gltf_name(mesh_name, meshes_by_name, name_counts)
            meshes_by_name[name] = object()
            attr = primitive.get("attributes", {})
            pos = _accessor_array(gltf, bin_blob, int(attr["POSITION"]))
            if "indices" in primitive:
                flat = _accessor_array(
                    gltf, bin_blob, int(primitive["indices"])
                ).reshape(-1)
            else:
                flat = np.arange(len(pos), dtype=np.int32)
            faces = (
                _triangle_strip_to_faces(flat)
                if mode == GLTF_TRIANGLE_STRIP
                else flat.reshape((-1, 3)).astype(np.int32)
            )
            uv_sets = {}
            for key, accessor_idx in attr.items():
                if not key.startswith("TEXCOORD_"):
                    continue
                texcoord = int(key.removeprefix("TEXCOORD_"))
                uv = _accessor_array(gltf, bin_blob, int(accessor_idx)).astype(
                    np.float32
                )
                if uv.shape[1] != 2:
                    raise RuntimeError(f"Invalid {key} accessor shape: {uv.shape}")
                uv = uv.copy()
                uv[:, 1] = 1.0 - uv[:, 1]
                uv_sets[texcoord] = uv[faces]
            primitive_uvs[name] = uv_sets

            # glTF §3.7.2.1: use TANGENT accessor (VEC4 FLOAT) when present
            tangent_arr: Optional[np.ndarray] = None
            if "TANGENT" in attr:
                t = _accessor_array(gltf, bin_blob, int(attr["TANGENT"])).astype(
                    np.float32
                )
                if t.ndim == 2 and t.shape[1] == 4:
                    tangent_arr = t
            primitive_tangents[name] = tangent_arr

    return primitive_uvs, primitive_tangents


def _material_used_texcoords(material: NvPbrMaterial) -> set[int]:
    texcoords = set()
    if material.base_color_texture is not None:
        texcoords.add(material.base_color_texcoord)
    if material.metallic_texture is not None:
        texcoords.add(material.metallic_texcoord)
    if material.roughness_texture is not None:
        texcoords.add(material.roughness_texcoord)
    if material.emissive_texture is not None:
        texcoords.add(material.emissive_texcoord)
    if material.alpha_texture is not None:
        texcoords.add(material.alpha_texcoord)
    if material.normal_texture is not None:
        texcoords.add(material.normal_texcoord)
    if material.occlusion_texture is not None:
        texcoords.add(material.occlusion_texcoord)
    return texcoords


def _disable_texture_slots_without_uv(
    material: NvPbrMaterial,
    available_texcoords: set[int],
) -> NvPbrMaterial:
    updates = {}
    for slot in (
        "base_color",
        "metallic",
        "roughness",
        "emissive",
        "alpha",
        "normal",
        "occlusion",
    ):
        if getattr(material, f"{slot}_texture") is None:
            continue
        if getattr(material, f"{slot}_texcoord") not in available_texcoords:
            updates[f"{slot}_texture"] = None
    return replace(material, **updates) if updates else material


def _generated_face_uvs(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """Blender-compatible fallback using local bounding-box XY."""
    xy = vertices[:, :2].astype(np.float32)
    lo = xy.min(axis=0)
    extent = xy.max(axis=0) - lo
    uv = np.divide(
        xy - lo,
        extent,
        out=np.full_like(xy, 0.5, dtype=np.float32),
        where=extent > 1e-8,
    )
    return uv[faces]


def _texture_sampler_info(
    texture_ref: object,
    gltf: Optional[dict],
) -> tuple[str, tuple[str, str], int, tuple[float, float], tuple[float, float], float]:
    """Return sampling metadata for a glTF texture reference."""
    if texture_ref is None or gltf is None:
        return "linear-mipmap-linear", ("wrap", "wrap"), 0, (0.0, 0.0), (1.0, 1.0), 0.0
    if not isinstance(texture_ref, dict):
        raise RuntimeError(f"Unsupported glTF texture reference: {texture_ref!r}")

    transform = texture_ref.get("extensions", {}).get("KHR_texture_transform", {})
    texcoord = int(transform.get("texCoord", texture_ref.get("texCoord", 0)))
    offset = tuple(float(v) for v in transform.get("offset", [0.0, 0.0]))
    scale = tuple(float(v) for v in transform.get("scale", [1.0, 1.0]))
    rotation = float(transform.get("rotation", 0.0))
    if len(offset) != 2 or len(scale) != 2:
        raise RuntimeError(
            f"Invalid KHR_texture_transform on texture reference: {texture_ref!r}"
        )
    textures = gltf.get("textures", [])
    tex_idx = texture_ref.get("index")
    if tex_idx is None:
        return (
            "linear-mipmap-linear",
            ("wrap", "wrap"),
            texcoord,
            offset,
            scale,
            rotation,
        )
    texture = textures[int(tex_idx)]
    sampler = (
        gltf.get("samplers", [])[int(texture.get("sampler", -1))]
        if "sampler" in texture
        else {}
    )

    wrap_s = int(sampler.get("wrapS", GLTF_REPEAT))
    wrap_t = int(sampler.get("wrapT", GLTF_REPEAT))
    boundary_map = {
        GLTF_REPEAT: "wrap",
        GLTF_CLAMP_TO_EDGE: "clamp",
        GLTF_MIRRORED_REPEAT: "mirror",
    }
    boundary_s = boundary_map.get(wrap_s)
    boundary_t = boundary_map.get(wrap_t)
    if boundary_s is None or boundary_t is None:
        raise RuntimeError(f"Unsupported glTF texture wrap mode: {(wrap_s, wrap_t)}")

    min_filter = int(sampler.get("minFilter", GLTF_LINEAR))
    mag_filter = int(sampler.get("magFilter", GLTF_LINEAR))
    if min_filter == GLTF_NEAREST and mag_filter == GLTF_NEAREST:
        filter_mode = "nearest"
    elif min_filter in (GLTF_NEAREST, GLTF_LINEAR):
        filter_mode = "linear" if mag_filter == GLTF_LINEAR else "nearest"
    else:
        filter_mode = (
            "linear-mipmap-linear"
            if mag_filter == GLTF_LINEAR
            else "linear-mipmap-nearest"
        )
    return filter_mode, (boundary_s, boundary_t), texcoord, offset, scale, rotation


def _prepare_texture_uv(
    uv: torch.Tensor,
    boundary_mode: tuple[str, str] | str,
    offset: tuple[float, float] = (0.0, 0.0),
    scale: tuple[float, float] = (1.0, 1.0),
    rotation: float = 0.0,
) -> tuple[torch.Tensor, str]:
    """Apply per-axis glTF wrap modes before nvdiffrast texture sampling."""
    uv = torch.stack([uv[..., 0], 1.0 - uv[..., 1]], dim=-1)
    if offset != (0.0, 0.0) or scale != (1.0, 1.0) or rotation != 0.0:
        transformed = uv * uv.new_tensor(scale)
        if rotation != 0.0:
            c = math.cos(rotation)
            s = math.sin(rotation)
            transformed = torch.stack(
                [
                    c * transformed[..., 0] - s * transformed[..., 1],
                    s * transformed[..., 0] + c * transformed[..., 1],
                ],
                dim=-1,
            )
        transformed = transformed + uv.new_tensor(offset)
        uv = transformed

    if isinstance(boundary_mode, str):
        return uv, boundary_mode
    wrap_s, wrap_t = boundary_mode
    if wrap_s == wrap_t and wrap_s in ("wrap", "clamp"):
        return uv, wrap_s

    def _apply(coord: torch.Tensor, mode: str) -> torch.Tensor:
        if mode == "wrap":
            return torch.remainder(coord, 1.0)
        if mode == "clamp":
            return coord.clamp(0.0, 1.0)
        if mode == "mirror":
            mirrored = torch.remainder(coord, 2.0)
            return torch.where(mirrored <= 1.0, mirrored, 2.0 - mirrored)
        raise RuntimeError(f"Unsupported texture boundary mode: {mode}")

    return torch.stack(
        [_apply(uv[..., 0], wrap_s), _apply(uv[..., 1], wrap_t)], dim=-1
    ), "clamp"


def _material_signature_from_json(mat: dict) -> tuple:
    pbr = mat.get("pbrMetallicRoughness", {})
    return (
        tuple(
            np.asarray(
                pbr.get("baseColorFactor", [1.0, 1.0, 1.0, 1.0]), dtype=np.float32
            ).round(6)
        ),
        round(float(pbr.get("metallicFactor", 1.0)), 6),
        round(float(pbr.get("roughnessFactor", 1.0)), 6),
        tuple(
            np.asarray(
                mat.get("emissiveFactor", [0.0, 0.0, 0.0]), dtype=np.float32
            ).round(6)
        ),
        mat.get("alphaMode", "OPAQUE"),
        round(float(mat.get("alphaCutoff", 0.5)), 6),
        bool(mat.get("doubleSided", False)),
        # texture presence — differentiates materials with identical factors but different textures
        "baseColorTexture" in pbr,
        "metallicRoughnessTexture" in pbr,
        "emissiveTexture" in mat,
        "normalTexture" in mat,
        "occlusionTexture" in mat,
    )


def _material_signature_from_trimesh(mat: object) -> tuple:
    bcf = getattr(mat, "baseColorFactor", None)
    if bcf is None:
        bcf = [1.0, 1.0, 1.0, 1.0]
    bcf_arr = np.asarray(bcf, dtype=np.float32).flatten()
    if len(bcf_arr) == 3:
        bcf_arr = np.concatenate([bcf_arr, np.array([1.0], dtype=np.float32)])
    metallic = getattr(mat, "metallicFactor", 1.0)
    roughness = getattr(mat, "roughnessFactor", 1.0)
    emissive = getattr(mat, "emissiveFactor", None)
    if emissive is None:
        emissive = [0.0, 0.0, 0.0]
    return (
        tuple(bcf_arr[:4].round(6)),
        round(float(1.0 if metallic is None else metallic), 6),
        round(float(1.0 if roughness is None else roughness), 6),
        tuple(np.asarray(emissive, dtype=np.float32).flatten()[:3].round(6)),
        getattr(mat, "alphaMode", None) or "OPAQUE",
        round(float(getattr(mat, "alphaCutoff", 0.5) or 0.5), 6),
        bool(getattr(mat, "doubleSided", False)),
        _get_texture_image(mat, "baseColorTexture") is not None,
        _get_texture_image(mat, "metallicRoughnessTexture") is not None,
        _get_texture_image(mat, "emissiveTexture") is not None,
        _get_texture_image(mat, "normalTexture") is not None,
        False,  # occlusionTexture not exposed by trimesh
    )


def _match_gltf_material(mat: object, gltf: Optional[dict]) -> Optional[dict]:
    if mat is None or gltf is None:
        return None
    materials = gltf.get("materials", [])
    name = getattr(mat, "name", None)
    if name:
        matches = [m for m in materials if m.get("name") == name]
        if len(matches) == 1:
            return matches[0]
        if len(matches) > 1:
            raise RuntimeError(f"Multiple glTF materials named {name!r}")

    signature = _material_signature_from_trimesh(mat)
    matches = [m for m in materials if _material_signature_from_json(m) == signature]
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        raise RuntimeError("Could not uniquely map trimesh material to glTF material")
    return None


def _load_texture(
    pil_img: Optional[Image.Image],
    mode: str,
    device: str,
) -> Optional[torch.Tensor]:
    """Load a PIL image as a float32 GPU tensor (H, W, C).

    Resizes to the nearest power-of-two dimensions so nvdiffrast
    ``dr.texture`` with mipmap filtering does not fail on odd extents.
    """
    if pil_img is None:
        return None
    arr = np.array(pil_img.convert(mode), dtype=np.float32) / 255.0
    tex = torch.from_numpy(arr).to(device)

    h, w = tex.shape[:2]
    th = 1 << (h - 1).bit_length()
    tw = 1 << (w - 1).bit_length()
    if h != th or w != tw:
        if tex.dim() == 2:
            t = tex.unsqueeze(0).unsqueeze(0)
            t = torch.nn.functional.interpolate(
                t, size=(th, tw), mode="bilinear", align_corners=False
            )
            tex = t.squeeze(0).squeeze(0).contiguous()
        else:
            t = tex.permute(2, 0, 1).unsqueeze(0)
            t = torch.nn.functional.interpolate(
                t, size=(th, tw), mode="bilinear", align_corners=False
            )
            tex = t.squeeze(0).permute(1, 2, 0).contiguous()
    return tex


def _normalize_gltf_colors(colors: np.ndarray) -> np.ndarray:
    """Normalize glTF COLOR_0 values to linear RGBA in [0, 1]."""
    colors = np.asarray(colors, dtype=np.float32)
    if colors.ndim != 2 or colors.shape[1] < 3:
        raise ValueError(f"Expected COLOR_0 with shape (N, 3|4), got {colors.shape}")
    if colors.max(initial=0.0) > 1.0:
        colors = colors / 255.0
    colors = colors[:, :4].clip(0.0, 1.0)
    if colors.shape[1] == 3:
        alpha = np.ones((colors.shape[0], 1), dtype=np.float32)
        colors = np.concatenate([colors, alpha], axis=1)
    return colors.astype(np.float32, copy=False)


def _try_get_face_vertex_colors(
    geom: trimesh.Trimesh,
    faces_local: np.ndarray,
) -> Optional[np.ndarray]:
    """Extract per-face-corner glTF COLOR_0 as linear RGBA, or None."""
    vis = getattr(geom, "visual", None)
    if vis is None:
        return None

    attrs = getattr(vis, "vertex_attributes", None)
    if attrs is not None and "color" in attrs:
        vc = np.asarray(attrs["color"])
        if vc.ndim == 2 and vc.shape[0] == len(geom.vertices) and vc.shape[1] >= 3:
            return _normalize_gltf_colors(vc)[faces_local]

    kind = getattr(vis, "kind", None)
    if kind == "vertex":
        vc = getattr(vis, "vertex_colors", None)
        if vc is None:
            return None
        vc = np.asarray(vc)
        if vc.ndim != 2 or vc.shape[0] != len(geom.vertices) or vc.shape[1] < 3:
            return None
        return _normalize_gltf_colors(vc)[faces_local]

    if kind == "face":
        fc = getattr(vis, "face_colors", None)
        if fc is None:
            return None
        fc = np.asarray(fc)
        if fc.ndim != 2 or fc.shape[0] != len(geom.faces) or fc.shape[1] < 3:
            return None
        return np.repeat(_normalize_gltf_colors(fc)[:, None, :], 3, axis=1)

    return None


def _compute_face_tangents(
    triangles: np.ndarray,
    face_normals: np.ndarray,
    face_uvs: np.ndarray,
) -> np.ndarray:
    """Generate per-face-corner tangent frames from positions and UVs."""
    edge1 = triangles[:, 1] - triangles[:, 0]
    edge2 = triangles[:, 2] - triangles[:, 0]
    duv1 = face_uvs[:, 1] - face_uvs[:, 0]
    duv2 = face_uvs[:, 2] - face_uvs[:, 0]

    det = duv1[:, 0] * duv2[:, 1] - duv2[:, 0] * duv1[:, 1]
    valid = np.abs(det) > 1e-8

    tangent = np.zeros_like(edge1, dtype=np.float32)
    bitangent = np.zeros_like(edge1, dtype=np.float32)
    inv_det = np.zeros_like(det, dtype=np.float32)
    inv_det[valid] = 1.0 / det[valid]
    tangent[valid] = (
        edge1[valid] * duv2[valid, 1:2] - edge2[valid] * duv1[valid, 1:2]
    ) * inv_det[valid, None]
    bitangent[valid] = (
        edge2[valid] * duv1[valid, 0:1] - edge1[valid] * duv2[valid, 0:1]
    ) * inv_det[valid, None]

    n = face_normals.mean(axis=1)
    n_norm = np.linalg.norm(n, axis=1, keepdims=True)
    n = np.divide(n, n_norm, out=np.zeros_like(n), where=n_norm > 1e-8)

    fallback_axis = np.tile(np.array([[1.0, 0.0, 0.0]], dtype=np.float32), (len(n), 1))
    y_axis = np.tile(np.array([[0.0, 1.0, 0.0]], dtype=np.float32), (len(n), 1))
    fallback_axis[np.abs(n[:, 0]) > 0.9] = y_axis[np.abs(n[:, 0]) > 0.9]
    fallback = fallback_axis - n * (fallback_axis * n).sum(axis=1, keepdims=True)

    tangent = tangent - n * (tangent * n).sum(axis=1, keepdims=True)
    t_norm = np.linalg.norm(tangent, axis=1, keepdims=True)
    tangent = np.divide(tangent, t_norm, out=fallback, where=t_norm > 1e-8)
    tangent_norm = np.linalg.norm(tangent, axis=1, keepdims=True)
    tangent = np.divide(
        tangent,
        tangent_norm,
        out=np.tile(np.array([[1.0, 0.0, 0.0]], dtype=np.float32), (len(n), 1)),
        where=tangent_norm > 1e-8,
    )

    sign = np.sign((np.cross(n, tangent) * bitangent).sum(axis=1, keepdims=True))
    sign[sign == 0.0] = 1.0
    tangent4 = np.concatenate([tangent, sign.astype(np.float32)], axis=1)
    return np.repeat(tangent4[:, None, :], 3, axis=1)


def _material_key(geom: trimesh.Trimesh) -> str:
    mat = getattr(geom.visual, "material", None)
    if mat is not None:
        return f"__mat_{id(mat)}"
    return f"__geom_{id(geom)}"


def _material_label(geom: trimesh.Trimesh) -> str:
    mat = getattr(geom.visual, "material", None)
    if mat is None:
        return "<none>"
    return getattr(mat, "name", None) or f"material@{id(mat)}"


def _material_has_texture(geom: trimesh.Trimesh) -> bool:
    mat = getattr(geom.visual, "material", None)
    if mat is None:
        return False
    return any(
        _get_texture_image(mat, name) is not None
        for name in (
            "baseColorTexture",
            "metallicRoughnessTexture",
            "emissiveTexture",
            "normalTexture",
        )
    )


# ---------------------------------------------------------------------------
# Material extraction
# ---------------------------------------------------------------------------


def _extract_material(
    geom: trimesh.Trimesh,
    device: str,
    gltf: Optional[dict] = None,
    use_transmission: bool = False,
) -> NvPbrMaterial:
    """Extract full PBR material from a trimesh geometry."""
    mat = getattr(geom.visual, "material", None)
    if mat is None:
        return NvPbrMaterial()
    gltf_mat = _match_gltf_material(mat, gltf)
    gltf_pbr = gltf_mat.get("pbrMetallicRoughness", {}) if gltf_mat is not None else {}

    # --- base color factor ---
    bc_factor = torch.ones(4, dtype=torch.float32)
    if hasattr(mat, "baseColorFactor") and mat.baseColorFactor is not None:
        bcf = np.array(mat.baseColorFactor, dtype=np.float32).flatten()
        if bcf.max() > 1.0:
            bcf = bcf / 255.0
        if len(bcf) >= 3:
            bc_factor[:3] = torch.tensor(bcf[:3], dtype=torch.float32)
        if len(bcf) >= 4:
            bc_factor[3] = float(bcf[3])

    # --- base color texture (sRGB) + alpha channel ---
    bc_tex = None
    alpha_tex = None
    (
        bc_filter,
        bc_boundary,
        bc_texcoord,
        bc_uv_offset,
        bc_uv_scale,
        bc_uv_rotation,
    ) = _texture_sampler_info(
        gltf_pbr.get("baseColorTexture"),
        gltf,
    )
    bc_img = _get_texture_image(mat, "baseColorTexture")
    if bc_img is not None:
        rgba = np.array(bc_img.convert("RGBA"), dtype=np.float32) / 255.0
        bc_tex = torch.from_numpy(rgba[..., :3]).to(device)
        if rgba.shape[-1] == 4:
            alpha_tex = torch.from_numpy(rgba[..., 3:4]).to(device)

    # --- metallic / roughness (from ORM texture) ---
    metallic_tex = None
    roughness_tex = None
    metallic_factor = getattr(mat, "metallicFactor", 1.0)
    if metallic_factor is None:
        metallic_factor = 1.0
    roughness_factor = getattr(mat, "roughnessFactor", 1.0)
    if roughness_factor is None:
        roughness_factor = 1.0

    mr_img = _get_texture_image(mat, "metallicRoughnessTexture")
    (
        mr_filter,
        mr_boundary,
        mr_texcoord,
        mr_uv_offset,
        mr_uv_scale,
        mr_uv_rotation,
    ) = _texture_sampler_info(
        gltf_pbr.get("metallicRoughnessTexture"),
        gltf,
    )
    if mr_img is not None:
        mr_arr = np.array(mr_img.convert("RGB"), dtype=np.float32) / 255.0
        # glTF: B = metallic, G = roughness
        metallic_tex = torch.from_numpy(mr_arr[..., 2:3]).to(device)
        roughness_tex = torch.from_numpy(mr_arr[..., 1:2]).to(device)

    # --- emissive ---
    emissive_factor = torch.zeros(3, dtype=torch.float32)
    em_raw = getattr(mat, "emissiveFactor", None)
    if em_raw is not None:
        emf = np.array(em_raw, dtype=np.float32).flatten()
        if len(emf) >= 3:
            emissive_factor = torch.tensor(emf[:3], dtype=torch.float32)

    (
        em_filter,
        em_boundary,
        em_texcoord,
        em_uv_offset,
        em_uv_scale,
        em_uv_rotation,
    ) = _texture_sampler_info(
        gltf_mat.get("emissiveTexture") if gltf_mat is not None else None,
        gltf,
    )
    emissive_tex = _load_texture(
        _get_texture_image(mat, "emissiveTexture"), "RGB", device
    )

    # --- normal map ---
    normal_tex_ref = gltf_mat.get("normalTexture") if gltf_mat is not None else None
    (
        normal_filter,
        normal_boundary,
        normal_texcoord,
        normal_uv_offset,
        normal_uv_scale,
        normal_uv_rotation,
    ) = _texture_sampler_info(normal_tex_ref, gltf)
    normal_tex = _load_texture(_get_texture_image(mat, "normalTexture"), "RGB", device)
    # glTF §5.22: scale multiplies XY of the decoded tangent-space normal
    normal_scale = (
        float((normal_tex_ref or {}).get("scale", 1.0)) if gltf_mat is not None else 1.0
    )

    # --- occlusion ---
    occ_ref = gltf_mat.get("occlusionTexture") if gltf_mat is not None else None
    (
        occ_filter,
        occ_boundary,
        occ_texcoord,
        occ_uv_offset,
        occ_uv_scale,
        occ_uv_rotation,
    ) = _texture_sampler_info(occ_ref, gltf)
    occ_img = (
        _get_texture_image(mat, "occlusionTexture") if gltf_mat is not None else None
    )
    if occ_img is None and occ_ref is not None:
        # trimesh may not expose occlusionTexture; read from gltf directly
        tex_idx = occ_ref.get("index")
        if tex_idx is not None:
            src_idx = (
                gltf.get("textures", [])[int(tex_idx)].get("source") if gltf else None
            )
            if src_idx is not None:
                images = (gltf or {}).get("images", [])
                if src_idx < len(images):
                    # best-effort: textures embedded in GLB are already loaded by trimesh
                    pass
    occ_tex: Optional[torch.Tensor] = None
    if occ_img is not None:
        occ_arr = np.array(occ_img.convert("RGB"), dtype=np.float32) / 255.0
        occ_tex = torch.from_numpy(occ_arr[..., 0:1]).to(device)
    occ_strength = (
        float((occ_ref or {}).get("strength", 1.0)) if gltf_mat is not None else 1.0
    )

    # --- alpha mode ---
    alpha_mode = ALPHA_OPAQUE
    alpha_cutoff = 0.5
    am = getattr(mat, "alphaMode", None)
    if am == "MASK":
        alpha_mode = ALPHA_MASK
        alpha_cutoff = getattr(mat, "alphaCutoff", 0.5) or 0.5
    elif am == "BLEND":
        alpha_mode = ALPHA_BLEND

    double_sided = bool(getattr(mat, "doubleSided", False))

    is_unlit = (
        "KHR_materials_unlit" in (gltf_mat.get("extensions") or {})
        if gltf_mat is not None
        else False
    )

    # --- KHR_materials_transmission (opt-in via use_transmission) ---
    # Approximation: effective_alpha = base_alpha * (1 - transmissionFactor * tex.R)
    # Forces alphaMode=BLEND so the depth-peeler composites transmissive layers correctly.
    transmission_factor = 0.0
    transmission_tex: Optional[torch.Tensor] = None
    tr_filter, tr_boundary, tr_texcoord, tr_uv_offset, tr_uv_scale, tr_uv_rotation = (
        "linear-mipmap-linear",
        ("wrap", "wrap"),
        0,
        (0.0, 0.0),
        (1.0, 1.0),
        0.0,
    )
    if use_transmission and gltf_mat is not None:
        tr_ext = (gltf_mat.get("extensions") or {}).get(
            "KHR_materials_transmission", {}
        )
        if tr_ext:
            transmission_factor = float(tr_ext.get("transmissionFactor", 0.0))
            tr_tex_ref = tr_ext.get("transmissionTexture")
            if tr_tex_ref is not None and gltf is not None:
                (
                    tr_filter,
                    tr_boundary,
                    tr_texcoord,
                    tr_uv_offset,
                    tr_uv_scale,
                    tr_uv_rotation,
                ) = _texture_sampler_info(tr_tex_ref, gltf)
            if transmission_factor > 0.0:
                alpha_mode = ALPHA_BLEND

    if gltf_mat is not None:
        _UNSUPPORTED_KHR = {
            "KHR_materials_specular",
            "KHR_materials_clearcoat",
            "KHR_materials_ior",
            "KHR_materials_sheen",
            "KHR_materials_volume",
            "KHR_materials_emissive_strength",
        }
        present = _UNSUPPORTED_KHR & set((gltf_mat.get("extensions") or {}).keys())
        if present:
            mat_name = gltf_mat.get("name", "<unnamed>")
            warnings.warn(
                f"Material {mat_name!r} uses unsupported extensions {sorted(present)}; "
                "their contribution will be ignored.",
                RuntimeWarning,
                stacklevel=2,
            )

    return NvPbrMaterial(
        base_color_texture=bc_tex,
        base_color_filter_mode=bc_filter,
        base_color_boundary_mode=bc_boundary,
        base_color_texcoord=bc_texcoord,
        base_color_uv_offset=bc_uv_offset,
        base_color_uv_scale=bc_uv_scale,
        base_color_uv_rotation=bc_uv_rotation,
        base_color_factor=bc_factor,
        metallic_texture=metallic_tex,
        metallic_filter_mode=mr_filter,
        metallic_boundary_mode=mr_boundary,
        metallic_texcoord=mr_texcoord,
        metallic_uv_offset=mr_uv_offset,
        metallic_uv_scale=mr_uv_scale,
        metallic_uv_rotation=mr_uv_rotation,
        metallic_factor=float(metallic_factor),
        roughness_texture=roughness_tex,
        roughness_filter_mode=mr_filter,
        roughness_boundary_mode=mr_boundary,
        roughness_texcoord=mr_texcoord,
        roughness_uv_offset=mr_uv_offset,
        roughness_uv_scale=mr_uv_scale,
        roughness_uv_rotation=mr_uv_rotation,
        roughness_factor=float(roughness_factor),
        emissive_texture=emissive_tex,
        emissive_filter_mode=em_filter,
        emissive_boundary_mode=em_boundary,
        emissive_texcoord=em_texcoord,
        emissive_uv_offset=em_uv_offset,
        emissive_uv_scale=em_uv_scale,
        emissive_uv_rotation=em_uv_rotation,
        emissive_factor=emissive_factor,
        normal_texture=normal_tex,
        normal_filter_mode=normal_filter,
        normal_boundary_mode=normal_boundary,
        normal_texcoord=normal_texcoord,
        normal_uv_offset=normal_uv_offset,
        normal_uv_scale=normal_uv_scale,
        normal_uv_rotation=normal_uv_rotation,
        normal_scale=normal_scale,
        occlusion_texture=occ_tex,
        occlusion_filter_mode=occ_filter,
        occlusion_boundary_mode=occ_boundary,
        occlusion_texcoord=occ_texcoord,
        occlusion_uv_offset=occ_uv_offset,
        occlusion_uv_scale=occ_uv_scale,
        occlusion_uv_rotation=occ_uv_rotation,
        occlusion_strength=occ_strength,
        alpha_texture=alpha_tex,
        alpha_filter_mode=bc_filter,
        alpha_boundary_mode=bc_boundary,
        alpha_texcoord=bc_texcoord,
        alpha_uv_offset=bc_uv_offset,
        alpha_uv_scale=bc_uv_scale,
        alpha_uv_rotation=bc_uv_rotation,
        alpha_factor=float(bc_factor[3]),
        alpha_mode=alpha_mode,
        alpha_cutoff=alpha_cutoff,
        double_sided=double_sided,
        is_unlit=is_unlit,
        transmission_factor=transmission_factor,
        transmission_texture=transmission_tex,
        transmission_filter_mode=tr_filter,
        transmission_boundary_mode=tr_boundary,
        transmission_texcoord=tr_texcoord,
        transmission_uv_offset=tr_uv_offset,
        transmission_uv_scale=tr_uv_scale,
        transmission_uv_rotation=tr_uv_rotation,
    )


# ---------------------------------------------------------------------------
# GLB loader
# ---------------------------------------------------------------------------


def load_glb(
    glb_path: Union[str, Path],
    device: str = "cuda",
    missing_uv_policy: str = "blender",
    use_transmission: bool = False,
) -> NvMesh:
    """Load a GLB file and return an :class:`NvMesh` with full PBR materials.

    UV v-coordinates are flipped (``v' = 1 - v``) to convert from glTF's
    top-left origin to nvdiffrast ``dr.texture``'s bottom-left (OpenGL)
    convention.
    """
    if missing_uv_policy not in ("blender", "strict", "error"):
        raise ValueError(
            "missing_uv_policy must be one of: 'blender', 'strict', 'error'"
        )

    glb_path = Path(glb_path)
    gltf, bin_blob = _read_glb_chunks(glb_path)
    primitive_uv_sets, primitive_tangents = (
        _collect_gltf_primitive_data(gltf, bin_blob) if bin_blob else ({}, {})
    )
    scene = trimesh.load(str(glb_path), force="scene")
    if not isinstance(scene, trimesh.Scene):
        raise RuntimeError(f"Expected a Scene from {glb_path}")

    all_vertices: list[np.ndarray] = []
    all_faces: list[np.ndarray] = []
    all_triangles: list[np.ndarray] = []
    all_normals: list[np.ndarray] = []
    all_tangents: list[np.ndarray] = []
    all_uv_sets: dict[int, list[np.ndarray]] = {}
    all_mat_ids: list[np.ndarray] = []
    all_vc_colors: list[np.ndarray] = []
    all_vc_mask: list[np.ndarray] = []
    materials: list[NvPbrMaterial] = []
    base_materials: dict[str, NvPbrMaterial] = {}
    mat_name_to_id: dict[str, int] = {}
    vertex_offset = 0
    total_faces = 0
    has_any_vc = False

    for node_name in scene.graph.nodes_geometry:
        try:
            transform, geom_name = scene.graph[node_name]
        except (ValueError, KeyError):
            continue
        if geom_name not in scene.geometry:
            continue
        geom = scene.geometry[geom_name]
        if not isinstance(geom, trimesh.Trimesh) or len(geom.faces) == 0:
            continue

        verts_local = np.array(geom.vertices, dtype=np.float32)
        verts = verts_local.copy()
        faces_local = np.array(geom.faces, dtype=np.int32)

        # Apply node transform
        if transform is not None:
            ones = np.ones((verts.shape[0], 1), dtype=np.float32)
            verts = (
                np.hstack([verts, ones]) @ np.array(transform, dtype=np.float32).T
            )[:, :3]

        rotation_scale = (
            np.array(transform[:3, :3], dtype=np.float32)
            if transform is not None
            else np.eye(3, dtype=np.float32)
        )
        flipped_winding = np.linalg.det(rotation_scale) < 0
        if flipped_winding:
            faces_local = faces_local[:, ::-1]

        # Triangles (world-space vertex positions per face)
        triangles = verts[faces_local]  # (F, 3, 3)

        # Vertex normals in world space
        # glTF §3.7.3: normals transform by inverse-transpose of the model matrix
        try:
            normal_matrix = np.linalg.inv(rotation_scale).T
        except np.linalg.LinAlgError:
            normal_matrix = rotation_scale
        normals_world = np.dot(geom.vertex_normals, normal_matrix)
        norms = np.linalg.norm(normals_world, axis=1, keepdims=True)
        normals_world = np.divide(
            normals_world, norms, out=np.zeros_like(normals_world), where=norms > 1e-6
        )
        face_normals = normals_world[faces_local]  # (F, 3, 3)

        # Material
        mat_key = _material_key(geom)
        mat_label = _material_label(geom)
        if mat_key not in base_materials:
            base_materials[mat_key] = _extract_material(
                geom, device, gltf=gltf, use_transmission=use_transmission
            )
        base_material = base_materials[mat_key]
        required_texcoords = _material_used_texcoords(base_material)

        # Trimesh's glTF loader already flips TEXCOORD_0 from glTF's UV
        # convention to OpenGL-style UVs for TextureVisuals.
        n_faces = faces_local.shape[0]
        raw_uv_sets = primitive_uv_sets.get(geom_name, {})
        available_texcoords = set(raw_uv_sets)
        if hasattr(geom.visual, "uv") and geom.visual.uv is not None:
            available_texcoords.add(0)
        missing_texcoords = required_texcoords - available_texcoords
        if missing_texcoords:
            if missing_uv_policy == "error":
                raise RuntimeError(
                    f"Geometry {geom_name} with material {mat_label} requires "
                    f"missing TEXCOORD sets {sorted(missing_texcoords)}"
                )
            if missing_uv_policy == "blender":
                raw_uv_sets = dict(raw_uv_sets)
                generated_uvs = _generated_face_uvs(verts_local, faces_local)
                for texcoord in missing_texcoords:
                    raw_uv_sets[texcoord] = generated_uvs
                    available_texcoords.add(texcoord)
                warnings.warn(
                    f"Geometry {geom_name} with material {mat_label} has textures "
                    f"requiring missing TEXCOORD sets {sorted(missing_texcoords)}; "
                    "using generated local XY UVs for Blender compatibility.",
                    RuntimeWarning,
                    stacklevel=2,
                )
            else:
                warnings.warn(
                    f"Geometry {geom_name} with material {mat_label} has textures "
                    f"requiring missing TEXCOORD sets {sorted(missing_texcoords)}; "
                    "texture slots using those coordinates were disabled.",
                    RuntimeWarning,
                    stacklevel=2,
                )
        material = (
            base_material
            if missing_uv_policy == "blender"
            else _disable_texture_slots_without_uv(base_material, available_texcoords)
        )
        material_key = f"{mat_key}|uv={tuple(sorted(available_texcoords))}"
        if material_key not in mat_name_to_id:
            mat_name_to_id[material_key] = len(materials)
            materials.append(material)
        mat_id = mat_name_to_id[material_key]
        used_texcoords = _material_used_texcoords(material)
        if hasattr(geom.visual, "uv") and geom.visual.uv is not None:
            uv_all = np.array(geom.visual.uv, dtype=np.float32)
            if uv_all.ndim != 2 or uv_all.shape[0] != len(geom.vertices):
                raise RuntimeError(
                    f"Invalid UV array for geometry {geom_name}: {uv_all.shape}"
                )
            face_uvs = uv_all[faces_local]
        elif 0 in raw_uv_sets:
            face_uvs = raw_uv_sets[0]
            if flipped_winding:
                face_uvs = face_uvs[:, ::-1]
        else:
            face_uvs = np.zeros((n_faces, 3, 2), dtype=np.float32)

        geom_uv_sets = {0: face_uvs}
        for texcoord in used_texcoords:
            if texcoord == 0:
                continue
            face_uv = raw_uv_sets.get(texcoord)
            if face_uv is None:
                raise RuntimeError(
                    f"Geometry {geom_name} with material {mat_label} requires "
                    f"TEXCOORD_{texcoord}"
                )
            if flipped_winding:
                face_uv = face_uv[:, ::-1]
            if face_uv.shape != (n_faces, 3, 2):
                raise RuntimeError(
                    f"Invalid TEXCOORD_{texcoord} face UV shape for {geom_name}: {face_uv.shape}"
                )
            geom_uv_sets[texcoord] = face_uv

        vc_rgba = _try_get_face_vertex_colors(geom, faces_local)
        if vc_rgba is not None:
            all_vc_colors.append(vc_rgba)
            all_vc_mask.append(np.ones(n_faces, dtype=bool))
            has_any_vc = True
        else:
            all_vc_colors.append(np.ones((n_faces, 3, 4), dtype=np.float32))
            all_vc_mask.append(np.zeros(n_faces, dtype=bool))

        # Use mesh-provided TANGENT accessor when available (glTF §3.7.2.1),
        # otherwise compute analytically from positions and UVs.
        mesh_tangents = primitive_tangents.get(geom_name)
        if mesh_tangents is not None and len(mesh_tangents) == len(verts_local):
            per_face_t = mesh_tangents[faces_local].astype(np.float32)  # (F, 3, 4)
            if flipped_winding:
                per_face_t = per_face_t[:, ::-1, :].copy()
                per_face_t[..., 3] *= -1  # flip handedness with winding
            tangents = per_face_t
        else:
            tangent_uvs = (
                geom_uv_sets.get(material.normal_texcoord, face_uvs)
                if material.normal_texture is not None
                else face_uvs
            )
            tangents = _compute_face_tangents(triangles, face_normals, tangent_uvs)

        all_vertices.append(verts)
        all_faces.append(faces_local + vertex_offset)
        all_triangles.append(triangles)
        all_normals.append(face_normals)
        all_tangents.append(tangents)
        for texcoord in list(all_uv_sets):
            if texcoord not in geom_uv_sets:
                geom_uv_sets[texcoord] = np.zeros((n_faces, 3, 2), dtype=np.float32)
        for texcoord, face_uv in geom_uv_sets.items():
            if texcoord not in all_uv_sets:
                all_uv_sets[texcoord] = [
                    np.zeros((total_faces, 3, 2), dtype=np.float32)
                ]
            all_uv_sets[texcoord].append(face_uv)
        all_mat_ids.append(np.full(n_faces, mat_id, dtype=np.int32))
        vertex_offset += verts.shape[0]
        total_faces += n_faces

    if not all_vertices:
        raise RuntimeError(f"No triangle meshes found in {glb_path}")

    fvc = None
    vc_mask = None
    if has_any_vc:
        fvc = torch.from_numpy(np.concatenate(all_vc_colors)).to(device).contiguous()
        vc_mask = torch.from_numpy(np.concatenate(all_vc_mask)).to(device).contiguous()

    uv_tensors = {
        texcoord: torch.from_numpy(np.concatenate(face_uvs)).to(device).contiguous()
        for texcoord, face_uvs in all_uv_sets.items()
    }

    return NvMesh(
        vertices=torch.from_numpy(np.concatenate(all_vertices)).to(device),
        faces=torch.from_numpy(np.concatenate(all_faces)).to(device),
        triangles=torch.from_numpy(np.concatenate(all_triangles))
        .to(device)
        .contiguous(),
        normals=torch.from_numpy(np.concatenate(all_normals)).to(device).contiguous(),
        tangents=torch.from_numpy(np.concatenate(all_tangents)).to(device).contiguous(),
        uv_coords=uv_tensors[0],
        material_ids=torch.from_numpy(np.concatenate(all_mat_ids))
        .to(device)
        .contiguous(),
        uv_sets=uv_tensors,
        materials=materials,
        face_vertex_colors=fvc,
        vertex_color_mask=vc_mask,
    )
