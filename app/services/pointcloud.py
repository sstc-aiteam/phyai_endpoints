import numpy as np


def encode_binary_ply(vertices: np.ndarray, colors: np.ndarray) -> bytes:
    """
    Encodes Nx3 xyz float32 vertices and Nx3 RGB uint8 colors as a binary little-endian PLY.
    """
    vertex_count = len(vertices)
    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        f"element vertex {vertex_count}\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
        "property uchar red\n"
        "property uchar green\n"
        "property uchar blue\n"
        "end_header\n"
    ).encode("ascii")

    ply_vertices = np.empty(
        vertex_count,
        dtype=[
            ("x", "<f4"),
            ("y", "<f4"),
            ("z", "<f4"),
            ("red", "u1"),
            ("green", "u1"),
            ("blue", "u1"),
        ],
    )
    ply_vertices["x"] = vertices[:, 0]
    ply_vertices["y"] = vertices[:, 1]
    ply_vertices["z"] = vertices[:, 2]
    ply_vertices["red"] = colors[:, 0]
    ply_vertices["green"] = colors[:, 1]
    ply_vertices["blue"] = colors[:, 2]

    return header + ply_vertices.tobytes()
