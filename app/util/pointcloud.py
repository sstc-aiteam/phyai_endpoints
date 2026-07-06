import numpy as np


def transform_camera_points_to_base(
    points: np.ndarray,
    T_cam_wrist: np.ndarray,
    R_gripper2base: np.ndarray,
    t_gripper2base_vec: np.ndarray,
) -> np.ndarray:
    """
    Transforms Nx3 points from camera coordinates into the robot base frame.
    """
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"Expected points with shape (N, 3), got {points.shape}")

    T_gripper2base = np.eye(4)
    T_gripper2base[:3, :3] = R_gripper2base
    T_gripper2base[:3, 3] = np.asarray(t_gripper2base_vec).reshape(3)

    T_cam2base = T_gripper2base @ T_cam_wrist
    points_homogeneous = np.column_stack((points, np.ones(len(points))))
    points_base = (T_cam2base @ points_homogeneous.T).T[:, :3]

    return points_base


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
