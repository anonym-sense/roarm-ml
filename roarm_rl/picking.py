"""Mouse-ray math for click-and-drag picking in the PyBullet GUI.

Builds the world-space ray through a window pixel by inverting the debug
visualizer's projection * view matrices (both returned by
p.getDebugVisualizerCamera), which avoids depending on PyBullet's
horizon/vertical basis vectors whose scale is not a plain world-unit frustum.
"""

import numpy as np
import pybullet as p

FAR = 100.0  # ray length in meters; reaches well past the arm


def camera_ray(mouse_x, mouse_y, client_id):
    """World-space (ray_from, ray_to, cam_forward) through pixel (mouse_x, mouse_y)."""
    width, height, view, proj, _up, cam_fwd, _hor, _ver, _yaw, _pitch, _dist, _target = (
        p.getDebugVisualizerCamera(physicsClientId=client_id)
    )
    view_m = np.array(view, dtype=float).reshape(4, 4).T
    proj_m = np.array(proj, dtype=float).reshape(4, 4).T
    inv = np.linalg.inv(proj_m @ view_m)

    ndc_x = 2.0 * mouse_x / float(width) - 1.0
    ndc_y = 1.0 - 2.0 * mouse_y / float(height)

    near_pt = inv @ np.array([ndc_x, ndc_y, -1.0, 1.0])
    far_pt = inv @ np.array([ndc_x, ndc_y, 1.0, 1.0])
    near_pt = near_pt[:3] / near_pt[3]
    far_pt = far_pt[:3] / far_pt[3]

    cam_pos = near_pt
    direction = far_pt - near_pt
    direction = direction / np.linalg.norm(direction)
    ray_to = cam_pos + direction * FAR
    return cam_pos.tolist(), ray_to.tolist(), list(cam_fwd)


def intersect_plane(ray_from, ray_to, plane_point, plane_normal):
    """Where the ray (ray_from -> ray_to) crosses the plane through
    plane_point with normal plane_normal. None if parallel."""
    ray_dir = [ray_to[i] - ray_from[i] for i in range(3)]
    denom = sum(ray_dir[i] * plane_normal[i] for i in range(3))
    if abs(denom) < 1e-9:
        return None
    numer = sum((plane_point[i] - ray_from[i]) * plane_normal[i] for i in range(3))
    t = numer / denom
    return [ray_from[i] + t * ray_dir[i] for i in range(3)]
