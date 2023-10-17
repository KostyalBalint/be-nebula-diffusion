import numpy as np

import downloader
import objaverse
import trimesh
import os

uids = objaverse.load_uids()
object_paths = downloader.load_object_paths()
print("Loaded object data")


def get_path_by_uid(uid):
    path = object_paths[uid][5:-4]
    return os.path.join("data", f'{path}.ply')


def get_point_cloud_by_uid(uid, scale=1):
    mesh = trimesh.load(get_path_by_uid(uid))

    # Center the mesh to (0, 0, 0)
    center = np.mean(mesh.vertices, axis=0)
    translation_vector = -center
    mesh.vertices += translation_vector

    # Scale the mesh inside a fixed sized bounding box
    extents = mesh.extents
    max_extent = max(extents)
    scaling_factor = (1.0 / max_extent) * scale
    mesh.apply_scale(scaling_factor)

    # Get the points, colors of the meshs
    colors = mesh.colors.tolist()
    points = mesh.vertices.tolist()
    for i in range(len(points)):
        points[i].extend(colors[i])
    return points
