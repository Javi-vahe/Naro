import numpy as np


def rotation(q):
    q = np.asarray([q.x, q.y, q.z, q.w], dtype=float)
    norm = np.linalg.norm(q)
    if not np.isfinite(norm) or norm < 1e-8:
        raise ValueError('invalid odometry quaternion')
    x, y, z, w = q / norm
    return np.array([
        [1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
        [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
        [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)],
    ])


def set_pose(message, matrix):
    r = np.asarray(matrix, dtype=float)[:3, :3]
    k = int(np.argmax([1+r[0, 0]+r[1, 1]+r[2, 2],
                       1+r[0, 0]-r[1, 1]-r[2, 2],
                       1-r[0, 0]+r[1, 1]-r[2, 2],
                       1-r[0, 0]-r[1, 1]+r[2, 2]]))
    if k == 0:
        q = [r[2, 1]-r[1, 2], r[0, 2]-r[2, 0], r[1, 0]-r[0, 1], 1+np.trace(r)]
    elif k == 1:
        q = [1+r[0, 0]-r[1, 1]-r[2, 2], r[0, 1]+r[1, 0], r[0, 2]+r[2, 0], r[2, 1]-r[1, 2]]
    elif k == 2:
        q = [r[0, 1]+r[1, 0], 1-r[0, 0]+r[1, 1]-r[2, 2], r[1, 2]+r[2, 1], r[0, 2]-r[2, 0]]
    else:
        q = [r[0, 2]+r[2, 0], r[1, 2]+r[2, 1], 1-r[0, 0]-r[1, 1]+r[2, 2], r[1, 0]-r[0, 1]]
    q = np.asarray(q) / np.linalg.norm(q)
    position = message.translation if hasattr(message, 'translation') else message.position
    orientation = message.rotation if hasattr(message, 'rotation') else message.orientation
    position.x, position.y, position.z = map(float, matrix[:3, 3])
    orientation.x, orientation.y, orientation.z, orientation.w = map(float, q)


def read_xyz(message):
    from sensor_msgs_py import point_cloud2
    data = point_cloud2.read_points(message, field_names=('x', 'y', 'z'), skip_nans=True)
    if isinstance(data, np.ndarray) and data.dtype.names:
        return np.column_stack([data[name].reshape(-1) for name in ('x', 'y', 'z')]).astype(float)
    return np.asarray(list(data), dtype=float).reshape(-1, 3)


def cloud_message(points, frame, stamp):
    from sensor_msgs_py.point_cloud2 import create_cloud_xyz32
    from std_msgs.msg import Header
    return create_cloud_xyz32(Header(frame_id=frame, stamp=stamp), np.asarray(points, dtype=np.float32))
