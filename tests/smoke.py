import ast
import json
from pathlib import Path
import tempfile
import xml.etree.ElementTree as ET

import numpy as np

from naro.cloud_io import load_cloud
from naro.map_tools import import_map
from naro.replay import run_case
from naro.ros_utils import cloud_message, read_xyz, rotation, set_pose


def main():
    root = Path(__file__).resolve().parents[1]
    for path in [*root.glob('*.py'), *root.glob('naro/**/*.py'), *root.glob('launch/*.py')]:
        ast.parse(path.read_text(encoding='utf-8'), filename=str(path))
    ET.parse(root / 'package.xml')
    from geometry_msgs.msg import Pose
    from builtin_interfaces.msg import Time
    pose = Pose()
    theta = 2.7
    matrix = np.eye(4)
    matrix[:2, :2] = [[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]]
    matrix[:3, 3] = [1., 2., 3.]
    set_pose(pose, matrix)
    np.testing.assert_allclose(rotation(pose.orientation), matrix[:3, :3], atol=1e-12)
    xyz = np.arange(360, dtype=np.float32).reshape(-1, 3) / 100.
    np.testing.assert_allclose(read_xyz(cloud_message(xyz, 'camera_init', Time())), xyz)
    with tempfile.TemporaryDirectory() as folder:
        folder = Path(folder)
        header = 'VERSION .7\nFIELDS x y z intensity\nSIZE 4 4 4 4\nTYPE F F F F\nCOUNT 1 1 1 1\nWIDTH 120\nHEIGHT 1\nPOINTS 120\nDATA binary\n'
        pcd = folder / 'map.pcd'
        pcd.write_bytes(header.encode() + np.column_stack([xyz, np.ones(len(xyz), np.float32)]).astype('<f4').tobytes())
        np.testing.assert_allclose(load_cloud(pcd), xyz)
        import_map(pcd, folder / 'site')
        with np.load(folder / 'site/map.npz') as grid:
            assert grid['free'].any() and not grid['observed_free'].any()
    row = run_case('courtyard', root / 'examples')
    assert row['verdict'] == 'correct', row['result'].get('reason')
    assert row['position_cm'] < 9., row['position_cm']
    print(json.dumps({key: value for key, value in row.items() if key != 'result'}, indent=2))


if __name__ == '__main__':
    main()
