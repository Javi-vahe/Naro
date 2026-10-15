from pathlib import Path

import numpy as np


def load_cloud(path):
    path = Path(path)
    if path.suffix.lower() == '.npy':
        points = np.load(path, allow_pickle=False)
    elif path.suffix.lower() == '.npz':
        with np.load(path, allow_pickle=False) as data:
            points = data['points']
    elif path.suffix.lower() == '.pcd':
        points = read_pcd(path)
    else:
        raise ValueError('supported point clouds: .pcd, .npy, .npz')
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] < 3:
        raise ValueError('point cloud must have shape N x 3')
    points = points[:, :3]
    return np.ascontiguousarray(points[np.isfinite(points).all(axis=1)])


def read_pcd(path):
    header = {}
    with Path(path).open('rb') as handle:
        while True:
            raw = handle.readline()
            if not raw:
                raise ValueError('PCD header has no DATA field')
            line = raw.decode('ascii').strip()
            if not line or line.startswith('#'):
                continue
            key, *values = line.split()
            header[key.upper()] = values
            if key.upper() == 'DATA':
                break
        fields = header.get('FIELDS', header.get('FIELD', []))
        if not all(axis in fields for axis in ('x', 'y', 'z')):
            raise ValueError('PCD needs x, y and z fields')
        counts = [int(v) for v in header.get('COUNT', ['1'] * len(fields))]
        if len(counts) != len(fields) or min(counts) < 1:
            raise ValueError('invalid PCD COUNT fields')
        mode = header['DATA'][0].lower()
        if mode == 'ascii':
            data = np.loadtxt(handle, ndmin=2)
            offsets = np.cumsum([0] + counts[:-1])
            return data[:, [offsets[fields.index(axis)] for axis in ('x', 'y', 'z')]]
        if mode != 'binary':
            raise ValueError('save the PCD with DATA ascii or DATA binary')
        kinds = {'F': 'f', 'I': 'i', 'U': 'u'}
        sizes = [int(v) for v in header['SIZE']]
        types = header['TYPE']
        if len(sizes) != len(fields) or len(types) != len(fields):
            raise ValueError('invalid PCD field layout')
        dtype = np.dtype([(name, '<' + kinds[kind] + str(size), (count,))
                          for name, kind, size, count in zip(fields, types, sizes, counts)])
        count = int(header.get('POINTS', [str(int(header['WIDTH'][0]) * int(header.get('HEIGHT', ['1'])[0]))])[0])
        data = np.frombuffer(handle.read(), dtype=dtype, count=count)
        return np.column_stack([data[axis][:, 0] for axis in ('x', 'y', 'z')])
