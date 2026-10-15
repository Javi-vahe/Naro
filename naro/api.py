from pathlib import Path
from threading import RLock

import numpy as np

from .core import relocalize_structural as core

_lock = RLock()


def localize(points, map_dir, here=None, sensor_origin=None, threads=4):
    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or points.shape[1] < 3:
        raise ValueError('points must have shape N x 3')
    points = points[:, :3]
    points = points[np.isfinite(points).all(axis=1)]
    here = np.zeros(3) if here is None else np.asarray(here, dtype=float)
    if here.shape != (3,) or not np.isfinite(here).all():
        raise ValueError('here must contain three finite coordinates')
    if sensor_origin is not None:
        sensor_origin = np.asarray(sensor_origin, dtype=float)
        if sensor_origin.shape != (3,) or not np.isfinite(sensor_origin).all():
            raise ValueError('sensor_origin must contain three finite coordinates')
    if threads < 1:
        raise ValueError('threads must be positive')
    with _lock:
        return core.relocalize(points, str(Path(map_dir).resolve()), here,
                               threads=int(threads), sensor_origin=sensor_origin)


def prepare(map_dir, threads=4):
    map_dir = str(Path(map_dir).resolve())
    if threads < 1:
        raise ValueError('threads must be positive')
    with _lock:
        db, _, built = core.load_database(map_dir)
        _, structure_built = core.base.structural_database(map_dir, db, threads=threads)
        extra = core.prepare_database_extras(map_dir, threads=threads)
    return dict(database_built=built, structural_database_built=structure_built,
                candidates=len(db['positions']), target_points=len(db['target']), **extra)
