import argparse
import json
from pathlib import Path

import numpy as np

from .api import prepare
from .cloud_io import load_cloud


def import_map(cloud, output, resolution=.25, min_height=.2, max_height=2.):
    if resolution <= 0 or min_height < 0 or max_height <= min_height:
        raise ValueError('invalid grid resolution or search heights')
    points = load_cloud(cloud)
    if len(points) < 100:
        raise ValueError('map needs at least 100 finite points')
    _, first = np.unique(np.floor(points / .1).astype(np.int64), axis=0, return_index=True)
    points = points[np.sort(first)]
    origin = np.floor((points.min(axis=0) - .5) / resolution) * resolution
    shape = np.ceil((points.max(axis=0) + .5 - origin) / resolution).astype(np.int64) + 1
    if np.prod(shape.astype(float)) > 25_000_000:
        raise ValueError('map grid exceeds 25 million cells; crop the map or increase --resolution')
    occupied = np.zeros(tuple(shape), dtype=bool)
    cells = np.floor((points - origin) / resolution).astype(np.int64)
    occupied[tuple(cells.T)] = True
    floor = float(np.quantile(points[:, 2], .05))
    z = origin[2] + (np.arange(shape[2]) + .5) * resolution
    heights = (z >= floor + min_height) & (z <= floor + max_height)
    free = ~occupied & heights[None, None, :]
    if not free.any():
        raise ValueError('no search cells within the requested height range')
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise ValueError('output directory must be empty')
    output.mkdir(parents=True, exist_ok=True)
    np.save(output / 'localization_cloud.npy', points.astype(np.float32))
    np.savez_compressed(output / 'map.npz', origin=origin, resolution=resolution,
                        occupied=occupied, free=free, observed_free=np.zeros_like(free))
    metadata = dict(source=Path(cloud).name, points=len(points), resolution=resolution,
                    search_height=[min_height, max_height], grid_shape=shape.tolist())
    (output / 'map.json').write_text(json.dumps(metadata, indent=2) + '\n', encoding='utf-8')
    return metadata


def main(argv=None):
    parser = argparse.ArgumentParser(description='Import a gravity-aligned point cloud map')
    parser.add_argument('--cloud', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--resolution', type=float, default=.25)
    parser.add_argument('--min-height', type=float, default=.2)
    parser.add_argument('--max-height', type=float, default=2.)
    parser.add_argument('--threads', type=int, default=4)
    args = parser.parse_args(argv)
    try:
        report = import_map(args.cloud, args.output, args.resolution, args.min_height, args.max_height)
        report.update(prepare(args.output, args.threads))
    except (OSError, ValueError, RuntimeError, ImportError) as error:
        parser.exit(1, str(error) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
