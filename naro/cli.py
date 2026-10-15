import argparse
import json
from pathlib import Path
import sys

from .api import localize, prepare
from .cloud_io import load_cloud


def main(argv=None):
    parser = argparse.ArgumentParser(description='Relocalize an odometry-frame scan')
    parser.add_argument('--map', required=True, type=Path)
    parser.add_argument('--points', type=Path)
    parser.add_argument('--here', nargs=3, type=float, default=[0., 0., 0.])
    parser.add_argument('--sensor-origin', nargs=3, type=float)
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--build', action='store_true')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args(argv)
    if not args.build and args.points is None:
        parser.error('--points is required for localization')
    try:
        result = (prepare(args.map, args.threads) if args.build else
                  localize(load_cloud(args.points), args.map, args.here,
                           args.sensor_origin, args.threads))
        text = json.dumps(result, ensure_ascii=False, allow_nan=False, indent=2)
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(text + '\n', encoding='utf-8')
        print(text)
        return 0 if args.build or result['accepted'] else 2
    except (OSError, ValueError, RuntimeError, ImportError) as error:
        print(str(error), file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
