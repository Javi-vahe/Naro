import argparse
import json
import math
from pathlib import Path
import time

import numpy as np

from .api import localize
from .cloud_io import load_cloud
from .paths import load_case


def score(result, case):
    if not result.get('accepted'):
        return dict(verdict='rejected')
    pose = np.asarray(result['T_map_odom']) @ np.asarray(case['body_odom'])
    truth = np.asarray(case['body_map_truth'])
    error = float(np.linalg.norm(pose[:3, 3] - truth[:3, 3]))
    angle = math.atan2(pose[1, 0], pose[0, 0]) - math.atan2(truth[1, 0], truth[0, 0])
    yaw = abs(math.degrees(math.atan2(math.sin(angle), math.cos(angle))))
    return dict(verdict='correct' if error < .3 and yaw < 3. else 'false_accept',
                position_cm=error * 100., yaw_deg=yaw)


def run_case(name, root=None, threads=4, baseline=False):
    root, case = load_case(name, root)
    points = load_cloud(root / case['points'])
    begun = time.monotonic()
    if baseline:
        from .core import relocalize_base
        result = relocalize_base.relocalize(points, str(root / case['map']), case['here'], threads=threads)
    else:
        result = localize(points, root / case['map'], case['here'], case['sensor_origin'], threads)
    return dict(case=name, algorithm='baseline' if baseline else 'naro',
                seconds=time.monotonic() - begun, **score(result, case), result=result)


def main(argv=None):
    parser = argparse.ArgumentParser(description='Replay saved Gazebo startup scans')
    parser.add_argument('--case', choices=['office', 'courtyard', 'warehouse'], default='courtyard')
    parser.add_argument('--all', action='store_true')
    parser.add_argument('--baseline', action='store_true')
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--data', type=Path)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args(argv)
    records = []
    for name in ['office', 'courtyard', 'warehouse'] if args.all else [args.case]:
        for baseline in [False, True] if args.baseline else [False]:
            row = run_case(name, args.data, args.threads, baseline)
            records.append(row)
            summary = {key: value for key, value in row.items() if key != 'result'}
            summary['reason'] = row['result'].get('reason', '')
            print(json.dumps(summary, ensure_ascii=False, allow_nan=False), flush=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(records, ensure_ascii=False, allow_nan=False, indent=2) + '\n', encoding='utf-8')
    return 0 if all(row['verdict'] == 'correct' for row in records if row['algorithm'] == 'naro') else 2


if __name__ == '__main__':
    raise SystemExit(main())
