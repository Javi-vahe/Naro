"""Original naro relocalization baseline."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time

import numpy as np

FIT_D, MIN_FITNESS, MAX_RMSE, MAX_TILT_DEG, MIN_MARGIN, MIN_POINTS = .2, .65, .09, 3., .12, 2000
DB_NAME, DB_VERSION = 'relocalization_db.npz', 2
SECTORS = 72                              # 5 degree yaw resolution
BANDS = ((-.55, -.05), (-.05, .55))       # height bands relative to the sensor (m)
RANGE, MIN_RANGE = 15., .3                # descriptor range (m); nearer points are the aircraft itself
XY_STEP, Z_STEP = .5, .5                  # candidate lattice (m); GICP absorbs the rest
TAU, MISSING = 1., .25                    # per-sector cost cap; cost when only the map sees something there
# Final two-sided check (consistency): right answers on the real lab survey <= 0.50; scans of another building
# that still reached fitness 0.66-0.70 scored 0.76-0.78 (sim2real/tests/bench_relocalize.py, 2026-10-08).
MAX_INCONSISTENCY = .6
TARGET_VOXEL, SOURCE_VOXEL = .1, .1


def voxel(points, size):
    keys = np.floor(points/size).astype(np.int64)
    _, index = np.unique(keys, axis=0, return_index=True)
    return points[np.sort(index)]


def map_cloud(map_dir):
    """Use the exported map installed on the aircraft; never read a development dataset."""
    with np.load(Path(map_dir)/'map.npz') as data:
        grid = {key: data[key] for key in data.files}
    scan = Path(map_dir)/'localization_cloud.npy'
    cloud = (np.load(scan)[:, :3].astype(float) if scan.is_file() else
             grid['origin']+(np.argwhere(grid['occupied'])+.5)*float(grid['resolution']))
    return voxel(cloud, TARGET_VOXEL), grid


def yaw_matrix(angle):
    c, s = math.cos(angle), math.sin(angle)
    T = np.eye(4)
    T[:2, :2] = [[c, -s], [s, c]]
    return T


def yaw_of(T):
    return math.atan2(T[1, 0], T[0, 0])


def tilt_deg(T):
    """Angle (deg) between the transform's z axis and the vertical: roll and pitch together."""
    return math.degrees(math.acos(float(np.clip(T[2, 2], -1., 1.))))


def different(a, b):
    yaw = abs(math.atan2(math.sin(yaw_of(a)-yaw_of(b)), math.cos(yaw_of(a)-yaw_of(b))))
    return np.linalg.norm(a[:3, 3]-b[:3, 3]) > .5 or yaw > math.radians(10.)


# ---------- descriptor ----------
def band_cells(points, z0):
    """Per height band around sensor height z0: the occupied 0.1 m cells in the horizontal plane (unique xy)."""
    out = []
    for low, high in BANDS:
        inside = points[(points[:, 2] >= z0+low) & (points[:, 2] < z0+high), :2]
        out.append(np.unique(np.floor(inside/.1).astype(np.int32), axis=0).astype(np.float32)*.1+.05)
    return out


def descriptor(cells, center):
    """(bands, sectors): horizontal range from center to the nearest occupied cell per sector, NaN when none."""
    out = np.full((len(BANDS), SECTORS), np.inf, np.float32)
    for b, xy in enumerate(cells):
        d = xy-np.asarray(center[:2], np.float32)
        r = np.hypot(d[:, 0], d[:, 1])
        keep = (r > MIN_RANGE) & (r < RANGE)
        if keep.any():
            sector = (np.floor(np.arctan2(d[keep, 1], d[keep, 0])*(SECTORS/(2*math.pi))).astype(int)) % SECTORS
            np.minimum.at(out[b], sector, r[keep])
    out[~np.isfinite(out)] = np.nan
    return out


def match(query, database, chunk=96):
    """Cost (N, SECTORS) of each candidate at each yaw shift k (yaw = k * 360/SECTORS degrees), lower is better."""
    q_valid = np.isfinite(query)
    q = np.where(q_valid, query, 0.).astype(np.float32)
    shifted = np.stack([np.roll(q, k, axis=1) for k in range(SECTORS)])               # (K, B, S)
    shifted_valid = np.stack([np.roll(q_valid, k, axis=1) for k in range(SECTORS)])
    cost = np.empty((len(database), SECTORS), np.float32)
    for start in range(0, len(database), chunk):
        block = database[start:start+chunk].astype(np.float32)                         # (n, B, S)
        valid = np.isfinite(block)
        d = np.where(valid, block, 1e3)                                                 # map sees nothing: far away
        c = np.minimum(np.abs(shifted[None]-d[:, None]), TAU)                           # (n, K, B, S)
        c = np.where(shifted_valid[None], c, MISSING*valid[:, None])
        cost[start:start+chunk] = c.mean(axis=(2, 3))
    return cost


# ---------- offline place database ----------
def map_key(map_dir):
    h = hashlib.sha256(('v%d %s %s %g %g %g' % (DB_VERSION, BANDS, SECTORS, RANGE, XY_STEP, Z_STEP)).encode())
    for name in ('map.npz', 'localization_cloud.npy'):
        f = Path(map_dir)/name
        if f.is_file():
            h.update(('%s %d' % (name, f.stat().st_size)).encode())
            with f.open('rb') as handle:
                for chunk in iter(lambda: handle.read(1 << 20), b''):
                    h.update(chunk)
    return h.hexdigest()[:24]


def candidates(grid):
    """Lattice positions (XY_STEP, Z_STEP) inside the map's free (flyable) space."""
    free = grid['free'].astype(bool)
    res, origin = float(grid['resolution']), np.asarray(grid['origin'], float)
    cells = np.argwhere(free)
    centers = origin+(cells+.5)*res
    keys = np.floor(np.column_stack([centers[:, :2]/XY_STEP, centers[:, 2:]/Z_STEP])).astype(np.int64)
    _, first = np.unique(keys, axis=0, return_index=True)
    return centers[np.sort(first)]


def build_database(map_dir, target=None, grid=None):
    begun = time.monotonic()
    if target is None:
        target, grid = map_cloud(map_dir)
    positions = candidates(grid)
    levels = np.unique(np.round(positions[:, 2], 3))
    desc = np.empty((len(positions), len(BANDS), SECTORS), np.float16)
    for z in levels:
        rows = np.flatnonzero(np.abs(positions[:, 2]-z) < 1e-3)
        cells = band_cells(target, z)
        for i in rows:
            near = [c[(np.abs(c[:, 0]-positions[i, 0]) < RANGE) & (np.abs(c[:, 1]-positions[i, 1]) < RANGE)]
                    for c in cells]
            desc[i] = descriptor(near, positions[i])
    seen = np.isfinite(desc.astype(np.float32)).mean(axis=(1, 2))
    keep = seen >= .25  # a place that sees almost nothing (outside the walls) cannot be recognised anyway
    return dict(key=np.array(map_key(map_dir)), positions=positions[keep].astype(np.float32), descriptors=desc[keep],
                target=target.astype(np.float32), build_s=np.array(time.monotonic()-begun))


def cache_path(map_dir):
    folder = Path(map_dir)
    if os.access(folder, os.W_OK):
        return folder/DB_NAME
    home = Path(os.environ.get('XDG_CACHE_HOME', Path.home()/'.cache'))/'naro'
    home.mkdir(parents=True, exist_ok=True)
    return home/('%s-%s' % (hashlib.sha1(str(folder.resolve()).encode()).hexdigest()[:12], DB_NAME))


def load_database(map_dir):
    """The cached database when it belongs to this map (shipped in the release), else built now and cached."""
    key = map_key(map_dir)
    with np.load(Path(map_dir)/'map.npz') as data:
        grid = {k: data[k] for k in data.files}
    for path in (Path(map_dir)/DB_NAME, cache_path(map_dir)):
        if path.is_file():
            try:
                with np.load(path) as data:
                    if str(data['key']) == key:
                        return {k: data[k] for k in data.files}, grid, False
            except (OSError, ValueError, KeyError):
                pass
    db = build_database(map_dir)
    target = cache_path(map_dir)
    temporary = target.with_name(target.name+'.tmp.npz')
    np.savez(temporary, **db)
    os.replace(temporary, target)
    return db, grid, True


# ---------- online ----------
def hypotheses_from_retrieval(cost, positions, here, top_k):
    """Best distinct (position, yaw) pairs; per candidate also its second yaw (symmetric rooms must show up)."""
    order = np.argsort(cost, axis=1)
    rows = []
    for i in np.argsort(cost.min(axis=1))[:max(top_k*8, 40)]:
        first = order[i, 0]
        rows.append((float(cost[i, first]), i, first))
        for k in order[i, 1:8]:
            if min((k-first) % SECTORS, (first-k) % SECTORS) >= 4:
                rows.append((float(cost[i, k]), i, k))
                break
    rows.sort()
    picked = []
    for c, i, k in rows:
        yaw = k*2*math.pi/SECTORS
        if any(np.linalg.norm(positions[i, :2]-positions[j, :2]) < 1. and
               abs(math.atan2(math.sin(yaw-y), math.cos(yaw-y))) < math.radians(20.) for _, j, y in picked):
            continue
        picked.append((c, i, yaw))
        if len(picked) == top_k:
            break
    out = []
    for c, i, yaw in picked:
        T = yaw_matrix(yaw)
        T[:3, 3] = positions[i]-T[:3, :3]@here  # the aircraft (here, odometry) is at the candidate position
        out.append(('place_%d' % len(out), T, round(c, 3)))
    return out


def relocalize(points, name, here=None, thorough=False, threads=4):
    if not (Path(name)/'localization_cloud.npy').is_file():
        # The exported occupancy grid is inflated for safety: registered against it, answers are biased by
        # 0.2-0.5 m while every check still passes (bench_relocalize.py --lattice). Refuse instead.
        # The node then only accepts "powered on at the mapping origin" (its walls-only check).
        return dict(accepted=False, unavailable=True,
                    reason='the map has no localization_cloud.npy (survey points); not relocalizing')
    import small_gicp
    begun = time.monotonic()
    db, grid, built = load_database(name)
    timing = dict(database_s=round(time.monotonic()-begun, 2), database_built=built)
    here = np.zeros(3) if here is None else np.asarray(here, float)
    src = points[np.linalg.norm(points[:, :2]-here[:2], axis=1) < 20.]
    src = voxel(src, SOURCE_VOXEL)
    if len(src) < MIN_POINTS:
        return dict(accepted=False, reason='too few LiDAR points (%d)' % len(src), timing=timing)
    t = time.monotonic()
    query = descriptor(band_cells(src, here[2]), here)
    cost = match(query, db['descriptors'])
    hypotheses = [('identity', np.eye(4), None)]
    hypotheses += hypotheses_from_retrieval(cost, db['positions'].astype(float), here, 10 if thorough else 6)
    timing['retrieval_s'] = round(time.monotonic()-t, 2)
    t = time.monotonic()
    target, tree = small_gicp.preprocess_points(db['target'].astype(float), downsampling_resolution=TARGET_VOXEL,
                                                num_neighbors=10, num_threads=threads)
    voxels = small_gicp.GaussianVoxelMap(.5)
    voxels.insert(target)
    coarse_source, _ = small_gicp.preprocess_points(src, downsampling_resolution=.25, num_neighbors=10,
                                                    num_threads=threads)
    timing['preprocess_s'] = round(time.monotonic()-t, 2)
    if thorough:
        hypotheses += kiss_hypotheses(src, db['target'])
    t = time.monotonic()
    # Coarse: every hypothesis, VGICP on 0.5 m Gaussian voxels (no tree searches; wrong guesses cost little).
    sample = src[::3]
    coarse = []
    for label, T0, retrieval_cost in hypotheses:
        r = small_gicp.align(voxels, coarse_source, init_T_target_source=T0, max_correspondence_distance=1.,
                             num_threads=threads, max_iterations=12)
        T = np.asarray(r.T_target_source)
        coarse.append(dict(label=label, T=T, cost=retrieval_cost, coarse_fitness=score(T, sample, tree, threads)[0]))
    top = max(h['coarse_fitness'] for h in coarse)
    # Fine: GICP at 0.1 m for the leader and anything that could still rival it. Nothing is refined when even the
    # leader is far from acceptable (another place: answer quickly).
    contenders = sorted([h for h in coarse if h['coarse_fitness'] >= max(top-.3, MIN_FITNESS-.15)],
                        key=lambda h: -h['coarse_fitness'])[:5 if thorough else 4]
    source, _ = small_gicp.preprocess_points(src, downsampling_resolution=SOURCE_VOXEL, num_neighbors=10,
                                             num_threads=threads)
    refined = []
    for h in coarse:
        T = h['T']
        if any(h is c for c in contenders):
            T = settle_height(T, src[::7], tree, threads)
            r = small_gicp.align(target, source, tree, init_T_target_source=T, registration_type='GICP',
                                 max_correspondence_distance=.3, num_threads=threads, max_iterations=15)
            T = np.asarray(r.T_target_source)
            fitness, rmse = score(T, src, tree, threads)
        else:  # clearly worse than the leader: keep the coarse numbers (never a rival within the margin)
            fitness, rmse = h['coarse_fitness'], 9.
        refined.append(dict(label=h['label'], T=T, fitness=fitness, rmse=rmse, cost=h['cost'], tilt=tilt_deg(T)))
    timing['registration_s'] = round(time.monotonic()-t, 2)
    timing['total_s'] = round(time.monotonic()-begun, 2)
    return decide(refined, grid, here, src, timing, MAX_RMSE, db['target'])


def score(T, src, tree, threads, near_d=FIT_D):
    moved = src@T[:3, :3].T+T[:3, 3]
    _, sq = tree.batch_nearest_neighbor_search(moved, num_threads=threads)
    dist = np.sqrt(np.asarray(sq, float))
    near = dist < near_d
    return float(near.mean()), float(np.sqrt((dist[near]**2).mean())) if near.any() else 9.


def settle_height(T, sample, tree, threads):
    """Walls look the same a little higher or lower; floor, ceiling and table tops do not. Before the fine
    registration, put the hypothesis at the height (+-1 m, 0.1 m steps) where the scan fits best."""
    shifts = np.arange(-1., 1.01, .1)
    fits = []
    for dz in shifts:
        moved = T.copy()
        moved[2, 3] += dz
        fits.append(score(moved, sample, tree, threads, near_d=.1)[0])
    out = T.copy()
    out[2, 3] += shifts[int(np.argmax(fits))]
    return out


def kiss_hypotheses(src, target):
    try:
        import kiss_matcher as km
    except ImportError:
        return []
    matcher = km.KISSMatcher(km.KISSMatcherConfig(.2))
    result = matcher.estimate(src.astype(np.float32), np.asarray(target, np.float32))
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = np.asarray(result.rotation), np.asarray(result.translation)
    return [('kiss', T, None)]


def consistency(T, src, here, target):
    """Two-sided check at the final pose: the descriptor of the map around the aircraft against the descriptor of
    the scan, both in the map frame (fitness only asks whether the scan lies on the map; this also asks whether
    the map's walls around the aircraft were seen where they should be)."""
    at = T@np.append(here, 1.)
    moved = src@T[:3, :3].T+T[:3, 3]
    near = target[(np.abs(target[:, 0]-at[0]) < RANGE) & (np.abs(target[:, 1]-at[1]) < RANGE)]
    seen = descriptor(band_cells(moved, at[2]), at)
    expected = descriptor(band_cells(near, at[2]), at)
    return float(match(seen, expected[None])[0, 0])


def decide(refined, grid, here, src, timing, max_rmse=MAX_RMSE, target=None):
    good = [h for h in refined if h['tilt'] <= MAX_TILT_DEG]
    good.sort(key=lambda h: -h['fitness'])
    report = [dict(label=h['label'], fitness=round(h['fitness'], 3), rmse=round(h['rmse'], 3), cost=h['cost'],
                   tilt=round(h['tilt'], 1), xyz=np.round(h['T'][:3, 3], 2).tolist(),
                   yaw_deg=round(math.degrees(yaw_of(h['T'])), 1)) for h in refined]
    out = dict(accepted=False, hypotheses=report, points=len(src), timing=timing)
    if not good:
        return dict(out, reason='no level alignment (roll/pitch within %g deg)' % MAX_TILT_DEG)

    def placed(T):
        where = T@np.append(here, 1.)
        cell = np.floor((where[:3]-grid['origin'])/float(grid['resolution'])).astype(int)
        inside = bool(np.all(cell >= 0) and np.all(cell < grid['occupied'].shape))
        return where, inside and not bool(grid['occupied'][tuple(cell)])

    def passes(h):
        return h['fitness'] >= MIN_FITNESS and h['rmse'] <= max_rmse and placed(h['T'])[1]

    best = good[0]
    # Prior: the operating procedure powers the aircraft on at the mapping origin. When the identity, refined, explains
    # the scan as well as anything else, that is the answer, even in a room that has a look-alike elsewhere.
    origin = next((h for h in good if h['label'] == 'identity'), None)
    if origin is not None and not different(origin['T'], np.eye(4)) and passes(origin) and \
            origin['fitness'] >= best['fitness']-.03:
        best = origin
    rival = next((h for h in good if h is not best and different(h['T'], best['T'])), None)
    where, free = placed(best['T'])
    out.update(T_map_odom=best['T'].tolist(), fitness=round(best['fitness'], 3), rmse=round(best['rmse'], 3),
               at_origin=not different(best['T'], np.eye(4)), aircraft_in_map=np.round(where[:3], 2).tolist(),
               rival_fitness=None if rival is None else round(rival['fitness'], 3), chosen=best['label'],
               consistency=None if target is None else round(consistency(best['T'], src, here, target), 3))
    if best['fitness'] < MIN_FITNESS:
        return dict(out, reason='scan does not match the map well enough (fitness %.2f)' % best['fitness'])
    if best['rmse'] > max_rmse:
        return dict(out, reason='alignment not tight enough (RMSE %.2f m)' % best['rmse'])
    if best['label'] != 'identity' and rival is not None and best['fitness']-rival['fitness'] < MIN_MARGIN:
        return dict(out, reason='ambiguous: another place fits almost as well (%.2f vs %.2f)'
                    % (best['fitness'], rival['fitness']))
    if not free:
        return dict(out, reason='the aircraft would be outside the map or inside an obstacle')
    if out['consistency'] is not None and out['consistency'] > MAX_INCONSISTENCY:
        return dict(out, reason='the surroundings do not look like the map there (%.2f)' % out['consistency'])
    return dict(out, accepted=True, reason='ok')


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--map', required=True, help='exported map directory')
    p.add_argument('--points', help='.npy N x 3, odometry frame')
    p.add_argument('--here', type=float, nargs=3, help='aircraft position in the odometry frame')
    p.add_argument('--thorough', action='store_true', help='more candidates and a KISS-Matcher hypothesis (slower)')
    p.add_argument('--threads', type=int, default=4)
    p.add_argument('--build', action='store_true', help='only make sure the place database is built and cached')
    a = p.parse_args()
    try:
        if a.build:
            db, _, built = load_database(a.map)
            result = dict(built=built, candidates=len(db['positions']), target_points=len(db['target']),
                          seconds=round(float(db['build_s']), 1))
        else:
            result = relocalize(np.load(a.points)[:, :3].astype(float), a.map, a.here, a.thorough, a.threads)
    except Exception as error:  # never let a crash look like a result
        result = dict(accepted=False, reason='relocalization error: %r' % error)
    print(json.dumps(result))
    sys.stdout.flush()


if __name__ == '__main__':
    main()
