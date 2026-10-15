"""Gravity-aligned structural correlation; fixed six registration proposals.

Project vertical surfaces, then correlate a sparse observed footprint against
the entire map at 5 degree yaw spacing. Unobserved sectors do not cast votes.
The FFT replaces the polar database comparison, not an extra registration pass.
"""
import math
import numpy as np


def build_map_index(target, positions, key):
    """Offline only: map projection, likelihood field and padded FFT spectra."""
    step = .25
    p = np.asarray(target.points())[:, :3]
    n = np.asarray(target.normals())[:, :3]
    map_floor = float(np.quantile(p[:, 2], .05))
    m = p[(np.abs(n[:, 2]) < .65) & (p[:, 2] > map_floor+.3) & (p[:, 2] < map_floor+2.5), :2]
    # Search every observed XY column, including above a desk or an upper
    # platform. A single global flight-height slice erases valid startup places.
    # XY retrieval is shared across levels; height is settled in registration.
    _,first=np.unique(np.round(positions[:,:2],3),axis=0,return_index=True)
    positions=positions[np.sort(first)]
    out = dict(key=np.array(key), map_floor=np.array(map_floor), positions=positions,
               map_vertical_points=np.array(len(m)))
    if len(m) < 50 or len(positions) == 0: return out
    origin = np.floor(np.minimum(m.min(axis=0), positions[:, :2].min(axis=0))/step)*step-1
    extent = np.maximum(m.max(axis=0), positions[:, :2].max(axis=0))-origin+2
    local_shape = np.ceil(extent/step).astype(int)
    grid = np.zeros(tuple(local_shape), np.float32)
    ij = np.floor((m-origin)/step).astype(int)
    grid[ij[:, 0], ij[:, 1]] = 1
    # Compact likelihood field: tolerance for coarse yaw and rasterization.
    field = grid.copy()
    for dx in range(-2, 3):
        for dy in range(-2, 3):
            if dx == 0 and dy == 0: continue
            field = np.maximum(field, np.roll(grid, (dx, dy), (0, 1))*math.exp(-(dx*dx+dy*dy)/2.))
    # Query offsets are within a 20 m radius at EVERY yaw. If the FFT period
    # exceeds both map/candidate bounds by that radius, negative query offsets
    # wrap only into the zero tail, and positive offsets never wrap at all.
    low = np.ceil(np.log2(np.maximum(8, np.ceil(extent/step)))).astype(int)
    high = np.ceil(np.log2(np.maximum(8, np.ceil((extent+20)/step)))).astype(int)
    out.update(origin=origin, extent=extent)
    for x in range(low[0], high[0]+1):
        for y in range(low[1], high[1]+1):
            shape = (2**int(x), 2**int(y))
            if np.prod(shape) <= 2**22:
                out['fft_%d_%d' % shape] = np.fft.rfft2(field, s=shape)
    return out


def proposals(source, target, positions, here, top_k=6, prepared=None):
    step = .25
    p = np.asarray(source.points())[:, :3]
    n = np.asarray(source.normals())[:, :3]
    query_floor = float(np.quantile(p[:, 2], .05))
    q = p[(np.abs(n[:, 2]) < .65) & (p[:, 2] > query_floor+.3) &
          (p[:, 2] < query_floor+2.5) & (np.linalg.norm(p[:, :2]-here[:2], axis=1) > .6), :2]-here[:2]
    index = build_map_index(target, positions, 'uncached') if prepared is None else prepared
    map_count = int(index['map_vertical_points']); positions = index['positions']
    map_floor = float(index['map_floor'])
    if len(q) < 50 or map_count < 50 or len(positions) == 0:
        return [], dict(reason='too few vertical surface samples')
    origin = index['origin']
    span = np.ceil((index['extent']+np.linalg.norm(q, axis=1).max())/step).astype(int)
    shape = tuple(int(2**math.ceil(math.log2(max(8, v)))) for v in span)
    name = 'fft_%d_%d' % shape
    if name not in index:
        return [], dict(reason='structural grid exceeds bounded workspace')
    transform = index[name]
    cells = np.floor((positions[:, :2]-origin)/step).astype(int)
    valid = np.all((cells >= 0) & (cells < np.asarray(shape)), axis=1)
    indices = np.flatnonzero(valid)
    cost = np.ones((len(positions), 72), np.float32)
    for k in range(72):
        yaw = k*2*math.pi/72
        c, s = math.cos(yaw), math.sin(yaw)
        xy = q@np.array([[c, s], [-s, c]])
        occupied = np.unique(np.floor(xy/step).astype(int), axis=0)
        scan = np.zeros(shape, np.float32)
        scan[occupied[:, 0] % shape[0], occupied[:, 1] % shape[1]] = 1
        corr = np.fft.irfft2(transform*np.conj(np.fft.rfft2(scan)), s=shape)
        cost[indices, k] = 1-corr[cells[indices, 0], cells[indices, 1]]/len(occupied)
    # Keep one height per XY/yaw; the subsequent existing height search and GICP
    # solve Z. This avoids spending the fixed budget on duplicate floor levels.
    flat = np.argsort(cost, axis=None)
    picked = []
    for f in flat:
        i, k = np.unravel_index(f, cost.shape)
        yaw = k*2*math.pi/72
        if any(np.linalg.norm(positions[i, :2]-positions[j, :2]) < 1. and
               min((k-l) % 72, (l-k) % 72) < 4 for j, l in picked): continue
        picked.append((i, k))
        if len(picked) == top_k: break
    out = []
    for i, k in picked:
        yaw = k*2*math.pi/72; c, s = math.cos(yaw), math.sin(yaw)
        T = np.eye(4); T[:2, :2] = [[c, -s], [s, c]]
        T[:3, 3] = positions[i]-T[:3, :3]@here
        T[2, 3] = map_floor-query_floor
        out.append(('correlation_%d' % len(out), T, round(float(cost[i, k]), 4)))
    return out, dict(shape=shape, source_vertical_points=len(q), map_vertical_points=map_count,
                     best_cost=float(cost.min()), query_floor=query_floor, map_floor=map_floor,
                     algorithm='vertical_surface_fft')
