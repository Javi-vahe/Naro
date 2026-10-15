"""Regional structure and measured free-ray decisions."""
import importlib.util
import math
from pathlib import Path
import time
import numpy as np

_spec=importlib.util.spec_from_file_location('structural_private_base',Path(__file__).with_name('relocalize_engine.py'))
base=importlib.util.module_from_spec(_spec);_spec.loader.exec_module(base)
VERSION='observed-ray-consensus-2.10.1'
AZIMUTH_REGIONS=12
REGION_LIMIT=80
MAX_STRUCTURAL_LOSS=.24


def _sample(cloud,here):
    p=np.asarray(cloud.points())[:,:3];n=np.asarray(cloud.normals())[:,:3]
    az=np.floor(np.arctan2(p[:,1]-here[1],p[:,0]-here[0])*AZIMUTH_REGIONS/(2*math.pi)).astype(int)%AZIMUTH_REGIONS
    # Horizontal patches below and above the aircraft are separate; walls form
    # the third class. Stratification prevents the ceiling from dominating.
    kind=np.where(np.abs(n[:,2])<.7,2,np.where(p[:,2]-here[2]<.15,0,1))
    ids=az*3+kind;chosen=[]
    for group in np.unique(ids):
        indices=np.flatnonzero(ids==group)
        if len(indices)<8:continue
        # First spread spatially, then take an even deterministic subset.
        order=np.lexsort((p[indices,2],p[indices,1],p[indices,0]));indices=indices[order]
        chosen.extend(indices[np.linspace(0,len(indices)-1,min(REGION_LIMIT,len(indices))).astype(int)])
    chosen=np.asarray(chosen,dtype=int)
    return p[chosen],ids[chosen]


def _evidence(T,p,groups,tree,threads):
    moved=p@T[:3,:3].T+T[:3,3]
    index,squared=tree.batch_nearest_neighbor_search(moved,num_threads=threads)
    dist=np.sqrt(np.asarray(squared))
    # Location evidence must contradict geometry, not reward a smoother patch
    # of an otherwise repeated wall. Distances inside the existing registration
    # correspondence tolerance are compatible with either place. Only excess
    # distance beyond that tolerance may distinguish competing locations.
    loss=np.minimum(np.maximum(dist-base.FIT_D,0)/base.FIT_D,1)**2
    # Normal direction is a sampling attribute, not positive place evidence:
    # sparse/noisy coplanar neighborhoods can give different PCA normals at
    # otherwise identical corners. Do not let that break geometric symmetry.
    regions=[float(loss[groups==g].mean()) for g in np.unique(groups)]
    contradictions=[]
    for g in np.unique(groups):
        mask=groups==g
        # Every stable surface sector must independently meet the SAME overlap
        # requirement as the full scan. A matching surface cannot compensate
        # for a contradictory surface in another observed direction.
        if mask.sum()<REGION_LIMIT//4:continue
        coverage=float((dist[mask]<base.FIT_D).mean())
        if coverage<base.MIN_FITNESS:
            contradictions.append(dict(group=int(g),points=int(mask.sum()),coverage=coverage))
    # Vertical geometry constrains the unknown horizontal place directly.
    # An isolated horizontal patch can be absent through view-dependent map
    # coverage; require disagreements in two directions before it vetoes a pose.
    vertical=any(c['group']%3==2 for c in contradictions)
    horizontal_directions={c['group']//3 for c in contradictions if c['group']%3!=2}
    veto=vertical or len(horizontal_directions)>=2
    return dict(loss=float(np.mean(regions)),regions=np.asarray(regions),covered=float((dist<.2).mean()),
                contradictions=contradictions,veto=veto)


def structural_decide(refined,grid,here,src,timing,max_rmse=base.MAX_RMSE,target=None,support=None):
    begun=time.monotonic()
    source_cloud,target_cloud,tree,threads=support[:4]
    points,groups=_sample(source_cloud,here)
    report=[dict(label=h['label'],fitness=round(h['fitness'],4),rmse=round(h['rmse'],4),
                 cost=h['cost'],tilt=round(h['tilt'],3),T=h['T'].tolist()) for h in refined]
    out=dict(accepted=False,hypotheses=report,points=len(src),timing=dict(timing),version=VERSION)
    if len(points)<80:return dict(out,reason='insufficient spatial support for structural decision')
    eligible=[]
    for h,r in zip(refined,report):
        placed,placement=base.localization_placement(h['T'],grid,here,tree,threads)
        r['placement']=placement
        # A rival must itself be a usable registration. A coarse-only alignment
        # with rmse=9, or a pose inside an obstacle, cannot veto a good answer.
        valid=h['tilt']<=base.MAX_TILT_DEG and h['fitness']>=base.MIN_FITNESS and h['rmse']<=max_rmse and placed
        r['eligible']=bool(valid)
        if not valid:continue
        e=_evidence(h['T'],points,groups,tree,threads)
        r.update(structural_loss=round(e['loss'],5),balanced_coverage=round(e['covered'],4),
                 contradictory_regions=e['contradictions'],regional_veto=e['veto'])
        if e['veto']:
            r['eligible']=False
            continue
        h=dict(h,evidence=e);eligible.append(h)
    out['structural_sample']=dict(points=len(points),regions=len(np.unique(groups)),max_per_region=REGION_LIMIT)
    out['timing']['decision_s']=round(time.monotonic()-begun,4)
    if not eligible:return dict(out,reason='no candidate passes geometry and placement or regional support')
    eligible.sort(key=lambda h:h['evidence']['loss']);best=eligible[0]
    out.update(T_map_odom=best['T'].tolist(),chosen=best['label'],fitness=round(best['fitness'],4),
               rmse=round(best['rmse'],4),structural_loss=round(best['evidence']['loss'],5),
               aircraft_in_map=(best['T']@np.append(here,1))[:3].tolist(),competitions=[])
    if best['evidence']['loss']>MAX_STRUCTURAL_LOSS:return dict(out,reason='inconsistent structural coverage')
    for rival in eligible[1:]:
        if not base.different(best['T'],rival['T']):continue
        difference=rival['evidence']['regions']-best['evidence']['regions']
        gain=float(difference.mean());spread=float(difference.std()/math.sqrt(len(difference)))
        beneficial=difference>.04;harmful=difference<-.04
        region_ids=np.unique(groups)
        directions=len(np.unique(region_ids[beneficial]//3))
        # Shared floor/ceiling regions contribute zeros. A fixed mean gap is
        # diluted by those irrelevant regions even when several wall directions
        # agree. Require directional support and a gain relative to its spread.
        clear=gain>.7*spread and directions>=2 and beneficial.sum()>harmful.sum()
        comparison=dict(rival=rival['label'],paired_gain=gain,regional_spread=spread,
                        supporting_directions=directions,better_regions=int(beneficial.sum()),
                        worse_regions=int(harmful.sum()),clear=bool(clear))
        out['competitions'].append(comparison)
        if not clear:return dict(out,reason='ambiguous structural evidence between distinct places')
    check=base.consistency(best['T'],src,here,target)
    out['consistency']=round(check,4)
    if check>base.MAX_INCONSISTENCY:return dict(out,reason='surrounding structure contradicts the candidate')
    out['timing']['decision_s']=round(time.monotonic()-begun,4)
    return dict(out,accepted=True,reason='distinct structural support')


def select_contenders(coarse, cloud, here, tree, threads, budget):
    """Spend the same four ICP calls on distinct, surface-balanced poses.

    Coarse inlier counts mostly measure floor area. Rank equal azimuth/surface
    groups instead, with a smooth robust loss at the coarse registration scale.
    Collapse converged duplicates before allocating the expensive fine stage.
    """
    points,groups=_sample(cloud,here)
    ids=np.unique(groups)
    for h in coarse:
        moved=points@h['T'][:3,:3].T+h['T'][:3,3]
        _,sq=tree.batch_nearest_neighbor_search(moved,num_threads=threads)
        loss=np.minimum(np.asarray(sq)/(.3**2),1.)
        h['information_loss']=float(np.mean([loss[groups==g].mean() for g in ids]))
    # Retrieval measures structural footprint; registration measures local fit.
    # Reserve one slot per retrieval family among the four fine alignments so a
    # locally smoother wrong floor cannot eliminate every global proposal.
    leaders=[]
    for prefix in ('correlation_','radial_'):
        family=[h for h in coarse if h['label'].startswith(prefix)]
        if family:leaders.append(min(family,key=lambda h:h['cost']))
    ranked=leaders+sorted(coarse,key=lambda h:h['information_loss'])
    picked=[]
    for h in ranked:
        if h['coarse_fitness']<base.MIN_FITNESS-.15:continue
        if any(not base.different(h['T'],c['T']) for c in picked):continue
        picked.append(h)
        if len(picked)==budget:break
    return picked


def full_view_decide(refined,grid,here,src,timing,max_rmse=base.MAX_RMSE,target=None,support=None):
    """An ambiguous near-field answer may be resolved by measured distant structure.

    Registration still uses 20 m and four ICP calls. Only rejected calls request
    this bounded witness. Missing horizontal map patches cannot erase a unique
    match of walls/trunks measured over the full LiDAR view.
    """
    result=structural_decide(refined,grid,here,src,timing,max_rmse,target,support)
    if result['accepted'] or len(support)<5:return result
    begun=time.monotonic();cloud,_,tree,threads,raw=support
    import small_gicp
    full,_=small_gicp.preprocess_points(np.asarray(raw,float),downsampling_resolution=.25,num_neighbors=10,num_threads=threads)
    p=np.asarray(full.points())[:,:3];n=np.asarray(full.normals())[:,:3]
    p=p[np.abs(n[:,2])<.65]
    az=np.floor(np.arctan2(p[:,1]-here[1],p[:,0]-here[0])*12/(2*math.pi)).astype(int)%12
    far=np.linalg.norm(p[:,:2]-here[:2],axis=1)>20
    ids=az*2+far;chosen=[]
    for g in np.unique(ids):
        ix=np.flatnonzero(ids==g)
        if len(ix)<8:continue
        chosen.extend(ix[np.linspace(0,len(ix)-1,min(80,len(ix))).astype(int)])
    chosen=np.asarray(chosen,int);p=p[chosen];ids=ids[chosen]
    if len(p)<80:return result
    candidates=[]
    for h in refined:
        if h['rmse']>max_rmse or h['tilt']>base.MAX_TILT_DEG or h['fitness']<base.MIN_FITNESS:continue
        if not base.localization_placement(h['T'],grid,here,tree,threads)[0]:continue
        _,sq=tree.batch_nearest_neighbor_search(p@h['T'][:3,:3].T+h['T'][:3,3],num_threads=threads)
        d=np.sqrt(sq);coverage=float((d<.2).mean())
        candidates.append(dict(h=h,d=d,coverage=coverage))
    candidates.sort(key=lambda c:-c['coverage'])
    witness=dict(points=len(p),far_points=int((ids%2==1).sum()),candidates=[dict(label=c['h']['label'],coverage=c['coverage']) for c in candidates])
    result['full_view_witness']=witness
    if not candidates:return result
    best=candidates[0];stable=[g for g in np.unique(ids) if (ids==g).sum()>=20]
    # Strong positive geometry is required; a small ranking advantage is not a
    # rescue. At least three directions must be observed and independently fit.
    supported=[g for g in stable if (best['d'][ids==g]<.2).mean()>=base.MIN_FITNESS]
    strong=best['coverage']>=.9 and len({g//2 for g in supported})>=3
    comparisons=[]
    for rival in candidates[1:]:
        if not base.different(best['h']['T'],rival['h']['T']):continue
        win=(best['d']<.2)&(rival['d']>.4);lose=(rival['d']<.2)&(best['d']>.4)
        winning=[int(g) for g in stable if win[ids==g].sum()>=20]
        losing=[int(g) for g in stable if lose[ids==g].sum()>=20]
        clear=len({g//2 for g in winning})>=2 and not losing
        comparisons.append(dict(rival=rival['h']['label'],winning_regions=winning,losing_regions=losing,clear=clear))
        strong &= clear
    witness.update(comparisons=comparisons,strong=bool(strong))
    result['timing']['full_view_witness_s']=round(time.monotonic()-begun,4)
    if not strong or not comparisons:return result
    h=best['h'];check=base.consistency(h['T'],src,here,target)
    if check>base.MAX_INCONSISTENCY:return result
    result['near_field_decision']={k:result.pop(k) for k in ('chosen','structural_loss','competitions') if k in result}
    return dict(result,accepted=True,reason='unique full-view structural witness',chosen=h['label'],
                T_map_odom=h['T'].tolist(),fitness=h['fitness'],rmse=h['rmse'],consistency=check,
                aircraft_in_map=(h['T']@np.r_[here,1])[:3].tolist())


def patch_vote_decide(refined,grid,here,src,timing,max_rmse=base.MAX_RMSE,target=None,support=None):
    """Compare support in spatially distinct 3-D patches, including one doorway.

    Missing support is compared pairwise, with a four-to-one supermajority in
    both patches and points. A minority of changed or poorly mapped patches is
    tolerated. Known-free contradictions are reported separately from missing
    surfaces: they are stronger evidence, not interchangeable observations.
    """
    result=full_view_decide(refined,grid,here,src,timing,max_rmse,target,support)
    if result['accepted']:return result
    begun=time.monotonic();cloud,_,tree,threads,raw=support
    import small_gicp
    source,_=small_gicp.preprocess_points(np.asarray(raw,float),downsampling_resolution=.15,num_neighbors=10,num_threads=threads)
    p=np.asarray(source.points())[:,:3]
    p=p[::max(1,int(np.ceil(len(p)/6000)))]
    _,groups=np.unique(np.floor((p-here)/.6).astype(int),axis=0,return_inverse=True)
    counts=np.bincount(groups);valid_patch=counts>=4
    evidence={h['label']:h for h in result['hypotheses']}
    candidates=[]
    for h in refined:
        if h['rmse']>max_rmse or h['tilt']>base.MAX_TILT_DEG or h['fitness']<base.MIN_FITNESS:continue
        if not base.localization_placement(h['T'],grid,here,tree,threads)[0]:continue
        moved=p@h['T'][:3,:3].T+h['T'][:3,3]
        _,sq=tree.batch_nearest_neighbor_search(moved,num_threads=threads);dist=np.sqrt(sq)
        cells=np.floor((moved-grid['origin'])/float(grid['resolution'])).astype(int)
        inside=((cells>=0)&(cells<np.asarray(grid['free'].shape))).all(axis=1)
        free=np.zeros(len(p),bool);free[inside]=grid.get('observed_free', grid['free'])[tuple(cells[inside].T)]
        # Pairwise votes cannot establish that any candidate actually explains
        # the scan. Keep the existing absolute structural-loss requirement,
        # while allowing votes to overcome a minority of missing map patches.
        loss=evidence[h['label']].get('structural_loss',float('inf'))
        candidates.append(dict(h=h,matched=dist<.2,contradiction=dist>.4,known_free=free&(dist>.4),
                               absolute_support=loss<=MAX_STRUCTURAL_LOSS))
    report=dict(points=len(p),patches=int(valid_patch.sum()),candidates=[],comparisons=[])
    for c in candidates:
        report['candidates'].append(dict(label=c['h']['label'],matched=int(c['matched'].sum()),
                                          unmatched=int(c['contradiction'].sum()),free_space_contradictions=int(c['known_free'].sum()),
                                          absolute_support=bool(c['absolute_support'])))
    result['patch_votes']=report
    # Fine registration collapses converged duplicates. A single supported
    # place has no rival to defeat; assess its absolute full-view evidence.
    # Require >=90% vertical coverage across >=3 directions (strong witness),
    # >=90% surface support and no points contradicting mapped free space.
    witness=result.get('full_view_witness',{})
    if len(candidates)==1 and witness.get('strong') and len(witness.get('candidates',[]))==1:
        c=candidates[0];h=c['h']
        supported=(witness['candidates'][0]['label']==h['label'] and c['absolute_support']
                   and float(c['matched'].mean())>=.9 and not c['known_free'].any())
        if supported:
            check=base.consistency(h['T'],src,here,target)
            if check<=base.MAX_INCONSISTENCY:
                result['previous_decision']={k:result.pop(k) for k in ('chosen','structural_loss','competitions','reason') if k in result}
                result['timing']['patch_votes_s']=round(time.monotonic()-begun,4)
                return dict(result,accepted=True,reason='unique full-view and surface-patch support',chosen=h['label'],
                    T_map_odom=h['T'].tolist(),fitness=h['fitness'],rmse=h['rmse'],consistency=check,
                    aircraft_in_map=(h['T']@np.r_[here,1])[:3].tolist())
    # First use measured empty-space contradictions. Only if they cannot settle
    # the place, compare supported surface patches with a stronger majority.
    for mode in ('known_free','contradiction'):
        winners=[]
        for c in candidates:
            rivals=[r for r in candidates if base.different(c['h']['T'],r['h']['T'])]
            supported=bool(rivals) and c['absolute_support']
            for rival in rivals:
                win=c['matched']&rival[mode];lose=rival['matched']&c[mode]
                wp=np.bincount(groups[win],minlength=len(counts));lp=np.bincount(groups[lose],minlength=len(counts))
                winning=int(((wp>=4)&valid_patch).sum());losing=int(((lp>=4)&valid_patch).sum())
                spatial=(winning>=3 and losing<=1) if mode=='known_free' else winning>=4*max(1,losing)
                clear=spatial and win.sum()>=20 and win.sum()>=4*max(1,lose.sum())
                report['comparisons'].append(dict(mode=mode,candidate=c['h']['label'],rival=rival['h']['label'],
                    winning_patches=winning,losing_patches=losing,win_points=int(win.sum()),lose_points=int(lose.sum()),clear=bool(clear)))
                supported &= clear
            if supported:winners.append(c)
        if len(winners)!=1:continue
        h=winners[0]['h'];check=base.consistency(h['T'],src,here,target)
        if check>base.MAX_INCONSISTENCY:continue
        result['previous_decision']={k:result.pop(k) for k in ('chosen','structural_loss','competitions','reason') if k in result}
        result['timing']['patch_votes_s']=round(time.monotonic()-begun,4)
        reason='observed free-space patch consensus' if mode=='known_free' else 'distributed surface-patch consensus'
        return dict(result,accepted=True,reason=reason,chosen=h['label'],
                    T_map_odom=h['T'].tolist(),fitness=h['fitness'],rmse=h['rmse'],consistency=check,
                    aircraft_in_map=(h['T']@np.r_[here,1])[:3].tolist())
    result['timing']['patch_votes_s']=round(time.monotonic()-begun,4)
    return result


base.select_contenders=select_contenders
base.decide=patch_vote_decide
load_database=base.load_database


_ray_spec=importlib.util.spec_from_file_location('private_ray_visibility',Path(__file__).with_name('ray_visibility.py'))
rays=importlib.util.module_from_spec(_ray_spec);_ray_spec.loader.exec_module(rays)
SUPPORTS_SENSOR_ORIGIN=True


def prepare_database_extras(map_dir,threads=4):
    _,built,path=rays.load_index(map_dir,base.map_key(map_dir),base.cache_path(map_dir),threads)
    return dict(ray_database=str(path),ray_database_built=built,ray_database_bytes=path.stat().st_size)


def visible_decide(refined,grid,here,src,timing,max_rmse=base.MAX_RMSE,target=None,support=None,
                   *,sensor_origin=None,map_dir=None):
    result=patch_vote_decide(refined,grid,here,src,timing,max_rmse,target,support)
    if result['accepted']:return result
    if sensor_origin is None:
        result['ray_visibility']=dict(status='calibrated sensor origin not supplied');return result
    evidence={h['label']:h for h in result['hypotheses']}
    viable=[h for h in refined if evidence[h['label']].get('eligible') and
            evidence[h['label']].get('structural_loss',float('inf'))<=MAX_STRUCTURAL_LOSS]
    if len(viable)<2:return result
    begun=time.monotonic();raw=np.asarray(support[4],float)
    observed=rays.measured_rays(raw,np.asarray(sensor_origin,float))
    if observed is None:return result
    index,built,path=rays.load_index(map_dir,base.map_key(map_dir),base.cache_path(map_dir),support[3])
    positions,valid,az=observed
    candidates=[dict(h=h,blocked=rays.blocked_rays(h['T'],positions,valid,index)) for h in viable]
    winners,comparisons=rays.consensus(candidates,az,base.different)
    result['ray_visibility']=dict(status='compared',rays=len(az),map_vertical_voxels=len(index[0]),
        sensor_origin_odom=np.asarray(sensor_origin).tolist(),candidates=[dict(label=c['h']['label'],
        blocked=int(c['blocked'].sum())) for c in candidates],comparisons=comparisons)
    result['timing']['ray_database_built']=built
    result['timing']['ray_visibility_s']=round(time.monotonic()-begun,4)
    if len(winners)!=1:return result
    h=winners[0];check=base.consistency(h['T'],src,here,target)
    if check>base.MAX_INCONSISTENCY:return result
    result['before_ray_decision']={k:result.pop(k) for k in ('chosen','structural_loss','competitions','reason') if k in result}
    return dict(result,accepted=True,reason='measured free-ray consensus',chosen=h['label'],
        T_map_odom=h['T'].tolist(),fitness=h['fitness'],rmse=h['rmse'],consistency=check,
        aircraft_in_map=(h['T']@np.r_[here,1])[:3].tolist())


def relocalize(points,name,here=None,thorough=False,threads=4,*,sensor_origin=None):
    # One-shot process API. Sensor origin is odometry + physical extrinsics,
    # never a prior global position or a simulator truth pose.
    previous=base.decide;begun=time.monotonic()
    def decision(*args,**kwargs):
        return visible_decide(*args,**kwargs,sensor_origin=sensor_origin,map_dir=name)
    base.decide=decision
    try:
        result=base.relocalize(points,name,here,thorough,threads)
        result.setdefault('timing',{})['total_s']=round(time.monotonic()-begun,2)
        return result
    finally:base.decide=previous


if __name__=='__main__':
    import argparse,json
    p=argparse.ArgumentParser();p.add_argument('--map',required=True);p.add_argument('--points')
    p.add_argument('--here',nargs=3,type=float);p.add_argument('--sensor-origin',nargs=3,type=float)
    p.add_argument('--threads',type=int,default=4);p.add_argument('--build',action='store_true')
    p.add_argument('--thorough',action='store_true');a=p.parse_args()
    if a.build:
        db,_,built=load_database(a.map);base.structural_database(a.map,db,threads=a.threads)
        reply=dict(built=built,**prepare_database_extras(a.map,a.threads))
    else:reply=relocalize(np.load(a.points)[:,:3].astype(float),a.map,a.here,a.thorough,a.threads,sensor_origin=a.sensor_origin)
    print(json.dumps(reply,allow_nan=False))
