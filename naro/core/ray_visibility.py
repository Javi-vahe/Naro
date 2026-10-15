"""Bounded free-ray evidence. Geometry cache is map-only and portable NumPy."""
import os
from pathlib import Path
import numpy as np

VOXEL=.2


def keys(coords,lo,shape):
    c=coords-lo;valid=np.all((c>=0)&(c<shape),axis=1)
    return (c[:,0]*shape[1]+c[:,1])*shape[2]+c[:,2],valid


def load_index(map_dir,key,cache_path,threads=4):
    path=Path(cache_path).with_suffix('.ray-visibility.npz');key=str(key)+' ray-visibility-v1'
    if path.exists():
        with np.load(path) as data:
            if str(data['key'])==key:
                return (data['occupied'],data['lo'],data['shape']),False,path
    import small_gicp
    cloud,_=small_gicp.preprocess_points(np.load(Path(map_dir)/'localization_cloud.npy')[:,:3].astype(float),
        downsampling_resolution=.1,num_neighbors=10,num_threads=threads)
    target=np.asarray(cloud.points())[:,:3];normals=np.asarray(cloud.normals())[:,:3]
    floor=float(np.percentile(target[:,2],5))
    # Ground voxels falsely occlude grazing rays long before their endpoints.
    coords=np.floor(target[(np.abs(normals[:,2])<.7)&(target[:,2]>floor+.15)]/VOXEL).astype(np.int64)
    if len(coords):
        lo=coords.min(axis=0)-2;shape=coords.max(axis=0)-lo+3
        occupied=np.unique(keys(coords,lo,shape)[0])
    else:lo=np.zeros(3,dtype=np.int64);shape=np.ones(3,dtype=np.int64);occupied=np.array([],dtype=np.int64)
    temporary=path.with_name(path.name+'.tmp.npz')
    np.savez(temporary,key=key,occupied=occupied,lo=lo,shape=shape);os.replace(temporary,path)
    return (occupied,lo,shape),True,path


def measured_rays(raw,origin):
    p=raw[:,:3]-origin;r=np.linalg.norm(p,axis=1);mask=(r>.8)&(r<40)&np.isfinite(r)
    p=p[mask];r=r[mask]
    if len(r)<32:return None
    d=p/r[:,None];az=np.floor((np.arctan2(d[:,1],d[:,0])+np.pi)*180/np.pi).astype(int)%360
    el=np.floor((np.arcsin(np.clip(d[:,2],-1,1))+np.pi/2)*180/np.pi).astype(int)
    angular=az*181+el;order=np.argsort(-r,kind='stable')
    _,first=np.unique(angular[order],return_index=True);selected=order[first]
    if len(selected)>512:selected=selected[np.linspace(0,len(selected)-1,512).astype(int)]
    d=d[selected];r=r[selected];az=az[selected]
    steps=np.arange(.5,float(r.max()),.1)
    grid=d[:,None,:]*steps[None,:,None]+origin
    return grid,steps[None,:]<(r[:,None]-.4),az


def blocked_rays(T,grid,valid,index):
    occupied,lo,shape=index
    moved=grid.reshape(-1,3)@T[:3,:3].T+T[:3,3]
    code,inside=keys(np.floor(moved/VOXEL).astype(np.int64),lo,shape)
    slot=np.searchsorted(occupied,code);exists=(slot<len(occupied))&inside
    exists[exists]=occupied[slot[exists]]==code[exists]
    return (exists.reshape(valid.shape)&valid).any(axis=1)


def consensus(candidates,az,different):
    winners=[];comparisons=[]
    for c in candidates:
        # At least 90% of the measured free rays must agree with the map.
        supported=float(c['blocked'].mean())<=.1;rivals=0
        for r in candidates:
            if c is r or not different(c['h']['T'],r['h']['T']):continue
            win=~c['blocked']&r['blocked'];lose=c['blocked']&~r['blocked']
            nwin=int(win.sum());nlose=int(lose.sum())
            gains=np.bincount(az[win]//30,minlength=12)-np.bincount(az[lose]//30,minlength=12)
            directions=int((gains>=4).sum())
            # Paired ray margin plus separated angular support; counts alone
            # must not turn a tiny discretization difference into a location.
            clear=nwin>=20 and nwin-nlose>=3*np.sqrt(nwin+nlose) and directions>=2
            comparisons.append(dict(candidate=c['h']['label'],rival=r['h']['label'],support=nwin,
                opposition=nlose,supporting_directions=directions,clear=bool(clear)))
            supported &= clear;rivals+=1
        if supported and rivals:winners.append(c['h'])
    return winners,comparisons
