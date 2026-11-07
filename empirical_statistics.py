"""Independent statistical primitives with calendar-aware paired inference.
All tests and intervals are exploratory. No inference is fabricated for degenerate samples.
"""
import hashlib
import json
import numpy as np
from scipy.stats import norm, chi2


def validate_series(values,positions=None):
    x=np.asarray(values,float)
    if x.ndim!=1 or not np.isfinite(x).all():raise ValueError('Expected a finite one-dimensional series')
    p=np.arange(len(x)) if positions is None else np.asarray(positions)
    if p.shape!=x.shape or not np.isfinite(p).all() or (np.diff(p)<=0).any():raise ValueError('Calendar positions must be unique and increasing')
    return x,p


def hac_mean(d,positions,bandwidth=None):
    x,pos=validate_series(d,positions);n=len(x)
    if n<10:return np.nan,np.nan,np.nan
    q=min(n-1,int(np.floor(4*(n/100)**(2/9))) if bandwidth is None else int(bandwidth))
    if q<0:raise ValueError('Negative bandwidth')
    u=x-x.mean();meat=float(u@u)
    # Pair by actual calendar distance, not by row lag after observations are removed.
    for lag in range(1,q+1):
        left=np.searchsorted(pos,pos-lag);valid=left<n
        valid &= pos[np.minimum(left,n-1)]==pos-lag
        j=np.flatnonzero(valid)
        meat+=2*(1-lag/(q+1))*float(u[j]@u[left[j]])
    se=float(np.sqrt(max(0.,meat))/n)
    if se<=1e-16:return float(x.mean()),se,np.nan
    return float(x.mean()),se,float(2*norm.sf(abs(x.mean()/se)))


def segments(positions):
    pos=np.asarray(positions)
    return np.split(np.arange(len(pos)),np.flatnonzero(np.diff(pos)!=1)+1)


def block_indices(positions,block,rng):
    """Preserve original segment lengths; never silently reduce the chosen block."""
    _,pos=validate_series(np.zeros(len(positions)),positions)
    if block<1:raise ValueError('Invalid block')
    output=[]
    for segment in segments(pos):
        n=len(segment)
        if n<block:raise ValueError('insufficient_blocks: segment shorter than requested block')
        starts=rng.integers(n-block+1,size=int(np.ceil(n/block)))
        output.append(segment[(starts[:,None]+np.arange(block)).ravel()[:n]])
    return np.concatenate(output) if output else np.array([],int)


def fingerprint(a,b,rf,positions,**settings):
    h=hashlib.sha256()
    for x in [a,b,rf,positions]:
        z=np.asarray(x,dtype='<f8');h.update(str(z.shape).encode());h.update(z.tobytes())
    h.update(json.dumps(settings,sort_keys=True).encode());return h.hexdigest()


def stable_rng(key,seed):
    entropy=int(hashlib.sha256((str(seed)+key).encode()).hexdigest()[:16],16)
    return np.random.default_rng(entropy)


def interval_test(observed,draws,alpha=.05):
    """Centered-error symmetric interval and its corresponding two-sided test."""
    draws=np.asarray(draws,float);draws=draws[np.isfinite(draws)]
    if not len(draws) or not np.isfinite(observed):return dict(p=np.nan,low=np.nan,high=np.nan,status='unsupported')
    errors=draws-observed
    if np.max(abs(errors))<1e-15:return dict(p=np.nan,low=observed,high=observed,status='degenerate_difference')
    distance=np.sort(abs(errors));rank=int(np.ceil((1-alpha)*(len(draws)+1)))-1
    radius=distance[rank] if rank<len(distance) else np.inf
    p=(1+int((distance>=abs(observed)).sum()))/(len(draws)+1)
    return dict(p=float(p),low=float(observed-radius),high=float(observed+radius),status='evaluated')


def upper_es(loss,tail=.05):
    loss=np.sort(np.asarray(loss,float))[::-1];n=len(loss);count=n*tail
    k=int(np.floor(count));frac=count-k
    return float((loss[:k].sum()+(frac*loss[k] if k<n else 0.))/count)


def statistic_vector(a,b,rf):
    def one(r):
        excess=r-rf;sd=np.std(r,ddof=1)
        return np.array([np.expm1(np.log1p(r).mean()*252),excess.mean()/sd*np.sqrt(252) if sd>1e-15 else np.nan,
                         upper_es(-r),upper_es(-r,.01),np.maximum(-r-.02,0).mean(),np.mean(r<-.02)])
    return one(a)-one(b)


STAT_NAMES=['annual_return','sharpe','ES95','ES99','threshold_loss','threshold_probability']

def paired_inference(a,b,rf,positions,replicates=2000,block=10,seed=42):
    a,pos=validate_series(a,positions);b,_=validate_series(b,positions);rf,_=validate_series(rf,positions)
    if len(a)!=len(b) or len(a)!=len(rf) or (a<=-1).any() or (b<=-1).any():raise ValueError('Invalid paired returns')
    if replicates<100:raise ValueError('At least 100 draws required')
    key=fingerprint(a,b,rf,pos,replicates=replicates,block=block,seed=seed)
    out={'inference_key':key,'inference_status':'evaluated','block_length':block}
    mean,se,p=hac_mean(a-b,pos);out.update(mean_daily_difference=mean,hac_standard_error=se,hac_p=p)
    if len(a)<10:out['inference_status']='insufficient_sample';return out
    loss=np.maximum(-a-.02,0)-np.maximum(-b-.02,0)
    lm,ls,lp=hac_mean(loss,pos)
    out.update(threshold_loss_mean_difference=lm,threshold_loss_hac_se=ls,threshold_loss_hac_p=lp)
    observed=statistic_vector(a,b,rf)
    out.update({'delta_'+name:float(v) for name,v in zip(STAT_NAMES,observed)})
    if len(a)<10:out['inference_status']='insufficient_sample';return out
    if np.array_equal(a,b):
        out['inference_status']='identical_series'
        for name,v in zip(STAT_NAMES,observed):out.update({name+'_bootstrap_p':np.nan,name+'_ci_low':v,name+'_ci_high':v})
        return out
    rng=stable_rng(key,seed);draws=np.empty((replicates,len(STAT_NAMES)))
    try:
        for rep in range(replicates):
            ix=block_indices(pos,block,rng);draws[rep]=statistic_vector(a[ix],b[ix],rf[ix])
    except ValueError as exc:
        if 'insufficient_blocks' not in str(exc):raise
        out['inference_status']='insufficient_blocks';return out
    for j,name in enumerate(STAT_NAMES):
        result=interval_test(observed[j],draws[:,j]);out.update({name+'_bootstrap_p':result['p'],name+'_ci_low':result['low'],name+'_ci_high':result['high'],name+'_inference_status':result['status']})
    return out


def holm(values):
    x=np.asarray(values,float);out=np.full(len(x),np.nan);valid=np.flatnonzero(np.isfinite(x))
    if ((x[valid]<0)|(x[valid]>1)).any():raise ValueError('Invalid p-value')
    order=valid[np.argsort(x[valid])]
    if len(order):out[order]=np.minimum(1,np.maximum.accumulate(x[order]*(len(order)-np.arange(len(order)))))
    return out


def adjust_unique(frame,columns,family=('purpose','scope')):
    frame=frame.copy()
    for col in columns:
        if col not in frame:frame[col]=np.nan
        frame[col+'_holm']=np.nan
        for _,idx in frame.groupby(list(family)).groups.items():
            group=frame.loc[idx]
            for _,same in group.groupby('inference_key'):
                values=same[col].dropna().to_numpy()
                if len(values)>1 and np.ptp(values)>1e-12:raise ValueError('Identical inference key has conflicting p-values')
            unique=group.drop_duplicates('inference_key')
            adjusted=dict(zip(unique.inference_key,holm(unique[col])))
            frame.loc[idx,col+'_holm']=group.inference_key.map(adjusted)
    return frame


def calendar_acf(d,positions,lags=(1,5,10,20)):
    x,pos=validate_series(d,positions);u=x-x.mean();den=float(u@u);out=[]
    for lag in lags:
        ix=np.searchsorted(pos,pos-lag);valid=(ix<len(pos))&(pos[np.minimum(ix,len(pos)-1)]==pos-lag)
        j=np.flatnonzero(valid)
        out.append(dict(lag=lag,n_pairs=len(j),acf=float(u[j]@u[ix[j]]/den) if den>0 else np.nan))
    return out


def window_state_contrast(y,state,positions,bandwidth=3):
    y,pos=validate_series(y,positions);state=np.asarray(state,int)
    if state.shape!=y.shape or not np.isin(state,[0,1]).all():raise ValueError('Binary state required')
    if len(y)<10 or min(np.sum(state==0),np.sum(state==1))<3:return dict(status='insufficient_state_windows',contrast=np.nan,se=np.nan,p=np.nan)
    X=np.c_[np.ones(len(y)),state];bread=np.linalg.inv(X.T@X);beta=bread@X.T@y
    score=X*(y-X@beta)[:,None];meat=score.T@score
    for lag in range(1,bandwidth+1):
        ii=np.searchsorted(pos,pos-lag);valid=(ii<len(pos))&(pos[np.minimum(ii,len(pos)-1)]==pos-lag);j=np.flatnonzero(valid)
        cross=score[j].T@score[ii[j]];meat+=(1-lag/(bandwidth+1))*(cross+cross.T)
    cov=bread@meat@bread;se=float(np.sqrt(max(0,cov[1,1])))
    return dict(status='evaluated' if se>1e-16 else 'degenerate_difference',contrast=float(beta[1]),se=se,p=float(2*norm.sf(abs(beta[1]/se))) if se>1e-16 else np.nan)


def ljung_box(d,positions,max_lag=20):
    x,pos=validate_series(d,positions);n=len(x)
    if n<max_lag+5 or (np.diff(pos)!=1).any():return dict(Q=np.nan,p=np.nan,status='unsupported_calendar_or_sample')
    acf=calendar_acf(x,pos,range(1,max_lag+1));r=np.array([z['acf'] for z in acf])
    if not np.isfinite(r).all():return dict(Q=np.nan,p=np.nan,status='degenerate_difference')
    Q=float(n*(n+2)*np.sum(r*r/(n-np.arange(1,max_lag+1))))
    return dict(Q=Q,p=float(chi2.sf(Q,max_lag)),status='exploratory_portmanteau')
