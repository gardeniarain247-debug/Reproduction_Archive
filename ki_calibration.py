"""Conditional membership-projection diagnostics; never behavioral identification.
Historical empirical midranks define a declared target curve. Fitting this target
quantifies representation dispersion, not true asset-specific investor preferences.
"""
from pathlib import Path
import argparse
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
import step1
from step3 import prepare_output,finish_metadata


def default_policy():
    return dict(k_min=.05,k_max=20.,grid_size=161,quadrature_nodes=65,flat_tolerance=1e-12,
                weak_tolerance=1e-5,variants=[dict(name='median_balanced',core='median',weights=[.375,.25,.375]),
                dict(name='median_equal_shoulders',core='median',weights=[.5,0.,.5]),
                dict(name='mean_balanced',core='mean',weights=[.375,.25,.375])])


def membership(x,endpoints,k=1.,powers=None):
    a,b,c,d=np.asarray(endpoints,float)
    if not (a<b<=c<d):raise ValueError('Degenerate endpoints')
    x=np.asarray(x,float);left,right=(1/k,k) if powers is None else powers
    if left<=0 or right<=0:raise ValueError('Positive powers required')
    result=np.zeros_like(x);lo=(x>a)&(x<b);hi=(x>c)&(x<d)
    result[lo]=((x[lo]-a)/(b-a))**left;result[hi]=((d-x[hi])/(d-c))**right
    result[(x>=b)&(x<=c)]=1.;return result


class EmpiricalCurve:
    def __init__(self,values,core):
        values=np.asarray(values,float)
        if np.isinf(values).any():raise ValueError('infinite return')
        values=np.sort(values[np.isfinite(values)])
        if len(values)<2 or values[0]==values[-1]:raise ValueError('Constant or insufficient sample')
        self.values=values;self.nodes,counts=np.unique(values,return_counts=True)
        self.probabilities=(np.cumsum(counts)-counts/2)/len(values)
        self.core=float(np.median(values) if core=='median' else np.mean(values))
        if core=='median':
            # Anchor median membership even with ties; this is a declared target convention.
            i=np.searchsorted(self.nodes,self.core)
            if i<len(self.nodes) and self.nodes[i]==self.core:self.probabilities[i]=.5
            else:
                self.nodes=np.insert(self.nodes,i,self.core);self.probabilities=np.insert(self.probabilities,i,.5)
        self.core_probability=.5 if core=='median' else float(self.cdf(self.core))
    def cdf(self,x):return np.interp(np.asarray(x,float),self.nodes,self.probabilities,left=0.,right=1.)
    def __call__(self,x):
        p=self.cdf(x);q=self.core_probability
        return np.clip(np.where(p<=q,p/q,(1-p)/(1-q)),0,1)


def construct_acf(values,core='median'):
    if core not in ['mean','median']:raise ValueError('Unknown core convention')
    return EmpiricalCurve(values,core)


class ShapeObjective:
    def __init__(self,endpoints,target,weights=(.375,.25,.375),nodes=65):
        self.e=np.asarray(endpoints,float);a,b,c,d=self.e
        if not (a<b<=c<d):raise ValueError('Degenerate endpoints')
        self.weights=np.asarray(weights,float)
        if np.any(self.weights<0) or not np.isclose(self.weights.sum(),1):raise ValueError('Invalid integration weights')
        zz,ww=np.polynomial.legendre.leggauss(nodes);self.z=(zz+1)/2;self.w=ww/2
        self.left=np.asarray(target(a+(b-a)*self.z));self.right=np.asarray(target(d-(d-c)*self.z))
        core=np.asarray(target(b+(c-b)*self.z));self.core_loss=float(self.weights[1]*self.w@((1-core)**2))
    def _side(self,theta,side,reciprocal=False):
        th=np.asarray(theta,float);power=np.exp(-th if reciprocal else th)
        curve=self.z[None,:]**power.reshape(-1,1)
        y=self.left if side=='left' else self.right;weight=self.weights[0 if side=='left' else 2]
        out=weight*((curve-y)**2)@self.w
        return out.reshape(th.shape) if th.ndim else float(out[0])
    def __call__(self,theta):return self._side(theta,'left',True)+self._side(theta,'right')+self.core_loss
    def exponent_objective(self,side):return lambda theta:self._side(theta,side)


def fit_scalar(objective,policy):
    lo,hi=np.log([policy['k_min'],policy['k_max']]);grid=np.linspace(lo,hi,policy['grid_size'])
    values=np.asarray(objective(grid),float)
    if values.shape!=grid.shape:values=np.array([objective(t) for t in grid])
    if not np.isfinite(values).all():raise ValueError('Nonfinite shape loss')
    if np.ptp(values)<=policy['flat_tolerance']:
        return dict(theta=np.nan,k=np.nan,loss=float(values.min()),status='flat_objective',eligible=False,minima=0)
    candidates=[(lo,float(values[0])),(hi,float(values[-1]))]
    for j in range(1,len(grid)-1):
        if values[j]<=values[j-1] and values[j]<=values[j+1]:
            res=minimize_scalar(objective,bounds=(grid[j-1],grid[j+1]),method='bounded',options={'xatol':1e-11})
            candidates.append((float(res.x),float(res.fun)))
    best=min(v for _,v in candidates);near=[(t,v) for t,v in candidates if v<=best+1e-9*max(1.,abs(best))]
    th,value=min(near,key=lambda z:z[1]);boundary=min(abs(th-lo),abs(th-hi))<1e-7
    status='multiple_minima' if len(near)>1 else ('boundary' if boundary else ('weak_profile' if np.ptp(values)<policy['weak_tolerance'] else 'identified_conditional'))
    eligible=status=='identified_conditional'
    return dict(theta=th,k=float(np.exp(th)),loss=value,status=status,eligible=eligible,minima=len(near),profile_range=float(np.ptp(values)))


def mean_match(endpoints,mean):
    a,b,c,d=np.asarray(endpoints,float);lower=(a+c)/2;upper=(b+d)/2
    if not lower<mean<upper:return dict(k_mean=np.nan,mean_status='no_positive_finite_solution')
    k=(b+d-2*mean)/(2*mean-a-c)
    return dict(k_mean=float(k),mean_status='near_boundary' if min(mean-lower,upper-mean)<.001*(upper-lower) else 'positive_finite')


def analyse(values,endpoints,spec=None,policy=None,free=True):
    policy=default_policy() if policy is None else policy;spec=policy['variants'][0] if spec is None else spec
    try:
        e=np.asarray(endpoints,float)
        if not np.isfinite(e).all() or not e[0]<e[1]<=e[2]<e[3]:raise ValueError('Degenerate endpoints')
        curve=construct_acf(values,spec['core']);obj=ShapeObjective(e,curve,spec['weights'],policy['quadrature_nodes'])
    except ValueError as exc:
        if 'infinite' in str(exc):raise
        return dict(k=np.nan,loss=np.nan,status='degenerate_calibration',eligible=False),None
    result=fit_scalar(obj,policy);result.update(mean_match(e,np.nanmean(values)))
    if free:
        left=fit_scalar(obj.exponent_objective('left'),policy);right=fit_scalar(obj.exponent_objective('right'),policy)
        result.update(free_left_power=left['k'],free_right_power=right['k'],free_loss=left['loss']+right['loss']+obj.core_loss,
                      restriction_loss=result['loss']-left['loss']-right['loss']-obj.core_loss)
    return result,obj


def run(input_dir,out_dir,overwrite=False,window_limit=None):
    out=prepare_output(out_dir,'ki_',overwrite);input_dir=Path(input_dir);policy=default_policy()
    returns=step1.read_daily_matrix(input_dir/'full_return_matrix.csv',True)
    endpoints=step1.read_result_table(input_dir/'endpoint_calibration.csv',True)
    windows=step1.read_result_table(input_dir/'rolling_windows.csv')
    if window_limit:windows=windows.head(window_limit)
    rows=[];common=[]
    for win in windows.itertuples(index=False):
        sample=returns.loc[pd.Timestamp(win.est_start):pd.Timestamp(win.est_end)]
        base=endpoints[endpoints.window_id.eq(win.window_id)&endpoints.endpoint_scheme.eq('baseline')]
        for variant in policy['variants']:
            objectives=[];individual=[];ids=[]
            for row in base.itertuples(index=False):
                values=sample[row.stock_code].dropna().to_numpy();e=np.array([row.a,row.b,row.c,row.d])
                result,obj=analyse(values,e,variant,policy)
                rows.append(dict(window_id=win.window_id,stock_code=row.stock_code,variant=variant['name'],**result))
                if obj is not None:objectives.append(obj);individual.append(result['loss']);ids.append(row.stock_code)
            if objectives:
                fitted=fit_scalar(lambda th:np.mean([obj(th) for obj in objectives],axis=0),policy)
                common.append(dict(window_id=win.window_id,variant=variant['name'],n_assets=len(ids),common_k=fitted['k'],common_loss=fitted['loss'],
                    individual_mean_loss=float(np.mean(individual)),common_restriction_loss=fitted['loss']-float(np.mean(individual)),status=fitted['status']))
    assets=pd.DataFrame(rows);summary=[]
    for variant,g in assets.groupby('variant'):
        eligible=g[g.eligible.fillna(False)]
        summary.append(dict(variant=variant,total_records=len(g),eligible_records=len(eligible),median_k=eligible.k.median(),iqr_k=eligible.k.quantile(.75)-eligible.k.quantile(.25),
            q05_k=eligible.k.quantile(.05),q95_k=eligible.k.quantile(.95),interpretation='conditional empirical-membership projection; not investor preference estimates'))
    outputs=[]
    for name,frame in [('ki_asset_estimates.csv',assets),('ki_conditional_dispersion.csv',pd.DataFrame(summary)),('ki_common_shape_restriction.csv',pd.DataFrame(common))]:
        path=out/name;frame.to_csv(path,index=False,encoding='utf-8-sig');outputs.append(path)
    return finish_metadata(out,'ki_run_metadata.json',dict(status='computed_conditional_shape_diagnostics',pipeline_model='ModelA',scope='pilot' if window_limit else 'full',
        source_endpoint_sha256=step1.sha256_file(input_dir/'endpoint_calibration.csv'),source_returns_sha256=step1.sha256_file(input_dir/'full_return_matrix.csv'),code_sha256=step1.sha256_file(__file__),policy=policy,
        estimand='dispersion of fitted representations conditional on target-curve convention, not identified behavioral ki'),outputs)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--input',type=Path,default=Path(__file__).parent/'results1');p.add_argument('--out',type=Path,default=Path(__file__).parent/'results_ki');p.add_argument('--overwrite',action='store_true');p.add_argument('--window-limit',type=int)
    a=p.parse_args();print(run(a.input,a.out,a.overwrite,a.window_limit)['status'])
if __name__=='__main__':main()
