"""Same-initial-state k=1 / heterogeneous-k comparisons, not rolling strategies."""
from pathlib import Path
import argparse
import json
import numpy as np
import pandas as pd
from scipy.integrate import quad
import step1,step2,step3


def closure_example():
    """Predeclared synthetic example: closure fails, cutwise additivity survives."""
    k=np.array([.5,1.5]);x=np.array([.5,.5])
    anchor=.5
    effective=np.log(np.sum(x*anchor**k))/np.log(anchor)
    rows=[]
    for lam in np.linspace(.01,.99,99):
        left=float(x@lam**k);right=float(2-x@lam**(1/k))
        rows.append(dict(lambda_level=lam,left_sum=left,right_sum=right,
            fitted_single_k=effective,single_left=lam**effective,
            single_right=2-lam**(1/effective),example='synthetic, endpoints (0,1,1,2), weights (.5,.5), k (.5,1.5)'))
    frame=pd.DataFrame(rows)
    if np.max(abs(frame.left_sum-frame.single_left))<1e-4:
        raise ValueError('Closure counterexample did not distinguish the families')
    exact=-sum(xi*((.4**(ki+1)-.1**(ki+1))/(2*(ki+1)*.15)) for xi,ki in zip(x,k))
    aggregate=-quad(lambda p:float(x@(2*p)**k),.05,.20,epsabs=1e-12)[0]/.15
    if abs(exact-aggregate)>1e-10:raise ValueError('Heterogeneous quantile additivity failed')
    return frame,dict(single_family_closure=False,quantile_integral_additivity_error=abs(exact-aggregate),
        interpretation='Minimum-t-norm nonnegative cut arithmetic; no claim for arbitrary probabilistic dependence')


def run(input_dir,stage2_dir,out):
    input_dir,stage2_dir,out=map(Path,[input_dir,stage2_dir,out])
    if out.exists() and any(out.iterdir()):raise FileExistsError('Use a fresh controlled output directory')
    policy=step2.load_policy(stage2_dir/'lp_policy_lock.json')
    contexts,_,signs=step2.load_inputs(input_dir,policy)
    returns=step1.read_daily_matrix(input_dir/'full_return_matrix.csv',True)
    turnover=step1.read_daily_matrix(input_dir/'full_turnover_matrix.csv',True)
    specs=[s for s in step2.experiment_specs(policy) if s['experiment_id'] in ['K_HETEROG','K1_NEUTRAL','FIXED_RULE_CONTROL']]
    rows=[];weights=[]
    for wid,c in contexts.items():
        for spec in specs:
            codes,x0,mu,rho,locks,forced,_,ep,ki=step3.coefficients(c,spec,pd.Series(dtype=float),returns,turnover,signs,policy)
            x,info=step2.solve_execution_modelA(mu,rho,c['rf_daily'],spec['target_annual'],x0,
                spec['exposure_floor'],policy,locks,forced,scales=spec['scales'])
            row=dict(window_id=wid,experiment_id=spec['experiment_id'],model_id=spec['model_id'],
                solver_status=info['status'],objective=info['objective'],fee=info['fee'],
                relaxation_scale=info['scale'],mean_mu=float(mu.mean()),mean_rho=float(rho.mean()),
                mean_k=float(ki.mean()),shared_initial='cash; reset independently every window',
                interpretation='One-decision coefficient/allocation diagnostic; not out-of-sample rolling welfare')
            rows.append(row)
            for j,code in enumerate(codes):
                weights.append(dict(window_id=wid,experiment_id=spec['experiment_id'],model_id=spec['model_id'],
                    stock_code=code,weight=float(x[j]) if x is not None else np.nan,
                    mu=float(mu[j]),rho=float(rho[j]),k_return=float(ki[j]),
                    a=float(ep[j,0]),b=float(ep[j,1]),c=float(ep[j,2]),d=float(ep[j,3])))
    frame=pd.DataFrame(rows);alloc=pd.DataFrame(weights);comparisons=[]
    for wid in contexts:
        for measure in ['RVaR','CVaR']:
            control=measure+'_heterog_control'
            records=frame[frame.window_id.eq(wid)&frame.experiment_id.eq('K_HETEROG')]
            aa=alloc[alloc.window_id.eq(wid)&alloc.experiment_id.eq('K_HETEROG')]
            for rec in records[~records.model_id.eq(control)].to_dict('records'):
                base=records[records.model_id.eq(control)]
                if base.empty:continue
                if not rec['model_id'].startswith(measure):continue
                x=aa[aa.model_id.eq(rec['model_id'])].set_index('stock_code').weight
                y=aa[aa.model_id.eq(control)].set_index('stock_code').weight
                comparisons.append(dict(window_id=wid,left_model=rec['model_id'],right_model=control,
                    both_accepted=rec['solver_status'] in ['Direct','Relaxed'] and base.iloc[0].solver_status in ['Direct','Relaxed'],
                    weight_l1=float(x.subtract(y,fill_value=0).abs().sum()) if len(x) and len(y) and x.notna().all() and y.notna().all() else np.nan,
                    mean_mu_difference=rec['mean_mu']-base.iloc[0].mean_mu,
                    mean_rho_difference=rec['mean_rho']-base.iloc[0].mean_rho))
    example,meta=closure_example();out.mkdir(parents=True,exist_ok=True)
    frame.to_csv(out/'controlled_same_cash_states.csv',index=False)
    alloc.to_parquet(out/'controlled_same_cash_weights.parquet',index=False)
    pd.DataFrame(comparisons).to_csv(out/'controlled_heterogeneity_comparisons.csv',index=False)
    example.to_csv(out/'synthetic_closure_example.csv',index=False)
    (out/'controlled_metadata.json').write_text(json.dumps(meta,indent=2))
    return frame


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for n in ['input','stage2','out']:p.add_argument('--'+n,type=Path,required=True)
    a=p.parse_args();run(a.input,a.stage2,a.out)
