"""Remaining reviewer diagnostics under Model A: controlled tail path,
endpoint uncertainty at actual reference states, and conditional shape calibration.
These resampled one-window optimizations are not independent OOS strategies.
"""
from pathlib import Path
import argparse
import numpy as np
import pandas as pd
import step1,step2,step3,ki_calibration


def controlled_substitution(contexts,policy,turnover):
    rows=[];weights=[]
    for wid,c in contexts.items():
        ep=c['schemes']['baseline'];codes=c['stocks'];liquid=turnover.loc[c['win'].est_end].reindex(codes).to_numpy(float)
        locks=np.where(np.isfinite(liquid)&(liquid>0),np.nan,0.)
        for k,target,floor in [(.5,.12,.9),(.8,.08,.5),(1.1,.05,.1)]:
            mu,risks=step2.three_measure_coefficients(ep,k)
            for theta in [0.,.25,.5,.75,1.]:
                rho=theta*risks['CVaR05']+(1-theta)*risks['RVaR']
                x,info=step2.solve_execution_modelA(mu,rho,c['rf_daily'],target,np.zeros(len(codes)),floor,policy,locks=locks)
                row=dict(window_id=wid,k=k,theta=theta,solver_status=info['status'],shared_initial_rule='cash for all criteria; no chaining',coherent_spectral_weight=theta>=.25,
                         objective=info['objective'],fee=info['fee'],relaxation_scale=info['scale'])
                for label,v in risks.items():row['cross_risk_'+label]=float(v@x-c['rf_daily']*info['cash']+info['fee']) if x is not None else np.nan
                rows.append(row)
                if x is not None:
                    for code,weight in zip(codes,x):weights.append(dict(window_id=wid,k=k,theta=theta,stock_code=code,weight=weight))
    return pd.DataFrame(rows),pd.DataFrame(weights)


def endpoint_uncertainty(input_dir,contexts,policy,assets,turnover,returns,signs,replicates,blocks):
    """Recompute endpoints/coefficients and solve at fixed actual initial states."""
    estimates=[];intervals=[];shape_draws=[]
    specs=[s for s in step2.experiment_specs(policy) if s['experiment_id']=='BASELINE_AUDIT' and s['risk_measure']!='MV']
    with np.load(Path(input_dir)/'bootstrap_indices.npz',allow_pickle=False) as frozen:
        for wid,c in contexts.items():
            selected_shape=None
            for code in sorted(c['stocks']):
                ep=c['schemes']['baseline'][c['stocks'].index(code)]
                if ep[0]<ep[1]<=ep[2]<ep[3]:selected_shape=code;break
            for block in blocks:
                indices=frozen[f'indices_win{wid}_block{block}'][:replicates]
                if len(indices)!=replicates:raise ValueError('Stage1 bootstrap sample count insufficient')
                np.testing.assert_array_equal(frozen[f'dates_win{wid}'],c['training'].index.to_numpy(dtype='datetime64[ns]'))
                if indices.shape[1]!=250 or indices.min()<0 or indices.max()>=250:raise ValueError('Invalid frozen indices')
                # A declared one-asset-per-window conditional calibration uncertainty diagnostic.
                if selected_shape:
                    source=c['training'][selected_shape].to_numpy()
                    for rep,ix in enumerate(indices):
                        values=source[ix];e=np.nanquantile(values,[.05,.25,.75,.95])
                        fit,_=ki_calibration.analyse(values,e,free=False)
                        shape_draws.append(dict(window_id=wid,stock_code=selected_shape,block_length=block,replicate=rep,**fit))
                cash_rates=step1.read_daily_matrix(Path(input_dir)/'risk_free_rate.csv').iloc[:,0]
                hold_dates=returns.loc[c['win'].test_start:c['win'].test_end].index
                cash_growth=float(np.prod(1+cash_rates.reindex(hold_dates)))
                for spec in specs:
                    ref=assets[assets.window_id.eq(wid)&assets.model_id.eq(spec['model_id'])&assets.experiment_id.eq('BASELINE_AUDIT')].set_index('stock_code')
                    if ref.empty:raise ValueError('Missing actual initial reference state')
                    previous=ref.initial_weight
                    codes,old,mu0,rho0,locks,forced,sigma,ep0,ki=step3.coefficients(c,spec,previous,returns,turnover,signs,policy)
                    ref_x=ref.posttrade_amount.reindex(codes,fill_value=0).to_numpy(float)
                    train=returns.loc[c['win'].est_start:c['win'].est_end,codes].to_numpy(float)
                    coefficient_draws=[];endpoint_draws=[];weight_draws=[]
                    for rep,ix in enumerate(indices):
                        ep=np.zeros_like(ep0);use=~forced
                        ep[use]=np.nanquantile(train[ix][:,use],step1.ENDPOINT_SCHEMES['baseline'],axis=0,method='linear').T
                        invalid=not np.isfinite(ep).all()
                        if invalid:
                            estimates.append(dict(window_id=wid,model_id=spec['model_id'],block_length=block,replicate=rep,solver_status='CalibrationFailed',failure_reason='nonfinite_resampled_endpoint',weight_l1=np.nan,mu_mae=np.nan,rho_mae=np.nan,conditional_window_return=np.nan,conditional_valuation_status='not_optimized'));continue
                        mu,rho=step2.portfolio_coefficients(ep,ki,spec['risk_measure'],spec['interval'],spec['gamma']);mu[forced]=0.;rho[forced]=0.
                        x,info=step2.solve_execution_modelA(mu,rho,c['rf_daily'],spec['target_annual'],old,spec['exposure_floor'],policy,locks,forced,scales=spec['scales'])
                        conditional_return=np.nan;valuation='not_optimized'
                        if x is not None:
                            active=x>1e-14;observed=returns.loc[hold_dates,np.array(codes)[active]].to_numpy(float)
                            if np.isfinite(observed).all() and (observed>=-1).all():
                                conditional_return=float(np.prod(1+observed,axis=0)@x[active]+cash_growth*info['cash']-1);valuation='complete'
                            else:valuation='incomplete_holding_quotes'
                        estimates.append(dict(window_id=wid,model_id=spec['model_id'],block_length=block,replicate=rep,solver_status=info['status'],relaxation_scale=info['scale'],conditional_window_return=conditional_return,conditional_valuation_status=valuation,
                            weight_l1=float(abs(x-ref_x).sum()) if x is not None else np.nan,
                            mu_mae=float(abs(mu-mu0).mean()),rho_mae=float(abs(rho-rho0).mean()),
                            reference_risk_coefficient_shift=float((rho-rho0)@ref_x),scope='conditional fixed actual initial holdings; not a rolling strategy'))
                        coefficient_draws.append(np.c_[mu,rho]);endpoint_draws.append(ep);weight_draws.append(x if x is not None else np.full(len(codes),np.nan))
                    if coefficient_draws:
                        cd=np.asarray(coefficient_draws);ed=np.asarray(endpoint_draws);xd=np.asarray(weight_draws)
                        # Fixed-k endpoint -> risk coefficients are exactly linear. Verify independently.
                        for j,code in enumerate(codes):
                            if forced[j]:continue
                            # Lower-tail coefficient is -(1-omega)*a-omega*b.
                            # Recover omega on an ordered canonical trapezoid, then
                            # verify exact linear propagation on independent draws.
                            _,canonical=step2.portfolio_coefficients(np.array([[0.,1.,1.,2.]]),ki[j],spec['risk_measure'],spec['interval'],spec['gamma'])
                            omega=-canonical[0];grad=np.array([-(1-omega),-omega,0.,0.])
                            predicted=(ed[:,j]-ep0[j])@grad
                            residual=float(np.max(abs(predicted-(cd[:,j,1]-rho0[j]))))
                            if residual>1e-10:raise ValueError('Endpoint-to-risk linear propagation check failed')
                            row=dict(window_id=wid,model_id=spec['model_id'],block_length=block,stock_code=code,n_calibrated_draws=len(cd),n_accepted_weight_draws=int(np.isfinite(xd[:,j]).sum()),propagation_residual=residual)
                            for label,v in [('mu',cd[:,j,0]),('rho',cd[:,j,1])]+[(name,ed[:,j,col]) for col,name in enumerate('abcd')]:
                                row[label+'_q025'],row[label+'_q975']=np.quantile(v,[.025,.975]);row[label+'_sd']=np.std(v,ddof=1) if len(v)>1 else np.nan
                            valid=xd[:,j][np.isfinite(xd[:,j])]
                            row['weight_q025'],row['weight_q975']=np.quantile(valid,[.025,.975]) if len(valid) else [np.nan,np.nan]
                            intervals.append(row)
                print(f'Conditional uncertainty window={wid}, block={block}, draws={replicates}',flush=True)
    return pd.DataFrame(estimates),pd.DataFrame(intervals),pd.DataFrame(shape_draws)


def run(input_dir,stage2_dir,stage3_dir,out_dir,overwrite=False,replicates=300,window_limit=None,run_shape=True):
    if not 1<=replicates<=300:raise ValueError('Replicates must be 1..300')
    source=step3.checked_manifest(stage3_dir,'step3_run_metadata.json','accepted_modelA_stage3')
    if source.get('execution_variant')=='next_close':
        raise ValueError('Formation-close conditional bootstrap cannot value next-close execution. Use baseline reference diagnostics only.')
    if source.get('source_stage2_metadata_sha256')!=step1.sha256_file(Path(stage2_dir)/'step2_run_metadata.json'):raise ValueError('Stage2/3 mismatch')
    policy=step2.load_policy(Path(stage2_dir)/'lp_policy_lock.json')
    contexts,meta,signs=step2.load_inputs(input_dir,policy)
    if window_limit:contexts=dict(list(contexts.items())[:window_limit])
    if not window_limit and replicates!=300:raise ValueError('Formal endpoint uncertainty uses 300 frozen draws; reduced counts require pilot window limit')
    out=step3.prepare_output(out_dir,'remaining_',overwrite)
    returns=step1.read_daily_matrix(Path(input_dir)/'full_return_matrix.csv',True);turn=step1.read_daily_matrix(Path(input_dir)/'full_turnover_matrix.csv',True)
    assets=pd.read_parquet(Path(stage3_dir)/'step3_assets.parquet')
    control,weights=controlled_substitution(contexts,policy,turn)
    estimates,intervals,shape=endpoint_uncertainty(input_dir,contexts,policy,assets,turn,returns,signs,replicates,[5,10])
    summary=estimates.groupby(['model_id','block_length','solver_status']).agg(draws=('replicate','size'),median_weight_l1=('weight_l1','median'),median_mu_mae=('mu_mae','median'),median_rho_mae=('rho_mae','median'),median_conditional_window_return=('conditional_window_return','median')).reset_index()
    shapes=[]
    if not shape.empty:
        for (wid,code,block),g in shape.groupby(['window_id','stock_code','block_length']):
            v=g.loc[g.eligible.fillna(False),'k'].dropna()
            shapes.append(dict(window_id=wid,stock_code=code,block_length=block,eligible_draws=len(v),total_draws=len(g),
                k_q025=v.quantile(.025),k_q975=v.quantile(.975),interpretation='conditional target-curve calibration, one selected asset per window; not cross-sectional preference uncertainty'))
    outputs=[]
    for name,frame in [('remaining_controlled_tail_path.csv',control),('remaining_controlled_weights.parquet',weights),('remaining_endpoint_draws.parquet',estimates),('remaining_endpoint_intervals.csv',intervals),('remaining_endpoint_summary.csv',summary),('remaining_shape_bootstrap.parquet',shape),('remaining_shape_uncertainty.csv',pd.DataFrame(shapes))]:
        path=out/name
        if path.suffix=='.parquet':frame.to_parquet(path,index=False)
        else:frame.to_csv(path,index=False,encoding='utf-8-sig')
        outputs.append(path)
    if run_shape:
        ki_calibration.run(input_dir,out/'shape_calibration',overwrite,window_limit)
    return step3.finish_metadata(out,'remaining_run_metadata.json',dict(status='computed_modelA_remaining_diagnostics',pipeline_model='ModelA',scope='pilot' if window_limit else 'full',
        endpoint_replicates=replicates,source_stage3_metadata_sha256=step1.sha256_file(Path(stage3_dir)/'step3_run_metadata.json'),code_sha256=step1.sha256_file(__file__),
        limitations='conditional calibration and fixed-reference uncertainty; not independently identified preferences or independent OOS replications'),outputs)


def main():
    base=Path(__file__).parent;p=argparse.ArgumentParser(description=__doc__)
    for key,default in [('input','results1'),('stage2','results2'),('stage3','results3'),('out','results_remaining')]:p.add_argument('--'+key,type=Path,default=base/default)
    p.add_argument('--replicates',type=int,default=300);p.add_argument('--window-limit',type=int);p.add_argument('--overwrite',action='store_true')
    a=p.parse_args();print(run(a.input,a.stage2,a.stage3,a.out,a.overwrite,a.replicates,a.window_limit)['status'])
if __name__=='__main__':main()
