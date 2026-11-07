"""Model A continuous-policy performance and paired inference; no Model B inputs."""
import argparse
from pathlib import Path
from datetime import datetime,timezone
import json
import numpy as np
import pandas as pd
from scipy.stats import norm
import empirical_statistics as es
import step1
from step3 import checked_manifest,prepare_output,finish_metadata
BASE_DIR=Path(__file__).resolve().parent


def tail_average(loss,a,b):
    """Integral of empirical inverse CDF over [a,b], including fractional boundary weights."""
    x=np.sort(np.asarray(loss,float));n=len(x)
    if not n:return np.nan
    left=np.arange(n)/n;right=(np.arange(n)+1)/n
    weights=np.maximum(0,np.minimum(right,b)-np.maximum(left,a))
    return float(weights@x/(b-a))


def metrics(r,rf,positions=None):
    r=np.asarray(r,float);rf=np.asarray(rf,float);n=len(r)
    if not n:return {k:np.nan for k in ['annual_return','annual_volatility','sharpe','sortino','max_drawdown','VaR95','ES95','RVaR_loss_80_95']}
    if not np.isfinite(r).all() or (r<=-1).any():raise ValueError('Invalid performance returns')
    if rf.shape!=r.shape or not np.isfinite(rf).all():raise ValueError('Invalid paired cash-rate vector')
    excess=r-rf;std=np.std(r,ddof=1) if n>1 else np.nan
    down=np.sqrt(np.mean(np.minimum(excess,0)**2))
    # Subset diagnostics never bridge excluded windows for drawdowns.
    positions=np.arange(n) if positions is None else np.asarray(positions)
    cuts=np.r_[0,np.flatnonzero(np.diff(positions)!=1)+1,n];mdd=0.
    for a,b in zip(cuts[:-1],cuts[1:]):
        wealth=np.r_[1.,np.cumprod(1+r[a:b])]
        mdd=max(mdd,float(np.max(1-wealth/np.maximum.accumulate(wealth))))
    return dict(annual_return=float(np.expm1(np.log1p(r).mean()*252)),annual_volatility=float(std*np.sqrt(252)),
        sharpe=float(excess.mean()/std*np.sqrt(252)) if std>0 else np.nan,
        sortino=float(excess.mean()/down*np.sqrt(252)) if down>0 else np.nan,
        max_drawdown=mdd,VaR95=float(np.quantile(-r,.95)),ES95=tail_average(-r,.95,1),RVaR_loss_80_95=tail_average(-r,.8,.95))


def hac_mean(d,positions,bandwidth=None):
    return es.hac_mean(d,positions,bandwidth)


def block_indices(positions,block,rng):
    return es.block_indices(positions,block,rng)


def pair_definitions(daily):
    available=set(zip(daily.experiment_id,daily.model_id));result=[];seen=set()
    def add(left,right,purpose):
        if left in available and right in available and left!=right and (left,right,purpose) not in seen:
            seen.add((left,right,purpose));result.append(dict(left_experiment=left[0],left_model=left[1],right_experiment=right[0],right_model=right[1],purpose=purpose))
    for e,m in sorted(available):
        if m.startswith('RVaR') and not m.endswith('_EMEW'):
            cv=m.replace('RVaR','CVaR',1)
            if e=='RVAR_INTERVAL':cv='CVaR_interval_control_'+m.rsplit('_',1)[-1]
            add((e,m),(e,cv),'rvar_cvar');add((e,m),(e,m+'_EMEW'),'allocation_channel')
            if e=='BASELINE_AUDIT':
                add((e,m),('BASELINE_AUDIT','MV'),'mv_reference');add((e,m),('BENCHMARK','SH50'),'index_reference')
        for measure in ['RVaR','CVaR']:
            if m.startswith(measure) and not m.endswith('_EMEW'):
                if e=='K_HETEROG' and not m.endswith('_control'):add((e,m),(e,measure+'_heterog_control'),'heterogeneous_vs_control')
                if e=='ENDPOINT_ALT' and '_baseline_' not in m:add((e,m),(e,measure+'_endpoint_baseline_'+m.rsplit('_',1)[-1]),'endpoint_vs_control')
                if e=='RVAR_INTERVAL' and measure=='RVaR' and '_baseline_' not in m:add((e,m),(e,'RVaR_interval_baseline_'+m.rsplit('_',1)[-1]),'interval_vs_control')
                if e=='KPURE_CONTROL' and not m.endswith('kpure05'):add((e,m),(e,measure+'_kpure05'),'pure_k_vs_05')
    for suffix,label in [('05','opt'),('08','neu'),('11','pes')]:
        a=('BASELINE_AUDIT','RVaR_'+label);b=('BASELINE_AUDIT','CVaR_'+label);c=('CVAR02_ROBUSTNESS','CVaR02_k'+suffix)
        add(a,c,'rvar_cvar20');add(b,c,'cvar05_cvar20')
    for measure in ['RVaR','CVaR']:
        for suffix in ['05','08','11']:
            add(('K1_NEUTRAL',measure+'_neutral_k1'),('FIXED_RULE_CONTROL',measure+'_fixed_k'+suffix),'neutral_vs_fixed_shape')
    for i,row in enumerate(result):row['pair_id']=f'P{i:03d}'
    return pd.DataFrame(result)


def holm(values):
    return es.holm(values)


def run(stage3_dir,out_dir,overwrite=False,replicates=2000,block=10,seed=42,allow_pilot=False,sensitivity=True):
    if replicates<100 or block<1:raise ValueError('Need >=100 replicates and positive block length')
    if not sensitivity and not allow_pilot:raise ValueError('Formal inference requires sensitivities; --no-sensitivity is pilot-only')
    source=checked_manifest(stage3_dir,'step3_run_metadata.json','accepted_modelA_stage3')
    if source.get('pipeline_revision')!='review_completion_v1':raise ValueError('Rerun the supplied Step3 with coefficient/structure records')
    if source.get('pipeline_model')!='ModelA':raise ValueError('Model A inputs required')
    if source['scope']!='full' and not allow_pilot:raise ValueError('Pilot input requires --allow-pilot')
    out=prepare_output(out_dir,'step4_',overwrite);d=pd.read_parquet(Path(stage3_dir)/'step3_daily_returns.parquet')
    d['date']=pd.to_datetime(d.date)
    dates=sorted(d.date.unique());positions={date:i for i,date in enumerate(dates)};d['position']=d.date.map(positions)
    w=pd.read_csv(Path(stage3_dir)/'step3_window_backtests.csv')
    summaries=[]
    for (e,m),g in d.groupby(['experiment_id','model_id'],sort=False):
        g=g.sort_values('date');z=w[(w.experiment_id==e)&(w.model_id==m)]
        summaries.append(dict(experiment_id=e,model_id=m,n_days=len(g),n_windows=g.window_id.nunique(),
            failed_optimization_windows=int(z.solver_status.eq('Failed').sum()),relaxed_windows=int(z.solver_status.eq('Relaxed').sum()),
            performance_scope='full executable policy including disclosed hold fallback',
            mean_turnover=float(z.turnover.mean()) if 'turnover' in z else np.nan,
            mean_risky_exposure=float(g.risky_exposure.mean()) if 'risky_exposure' in g else np.nan,
            **metrics(g.net_return,g.rf_daily,g.position)))
    definitions=pair_definitions(d);pairs=[];cache={};sensitivity_rows=[];acf_rows=[]
    for rec in definitions.to_dict('records'):
        left=d[(d.experiment_id==rec['left_experiment'])&(d.model_id==rec['left_model'])]
        right=d[(d.experiment_id==rec['right_experiment'])&(d.model_id==rec['right_model'])]
        both=left.merge(right,on='date',suffixes=('_left','_right'),validate='one_to_one').sort_values('date')
        print(f"Inference {rec['pair_id']}: {rec['left_model']} vs {rec['right_model']}",flush=True)
        for scope in ['full_policy','both_direct_only']:
            z=both if scope=='full_policy' else both[both.direct_eligible_left.fillna(False)&both.direct_eligible_right.fillna(False)]
            if not np.allclose(z.rf_daily_left,z.rf_daily_right,rtol=0,atol=1e-14):raise ValueError('Paired strategies have different cash-rate calendars')
            a=z.net_return_left.to_numpy();b=z.net_return_right.to_numpy();rf=z.rf_daily_left.to_numpy();pos=z.position_left.to_numpy()
            row=dict(rec,scope=scope,n_days=len(z),n_windows=z.window_id_left.nunique(),
                     interpretation='full executable policies' if scope=='full_policy' else 'selected direct-only windows; not full-period performance')
            if len(a):
                ma=metrics(a,rf,pos);mb=metrics(b,rf,pos)
                for key in ma:row['delta_'+key]=ma[key]-mb[key]
            key=es.fingerprint(a,b,rf,pos,replicates=replicates,block=block,seed=seed)
            if key not in cache:cache[key]=es.paired_inference(a,b,rf,pos,replicates,block,seed)
            row.update(cache[key]);pairs.append(row)
            if (rec['purpose']=='rvar_cvar' and rec['left_experiment']=='BASELINE_AUDIT' and scope=='full_policy'):
                for ar in es.calendar_acf(a-b,pos):acf_rows.append(dict(rec,scope=scope,**ar))
                for q in [5,10,20]:
                    box=es.ljung_box(a-b,pos,q);acf_rows.append(dict(rec,scope=scope,lag=q,ljung_box_Q=box['Q'],ljung_box_p=box['p'],diagnostic_status=box['status']))
                if sensitivity:
                    for length in [5,10,20,40]:
                        result=es.paired_inference(a,b,rf,pos,replicates,length,seed)
                        sensitivity_rows.append(dict(rec,scope=scope,sensitivity='block',setting=length,**result))
                    for lag in [5,10,20]:
                        mean,se,pv=es.hac_mean(a-b,pos,lag)
                        sensitivity_rows.append(dict(rec,scope=scope,sensitivity='HAC_lag',setting=lag,mean_daily_difference=mean,hac_standard_error=se,hac_p=pv))
    pairs=pd.DataFrame(pairs)
    pairs=es.adjust_unique(pairs,['hac_p','sharpe_bootstrap_p','ES95_bootstrap_p','ES99_bootstrap_p','threshold_loss_bootstrap_p','threshold_probability_bootstrap_p','threshold_loss_hac_p'])
    outputs=[]
    tables=[('step4_performance.csv',pd.DataFrame(summaries)),('step4_paired_comparisons.csv',pairs),('step4_pair_definitions.csv',definitions),
            ('step4_feasibility.csv',w.groupby(['experiment_id','model_id','solver_status']).size().rename('windows').reset_index())]
    tables.extend([('step4_inference_sensitivity.csv',pd.DataFrame(sensitivity_rows)),('step4_difference_acf.csv',pd.DataFrame(acf_rows))])
    # Three-strategy common-date summaries, including the same three direct-only window subset.
    trios=[]
    for suffix,label in [('05','opt'),('08','neu'),('11','pes')]:
        ids=[('BASELINE_AUDIT','CVaR_'+label),('BASELINE_AUDIT','RVaR_'+label),('CVAR02_ROBUSTNESS','CVaR02_k'+suffix)]
        groups=[d[(d.experiment_id==e)&(d.model_id==m)].set_index('date') for e,m in ids]
        if any(g.empty for g in groups):continue
        common=groups[0].index.intersection(groups[1].index).intersection(groups[2].index)
        direct=common
        for g in groups:direct=direct.intersection(g.index[g.direct_eligible.fillna(False)])
        for scope,cal in [('full_policy',common),('all_three_direct_only',direct)]:
            cal=cal.sort_values()
            for (e,m),g in zip(ids,groups):
                z=g.loc[cal];trios.append(dict(k_suffix=suffix,experiment_id=e,model_id=m,scope=scope,n_days=len(z),**metrics(z.net_return,z.rf_daily,z.position)))
    tables.append(('step4_substitution_common_samples.csv',pd.DataFrame(trios)))
    stability=[];regimes=[]
    market=d[d.model_id.eq('SH50')].groupby('window_id').net_return.apply(lambda x: float(np.prod(1+x)-1))
    for rec in definitions.to_dict('records'):
        left=w[(w.experiment_id==rec['left_experiment'])&(w.model_id==rec['left_model'])]
        right=w[(w.experiment_id==rec['right_experiment'])&(w.model_id==rec['right_model'])]
        z=left.merge(right,on='window_id',suffixes=('_left','_right'),validate='one_to_one')
        if z.empty:continue
        diff=z.window_total_return_left-z.window_total_return_right
        stability.append(dict(rec,n_windows=len(z),left_win_rate=float(diff.gt(0).mean()),mean_window_difference=float(diff.mean()),
            both_direct_windows=int((z.solver_status_left.eq('Direct')&z.solver_status_right.eq('Direct')).sum())))
        for state in ['nonnegative_market','negative_market']:
            mask=z.window_id.map(market).ge(0) if state=='nonnegative_market' else z.window_id.map(market).lt(0)
            v=diff[mask]
            if len(v):regimes.append(dict(rec,state=state,n_windows=len(v),mean_window_difference=float(v.mean()),left_win_rate=float(v.gt(0).mean()),
                interpretation='descriptive conditioning on realized index window return; not a predictive trading signal'))
    tables.extend([('step4_window_stability.csv',pd.DataFrame(stability)),('step4_state_descriptive.csv',pd.DataFrame(regimes))])

    tables.extend(additional_tables(d,w,definitions,stage3_dir))
    tables.append(('step4_neutral_reference.csv',pairs[pairs.left_experiment.eq('K1_NEUTRAL')]))
    for name,frame in tables:
        path=out/name;frame.to_csv(path,index=False,encoding='utf-8-sig');outputs.append(path)
    meta=dict(stage=4,status='accepted_modelA_stage4',pipeline_model='ModelA',pipeline_revision='review_completion_v1',scope=source['scope'],
              source_stage3_metadata_sha256=step1.sha256_file(Path(stage3_dir)/'step3_run_metadata.json'),
              statistics_code_sha256=step1.sha256_file(es.__file__),code_sha256=step1.sha256_file(__file__),started_utc=datetime.now(timezone.utc).isoformat(),
              inference=dict(replicates=replicates,block_length=block,seed=seed,hac='Bartlett Newey-West with automatic bandwidth; no cross-gap autocovariance',
              bootstrap='paired moving blocks preserving segment lengths; short segments flagged, block never reduced',holm_family='unique data fingerprints within purpose x scope x statistic',
              sharpe_p='two-sided centered-error paired bootstrap with matching symmetric centered-error CI',minimum_sample=10),
              result_scope='full-policy results include holds after infeasible optimization; direct-only results are selected-sample diagnostics')
    return finish_metadata(out,'step4_run_metadata.json',meta,outputs)



def additional_tables(d,w,definitions,stage3_dir):
    assets=pd.read_parquet(Path(stage3_dir)/'step3_assets.parquet')
    structures=[]
    for (e,m,wid),g in assets.groupby(['experiment_id','model_id','window_id']):
        x=g.posttrade_amount.to_numpy(float);total=x.sum();positive=x[x>1e-12]
        if total>1e-12:
            weights=positive/total;hhi=float(weights@weights);maximum=float(weights.max());neff=1/hhi
            if hhi>maximum+1e-10 or neff>len(positive)+1e-8:raise ValueError('Concentration identity failed')
        else:hhi=maximum=neff=np.nan
        structures.append(dict(experiment_id=e,model_id=m,window_id=wid,risky_amount=total,hhi_risky=hhi,effective_n=neff,max_normalized_weight=maximum,occupied_assets=len(positive)))
    structure=pd.DataFrame(structures)
    summary=structure.groupby(['experiment_id','model_id']).agg(mean_hhi=('hhi_risky','mean'),mean_effective_n=('effective_n','mean'),mean_max_normalized_weight=('max_normalized_weight','mean'),mean_risky_amount=('risky_amount','mean')).reset_index()
    # Realized states are post-hoc descriptions; labels never enter an optimization.
    benchmark=d[d.model_id.eq('SH50')]
    market=[]
    for wid,g in benchmark.groupby('window_id'):
        market.append(dict(window_id=wid,market_return=float(np.prod(1+g.net_return)-1),market_vol=float(g.net_return.std(ddof=1)),hold_start=g.date.min()))
    market=pd.DataFrame(market);state_rows=[];contrasts=[]
    if not market.empty:
        market['Up']=market.market_return.ge(0);market['HighVol']=market.market_vol.ge(market.market_vol.median());market['Post2021']=market.hold_start.ge(pd.Timestamp('2021-01-01'))
        for rec in definitions.to_dict('records'):
            if rec['purpose']!='rvar_cvar' or rec['left_experiment']!='BASELINE_AUDIT':continue
            left=d[(d.experiment_id==rec['left_experiment'])&(d.model_id==rec['left_model'])]
            right=d[(d.experiment_id==rec['right_experiment'])&(d.model_id==rec['right_model'])]
            z=left.merge(right,on='date',suffixes=('_left','_right'),validate='one_to_one').sort_values('date')
            win=z.groupby('window_id_left')[['net_return_left','net_return_right']].apply(lambda a:float(np.prod(1+a.net_return_left)-np.prod(1+a.net_return_right))).rename('difference').reset_index().rename(columns={'window_id_left':'window_id'})
            win=win.merge(market,on='window_id').sort_values('window_id')
            for column,negative in [('Up','Down'),('HighVol','LowVol'),('Post2021','Pre2021')]:
                for sign,label in [(True,column),(False,negative)]:
                    ids=market.loc[market[column].eq(sign),'window_id'];q=z[z.window_id_left.isin(ids)]
                    if q.empty:continue
                    a=metrics(q.net_return_left,q.rf_daily_left,q.position_left);b=metrics(q.net_return_right,q.rf_daily_right,q.position_right)
                    state_rows.append(dict(rec,state=label,n_days=len(q),n_windows=q.window_id_left.nunique(),**{'delta_'+k:a[k]-b[k] for k in a},interpretation='post-hoc diagnostic; not a trading signal'))
                for lag in [1,3,5]:
                    contrast=es.window_state_contrast(win.difference.to_numpy(),win[column].astype(int).to_numpy(),win.window_id.to_numpy(),lag)
                    contrasts.append(dict(rec,state_axis=column,bandwidth=lag,n_windows=len(win),**contrast,estimand='difference in mean 60-day paired return differential across states'))
    contrasts=pd.DataFrame(contrasts)
    if not contrasts.empty:
        contrasts['p_holm']=np.nan
        for lag,ix in contrasts.groupby('bandwidth').groups.items():contrasts.loc[ix,'p_holm']=es.holm(contrasts.loc[ix,'p'])
    # Preserve shape-neutral results as their own table; no interpolated k=1 claims.
    neutral_pairs=definitions[definitions.left_experiment.eq('K1_NEUTRAL')&definitions.purpose.eq('rvar_cvar')]
    heter=[]
    for rec in definitions[definitions.purpose.eq('heterogeneous_vs_control')].to_dict('records'):
        aa=assets[(assets.experiment_id==rec['left_experiment'])&assets.model_id.eq(rec['left_model'])]
        bb=assets[(assets.experiment_id==rec['right_experiment'])&assets.model_id.eq(rec['right_model'])]
        for wid,g in aa.groupby('window_id'):
            h=bb[bb.window_id.eq(wid)].set_index('stock_code');g=g.set_index('stock_code');union=g.index.union(h.index)
            row=dict(rec,window_id=wid,weight_l1=float(g.posttrade_weight.reindex(union,fill_value=0).subtract(h.posttrade_weight.reindex(union,fill_value=0)).abs().sum()),initial_weight_l1=float(g.initial_weight.reindex(union,fill_value=0).subtract(h.initial_weight.reindex(union,fill_value=0)).abs().sum()))
            for col in ['mu','rho','k_return']:
                common=g.index.intersection(h.index)
                if col in g and col in h:row['mean_abs_'+col+'_difference']=float((g.loc[common,col]-h.loc[common,col]).abs().mean())
            heter.append(row)
    return [('step4_structure_by_window.csv',structure),('step4_structure_summary.csv',summary),('step4_state_performance.csv',pd.DataFrame(state_rows)),('step4_state_contrasts.csv',contrasts),('step4_market_labels.csv',market),('step4_heterogeneity_mechanisms.csv',pd.DataFrame(heter))]


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--stage3',type=Path,default=BASE_DIR/'results3')
    p.add_argument('--out',type=Path,default=BASE_DIR/'results4');p.add_argument('--overwrite',action='store_true')
    p.add_argument('--replicates',type=int,default=2000);p.add_argument('--block-length',type=int,default=10)
    p.add_argument('--seed',type=int,default=42);p.add_argument('--allow-pilot',action='store_true');p.add_argument('--no-sensitivity',action='store_true',help='Pilot only')
    a=p.parse_args(argv);m=run(a.stage3,a.out,a.overwrite,a.replicates,a.block_length,a.seed,a.allow_pilot,not a.no_sensitivity);print(m['status'])
if __name__=='__main__':main()
