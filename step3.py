"""Model A sequential execution. Re-solves with actual holdings, then buys and holds.
No historical MV files or external execution_correction module are used.
"""
import argparse
from pathlib import Path
from datetime import datetime,timezone
import json
import time
import numpy as np
import pandas as pd
import step1
import step2
BASE_DIR=Path(__file__).resolve().parent


def checked_manifest(folder,name,status=None):
    folder=Path(folder); meta=json.loads((folder/name).read_text(encoding='utf-8'))
    if status and meta.get('status')!=status:raise ValueError(f'{name}: unexpected status {meta.get("status")}')
    for item in meta.get('outputs',[]):
        p=folder/Path(item['path']).name
        if not p.is_file() or step1.sha256_file(p)!=item['sha256']:
            raise ValueError(f'Output missing/hash mismatch: {p}')
    return meta


def prepare_output(folder,prefix,overwrite):
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    if not overwrite and list(folder.glob(prefix+'*')):
        raise FileExistsError(f'{folder}: existing {prefix} outputs; use another directory or --overwrite')
    return folder


def finish_metadata(folder,name,meta,outputs):
    meta['finished_utc']=datetime.now(timezone.utc).isoformat()
    meta['outputs']=[dict(path=str(Path(p).resolve()),sha256=step1.sha256_file(p)) for p in outputs]
    (Path(folder)/name).write_text(json.dumps(meta,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    return meta


def advance_holdings(weights,cash,returns,rf,fee=0.):
    """No intrahold rebalancing; fee is charged before first holding-day return."""
    weights=weights.astype(float).copy()
    if not np.isfinite(weights).all() or (weights<0).any() or not np.isfinite(cash) or not np.isfinite(fee) or fee<0:
        raise ValueError('Invalid holding amounts or fee')
    if returns.index.has_duplicates or not returns.index.is_monotonic_increasing:raise ValueError('Invalid holding calendar')
    if cash < -1e-7 or abs(weights.sum()+cash+fee-1)>1e-7:
        raise ValueError('Self-financing identity failed')
    values=weights.copy();cash_value=max(0.,float(cash));last_nav=1.;rows=[]
    for i,date in enumerate(returns.index):
        active=values[values>1e-14]
        rr=returns.loc[date].reindex(active.index)
        if not np.isfinite(rr).all() or (rr < -1).any():
            raise ValueError(f'Missing/invalid return for held assets on {date}: {rr[~np.isfinite(rr) | (rr < -1)].index.tolist()}; no zero imputation')
        rate=float(rf.loc[date])
        if not np.isfinite(rate) or rate<=-1:raise ValueError(f'Invalid cash return {date}')
        values.loc[active.index]=active*(1+rr)
        cash_value*=1+rate
        nav=float(values.sum()+cash_value)
        if nav<=0 or not np.isfinite(nav):raise ValueError('Invalid/ruined portfolio NAV')
        r=nav/last_nav-1
        rows.append(dict(date=date,net_return=r,rf_daily=rate,nav_relative=nav,
                         risky_exposure=float(values.sum()/nav),cash_weight=float(cash_value/nav),
                         trading_cost=fee if i==0 else 0.,valuation_valid=True))
        last_nav=nav
    return pd.DataFrame(rows),values/last_nav,cash_value/last_nav


def coefficients(context,spec,previous,returns,turnover,signs,policy):
    """Only estimation data and turnover at the decision close are used."""
    stocks=list(context['stocks']);date=context['win'].est_end
    held=previous[previous>1e-14]
    codes=stocks+sorted(set(held.index)-set(stocks));n=len(codes)
    old=previous.reindex(codes,fill_value=0.).to_numpy()
    liquid=turnover.loc[date].reindex(codes).to_numpy(float)
    tradable=np.isfinite(liquid)&(liquid>0)
    locks=np.where(tradable,np.nan,old)
    outside=~np.isin(codes,stocks)
    forced=outside&tradable
    endpoints=np.zeros((n,4));mu=np.zeros(n);rho=np.zeros(n)
    current=context['schemes'][spec['endpoint_scheme']]
    endpoints[:len(stocks)]=current
    for j in range(len(stocks),n):
        if forced[j]:continue
        sample=returns.loc[context['win'].est_start:date,codes[j]].dropna()
        if len(sample)<200:raise ValueError(f'Insufficient calibration for locked exiting holding {codes[j]}')
        endpoints[j]=np.quantile(sample,step1.ENDPOINT_SCHEMES[spec['endpoint_scheme']],method='linear')
    ki=step2.heterogeneous_shapes(stocks,spec,policy,signs)
    extra=[spec['k']*(1+spec.get('delta',0)*signs[code]) if spec.get('heterog_scope')=='all' else spec['k'] for code in codes[len(stocks):]]
    ki=np.r_[ki,extra]
    use=~forced
    mu[use],rho[use]=step2.portfolio_coefficients(endpoints[use],ki[use],spec['risk_measure'],spec['interval'],spec['gamma'])
    sigma=None
    if spec['risk_measure']=='MV':
        train=returns.loc[context['win'].est_start:date,np.array(codes)[use]]
        if train.isna().any().any():raise ValueError('MV covariance requires finite training for retained assets')
        sigma=np.zeros((n,n));sigma[np.ix_(use,use)]=train.cov().to_numpy()+np.eye(use.sum())*1e-8
    return codes,old,mu,rho,locks,forced,sigma,endpoints,ki


def equal_weight_execution(previous, stocks, turnover_at_close, exposure, policy):
    """Match parent's post-fee risky exposure, with own turnover and locked positions."""
    from scipy.optimize import brentq
    codes=list(stocks)+sorted(set(previous[previous>1e-14].index)-set(stocks))
    old=previous.reindex(codes,fill_value=0.).to_numpy(float)
    liquid=turnover_at_close.reindex(codes).to_numpy(float)
    tradable=np.isfinite(liquid)&(liquid>0)
    locked=np.where(tradable,0.,old)
    invest=tradable&np.isin(codes,stocks)
    def amount(q):
        x=locked.copy()
        if invest.any():x[invest]=max(0.,exposure*q-locked.sum())/invest.sum()
        return x
    def cost(x):return float(policy['cost_buy']*np.maximum(x-old,0).sum()+policy['cost_sell']*np.maximum(old-x,0).sum())
    lo=locked.sum()/exposure if exposure>0 else (0. if locked.sum()==0 else 2.)
    if lo>1 or (not invest.any() and abs(exposure-locked.sum())>1e-10):
        return pd.Series(old,index=codes),1-old.sum(),0.,False
    def f(q):return q+cost(amount(q))-1
    if f(lo)>1e-10 or f(1)<-1e-10:
        return pd.Series(old,index=codes),1-old.sum(),0.,False
    q=brentq(f,lo,1,xtol=1e-14) if lo<1 else 1.
    x=amount(q);fee=cost(x)
    return pd.Series(x,index=codes),1-x.sum()-fee,fee,True


def run_stage3(input_dir,stage2_dir,data_dir,output_dir,overwrite=False,window_limit=None):
    started=time.perf_counter();input_dir=Path(input_dir);stage2_dir=Path(stage2_dir)
    out=prepare_output(output_dir,'step3_',overwrite)
    source=checked_manifest(stage2_dir,'step2_run_metadata.json','complete_stage2_inputs')
    if source.get('pipeline_model')!='ModelA':raise ValueError('Rerun the supplied Model A step2.py first')
    if source.get('scope')!='full':raise ValueError('Stage2 pilot outputs cannot enter the sequential main run')
    if source.get('code_sha256')!=step1.sha256_file(step2.__file__):
        raise ValueError('Step2 source changed since optimization: rerun supplied step2.py')
    policy=step2.load_policy(stage2_dir/'lp_policy_lock.json')
    contexts,_,signs=step2.load_inputs(input_dir,policy)
    returns=step1.read_daily_matrix(input_dir/'full_return_matrix.csv',True)
    turnover=step1.read_daily_matrix(input_dir/'full_turnover_matrix.csv',True)
    rf=step1.read_daily_matrix(input_dir/'risk_free_rate.csv').iloc[:,0]
    selected=step2.select_experiments(step2.experiment_specs(policy),source['experiment_id'])
    # Endpoint bootstrap is a conditional coefficient diagnostic, never a live strategy.
    specs=[s for s in step2.experiment_specs(policy) if s['experiment_id'] in selected]
    if window_limit:
        if not 1<=window_limit<=policy['expected_windows']:raise ValueError('Invalid window limit')
        contexts=dict(list(contexts.items())[:window_limit])
    calendar=[]
    for wid,c in contexts.items():
        d=returns.loc[c['win'].test_start:c['win'].test_end].index
        if len(d)!=60:raise ValueError(f'Window {wid}: expected 60 holding days')
        calendar.extend(d)
    calendar=pd.DatetimeIndex(calendar)
    if not calendar.is_monotonic_increasing or calendar.has_duplicates:raise ValueError('Overlapping/unsorted holding windows')
    full=returns.loc[calendar[0]:calendar[-1]].index
    if not full.equals(calendar):raise ValueError('Missing holding windows; cannot splice wealth')
    meta=dict(stage=3,pipeline_model='ModelA',pipeline_revision='review_completion_v1',status='running',scope='pilot' if window_limit else 'full',
              started_utc=datetime.now(timezone.utc).isoformat(),stage2_run_id=source['run_id'],
              source_stage2_metadata_sha256=step1.sha256_file(stage2_dir/'step2_run_metadata.json'),
              code_sha256=step1.sha256_file(__file__),step2_code_sha256=step1.sha256_file(step2.__file__),
              policy=policy,execution_rule='trade at estimation-end close; buy-and-hold; cash pays realized daily rate',
              failure_rule='infeasible: no trades, hold actual positions/cash; numerical/data error: abort',
              budget_rule='asset amounts and fees relative to pretrade NAV; cash=1-sum(x)-fee',
              cost_return_rule='original conservative net daily criterion: deduct entire rebalance fee, not fee/60',
              bootstrap_scope='Stage2 conditional coefficient samples only; no independent sequential bootstrap strategy')
    static_states=pd.read_csv(stage2_dir/'step2_windows.csv').set_index(['experiment_id','model_id','window_id']).solver_status
    days=[];windows=[];asset_rows=[];attempts=[];cross=[]
    try:
        for spec in specs:
            previous=pd.Series(dtype=float);cash=1.;wealth=1.
            ew_previous=pd.Series(dtype=float);ew_wealth=1.
            for wid,context in contexts.items():
                codes,x0,mu,rho,locks,forced,sigma,ep,ki=coefficients(context,spec,previous,returns,turnover,signs,policy)
                x,info=step2.solve_execution_modelA(mu,rho,context['rf_daily'],spec['target_annual'],x0,spec['exposure_floor'],policy,
                                                   locks,forced,sigma,spec['scales'])
                identity=dict(experiment_id=spec['experiment_id'],model_id=spec['model_id'],window_id=int(wid))
                for a in info['attempts']:attempts.append(dict(identity,**a))
                optimized=x is not None
                if not optimized:
                    x=x0.copy(); fee=0.;cash_start=1-float(x.sum()); action='hold_after_infeasible'
                else:
                    fee=info['fee'];cash_start=info['cash'];action='optimized_rebalance'
                target=pd.Series(x,index=codes)
                dates=returns.loc[context['win'].test_start:context['win'].test_end].index
                daily,previous,cash=advance_holdings(target,cash_start,returns.loc[dates],rf,fee)
                # Retain full-date strategy returns, including explicitly labelled fallback periods.
                for key,value in identity.items():daily[key]=value
                daily['solver_status']=info['status'];daily['execution_action']=action;daily['optimized']=optimized
                daily['direct_eligible']=info['status']=='Direct'
                daily['nav']=wealth*daily.nav_relative
                wealth=float(daily.nav.iloc[-1]);days.append(daily)
                hhi=float((x/x.sum())@(x/x.sum())) if x.sum()>0 else np.nan
                windows.append(dict(identity,risk_measure=spec['risk_measure'],k=spec['k'],solver_status=info['status'],
                    execution_action=action,optimized=optimized,target_annual=spec['target_annual'],effective_target=info['target_annual_effective'],
                    relaxation_scale=info['scale'],exposure_floor=spec['exposure_floor'],risky_exposure=float(x.sum()),cash_weight=cash_start,
                    trading_cost=fee,turnover=.5*float(abs(x-x0).sum()),initial_weight_sum=float(x0.sum()),
                    forced_exit_sales=float(x0[forced].sum()) if optimized else 0.,locked_assets=int(np.isfinite(locks).sum()),
                    expected_net_daily=info['expected_net_daily'],return_constraint_slack=info['return_slack'],
                    max_constraint_violation=info['residual'],hhi_risky=hhi,effective_n=1/hhi if hhi>0 else np.nan,
                    window_total_return=float(daily.nav_relative.iloc[-1]-1),hold_start=dates[0],hold_end=dates[-1],
                    static_solver_status=static_states.loc[(spec['experiment_id'],spec['model_id'],wid)]))
                for j,code in enumerate(codes):
                    asset_rows.append(dict(identity,stock_code=code,initial_weight=x0[j],posttrade_amount=x[j],
                        posttrade_weight=x[j]/(1-fee),terminal_weight=float(previous.get(code,0)),
                        mu=float(mu[j]),rho=float(rho[j]),k_return=float(ki[j]),a=float(ep[j,0]),b=float(ep[j,1]),c=float(ep[j,2]),d=float(ep[j,3]),
                        buy=max(0.,x[j]-x0[j]),sell=max(0.,x0[j]-x[j]),locked=bool(np.isfinite(locks[j]))))
                if spec['risk_measure']!='MV':
                    _,risks=step2.three_measure_coefficients(ep,ki)
                    record=dict(identity,k=spec['k'],solver_status=info['status'],optimized=optimized)
                    for label,v in risks.items():record['risk_'+label]=float(v@x-context['rf_daily']*cash_start+fee)
                    record['identity_residual']=record['risk_CVaR20']-.25*record['risk_CVaR05']-.75*record['risk_RVaR']
                    if abs(record['identity_residual'])>1e-10:raise ValueError('Cross-risk identity failed')
                    cross.append(record)
                if spec['risk_measure']=='RVaR':
                    ew_id=dict(identity,model_id=spec['model_id']+'_EMEW')
                    desired=float(x.sum()/(1-fee))
                    ew_old=ew_previous.copy()
                    ew_x,ew_cash,ew_fee,matched=equal_weight_execution(ew_previous,context['stocks'],turnover.loc[context['win'].est_end],desired,policy)
                    ew_day,ew_previous,unused=advance_holdings(ew_x,ew_cash,returns.loc[dates],rf,ew_fee)
                    for key,value in ew_id.items():ew_day[key]=value
                    ew_day['solver_status']='Benchmark';ew_day['execution_action']='exposure_matched_equal_weight' if matched else 'hold_unmatched'
                    ew_day['optimized']=False;ew_day['direct_eligible']=info['status']=='Direct' and matched
                    ew_diff=ew_x.subtract(ew_old,fill_value=0.)
                    for code in ew_x.index:
                        asset_rows.append(dict(ew_id,stock_code=code,initial_weight=float(ew_old.get(code,0)),posttrade_amount=float(ew_x[code]),
                            posttrade_weight=float(ew_x[code]/(1-ew_fee)),terminal_weight=float(ew_previous.get(code,0)),
                            buy=float(max(0,ew_diff[code])),sell=float(max(0,-ew_diff[code]))))
                    ew_day['nav']=ew_wealth*ew_day.nav_relative;ew_wealth=float(ew_day.nav.iloc[-1]);days.append(ew_day)
                    windows.append(dict(ew_id,risk_measure='EMEW',k=spec['k'],solver_status='Benchmark',execution_action=ew_day.execution_action.iloc[0],
                        optimized=False,risky_exposure=float(ew_x.sum()),cash_weight=ew_cash,trading_cost=ew_fee,
                        turnover=.5*float(ew_diff.abs().sum()),
                        window_total_return=float(ew_day.nav_relative.iloc[-1]-1),hold_start=dates[0],hold_end=dates[-1],
                        exposure_match_gap=float(ew_x.sum()/(1-ew_fee)-desired),exposure_matched=matched))
            print(f"{spec['experiment_id']} {spec['model_id']}: {pd.Series([w['solver_status'] for w in windows if w['model_id']==spec['model_id'] and w['experiment_id']==spec['experiment_id']]).value_counts().to_dict()}",flush=True)
        d=pd.concat(days,ignore_index=True);w=pd.DataFrame(windows)
        # An optional observed index is a benchmark, never substituted for missing strategy returns.
        index_path=Path(data_dir)/'sh50_index_return.csv'
        if index_path.exists():
            idx=step1.read_daily_matrix(index_path)['return'].reindex(calendar)
            if not np.isfinite(idx).all() or (idx<=-1).any():raise ValueError('Invalid index returns')
            bench=pd.DataFrame(dict(date=calendar,net_return=idx.to_numpy(),rf_daily=rf.reindex(calendar).to_numpy(),
                nav=np.cumprod(1+idx.to_numpy()),model_id='SH50',experiment_id='BENCHMARK',optimized=True,
                execution_action='observed_index',solver_status='Direct',direct_eligible=True))
            lookup={date:wid for wid,c in contexts.items() for date in returns.loc[c['win'].test_start:c['win'].test_end].index}
            bench['window_id']=bench.date.map(lookup);d=pd.concat([d,bench],ignore_index=True)
        checks=[dict(check='finite_full_calendar_returns',passed=bool(np.isfinite(d.net_return).all())),
                dict(check='one_row_per_strategy_date',passed=not d.duplicated(['experiment_id','model_id','date']).any()),
                dict(check='all_strategies_same_calendar',passed=bool(d.groupby(['experiment_id','model_id']).size().eq(len(calendar)).all())),
                dict(check='accepted_constraints',passed=bool(w.loc[w.optimized,'max_constraint_violation'].le(policy['constraint_tolerance']).all())),
                dict(check='failed_windows_are_hold_actions',passed=bool(w.loc[w.solver_status.eq('Failed'),'execution_action'].eq('hold_after_infeasible').all())),
                dict(check='cash_assets_fees_sum_one',passed=bool(np.allclose(w.risky_exposure+w.cash_weight+w.trading_cost,1,rtol=0,atol=1e-7)))]
        if not all(c['passed'] for c in checks):raise ValueError('Stage3 acceptance failure')
        allocations=pd.DataFrame(asset_rows); weight_pairs=[]
        for k,suffix,c05,rv,c20 in step2.SUBSTITUTION_TRIOS:
            for wid in contexts:
                for left,right in [(rv,c05),(rv,c20),(c05,c20)]:
                    a=allocations[(allocations.model_id==left)&allocations.window_id.eq(wid)].set_index('stock_code')
                    b=allocations[(allocations.model_id==right)&allocations.window_id.eq(wid)].set_index('stock_code')
                    if a.empty or b.empty:continue
                    distance=a.posttrade_weight.subtract(b.posttrade_weight,fill_value=0.)
                    initial=a.initial_weight.subtract(b.initial_weight,fill_value=0.)
                    states=w[w.window_id.eq(wid)&w.model_id.isin([left,right])]
                    weight_pairs.append(dict(window_id=wid,k=k,left_model=left,right_model=right,
                        weight_l1=float(distance.abs().sum()),initial_weight_l1=float(initial.abs().sum()),
                        both_optimized=bool(states.optimized.all()),
                        diagnostic_scope='sequential policies with own actual initial holdings, including labelled holds'))
        outputs=[]
        for name,frame in [('step3_daily_returns.parquet',d),('step3_window_backtests.csv',w),('step3_assets.parquet',allocations),('step3_weight_pairs.csv',pd.DataFrame(weight_pairs)),
                           ('step3_attempts.csv',pd.DataFrame(attempts)),('step3_cross_risks.csv',pd.DataFrame(cross)),('step3_acceptance_checks.csv',pd.DataFrame(checks))]:
            path=out/name
            if path.suffix=='.parquet':frame.to_parquet(path,index=False)
            else:frame.to_csv(path,index=False,encoding='utf-8-sig')
            outputs.append(path)
        meta.update(status='accepted_modelA_stage3',failed_optimization_windows=int(w.solver_status.eq('Failed').sum()),
                    relaxed_windows=int(w.solver_status.eq('Relaxed').sum()),strategies=len(specs),
                    fallback_returns_included=True,index_benchmark_included=index_path.exists(),elapsed_seconds=time.perf_counter()-started)
        return finish_metadata(out,'step3_run_metadata.json',meta,outputs)
    except Exception as exc:
        meta.update(status='error',error=f'{type(exc).__name__}: {exc}')
        finish_metadata(out,'step3_run_metadata.json',meta,[])
        raise


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',type=Path,default=BASE_DIR/'results1');p.add_argument('--stage2',type=Path,default=BASE_DIR/'results2')
    p.add_argument('--data',type=Path,default=BASE_DIR/'data');p.add_argument('--out',type=Path,default=BASE_DIR/'results3')
    p.add_argument('--overwrite',action='store_true');p.add_argument('--window-limit',type=int)
    a=p.parse_args(argv);m=run_stage3(a.input,a.stage2,a.data,a.out,a.overwrite,a.window_limit)
    print(m['status']);return 0
if __name__=='__main__':raise SystemExit(main())
