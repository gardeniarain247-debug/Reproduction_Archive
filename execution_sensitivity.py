"""Separate execution-policy sensitivities; never overwrites the original run.

next_close: keep the original 250 formation observations and 60 valuation days.
Old holdings earn the first holding-day return; rebalance at that day's close,
then new holdings earn only the following 59 days. Signals and order eligibility
use formation-end information. Execution-day turnover can only reject a planned
trade, never select/reoptimize it. Close-value execution remains an idealization.
"""
from pathlib import Path
import argparse
import json
import time
import numpy as np
import pandas as pd
from scipy.optimize import linprog
import step1,step2,step3
import review_completion as rc


def repair_portfolio(previous,floor,policy,locks,forced):
    """Minimize traded amount under ORIGINAL pretrade-NAV bounds; drop only target.

    This is a separate fallback policy, not an optimal solution of the original
    return-constrained Model A. If locks prevent repair, return None and disclose.
    """
    previous=np.asarray(previous,float);n=len(previous)
    fee=np.r_[np.zeros(n),np.full(n,policy['cost_buy']),np.full(n,policy['cost_sell'])]
    expo=np.r_[np.ones(n),np.zeros(2*n)]
    A=np.vstack([expo+fee,-expo]);rhs=np.array([1-policy['cash_lower'],-floor])
    E=np.c_[np.eye(n),-np.eye(n),np.eye(n)]
    bounds=[(0,policy['weight_upper']) for _ in range(n)]+[(0,None)]*(2*n)
    for j in range(n):
        if np.isfinite(locks[j]):
            bounds[j]=(previous[j],previous[j]);bounds[n+j]=bounds[2*n+j]=(0,0)
        elif forced[j]:bounds[j]=(0,0)
    result=linprog(np.r_[np.zeros(n),np.ones(2*n)],A_ub=A,b_ub=rhs,A_eq=E,
        b_eq=previous,bounds=bounds,method=policy['solver_method'],options=policy['solver_options'])
    if result.status==2:return None,dict(status='repair_infeasible')
    if not result.success:raise RuntimeError('Repair numerical failure: '+result.message)
    x=np.maximum(result.x[:n],0.)
    actual=np.r_[x,np.maximum(x-previous,0),np.maximum(previous-x,0)]
    residual=max(0.,float(np.max(A@actual-rhs)),float(np.max(abs(E@actual-previous))))
    for j,(lo,hi) in enumerate(bounds):
        residual=max(residual,lo-actual[j],actual[j]-hi if hi is not None else 0.)
    if residual>policy['constraint_tolerance']:raise RuntimeError('Repair residual audit failed')
    cost=float(fee@actual)
    return x,dict(status='repaired_without_return_target',fee=cost,cash=1-x.sum()-cost,residual=residual)


def before_trade(previous,dates,returns,rf,timing):
    if timing=='formation_close':return previous.copy(),None
    if timing!='next_close':raise ValueError('Unknown timing')
    first,end,cash=step3.advance_holdings(previous,1-previous.sum(),returns.loc[dates[:1]],rf)
    return end,first.iloc[0].to_dict()


def after_trade(x,cash,fee,dates,returns,rf,prefix):
    if prefix is None:return step3.advance_holdings(x,cash,returns.loc[dates],rf,fee)
    q=1-fee
    if q<=0 or abs(x.sum()+cash+fee-1)>1e-7:raise ValueError('Invalid delayed budget')
    first=dict(prefix)
    first.update(net_return=prefix['nav_relative']*q-1,
        nav_relative=prefix['nav_relative']*q,risky_exposure=float(x.sum()/q),
        cash_weight=float(cash/q),trading_cost=fee,valuation_valid=True)
    if len(dates)==1:return pd.DataFrame([first]),x/q,cash/q
    rest,end,cash_end=step3.advance_holdings(x/q,cash/q,returns.loc[dates[1:]],rf)
    rest['nav_relative']*=first['nav_relative']
    return pd.concat([pd.DataFrame([first]),rest],ignore_index=True),end,cash_end


def reconcile_execution_fills(old, planned, turnover_at_execution, policy, tolerance=1e-12):
    """Apply formation-time orders at next close without reoptimizing on future data.

    An asset with unavailable/nonpositive turnover cannot trade: its pretrade
    amount is carried forward. Execute eligible sells, then scale eligible buys
    proportionally if necessary to respect the original cash lower bound.
    All amounts and fees are in execution pretrade NAV units. This is a
    conservative all-or-nothing *eligibility proxy*, not actual order fills.
    """
    index=old.index.union(planned.index, sort=False)
    previous=old.reindex(index,fill_value=0.).astype(float)
    goal=planned.reindex(index,fill_value=0.).astype(float)
    vol=turnover_at_execution.reindex(index)
    changed=(goal-previous).abs().gt(tolerance)
    blocked=changed & (~np.isfinite(vol) | vol.le(0))
    result=goal.copy()
    result.loc[blocked]=previous.loc[blocked]
    # Sales are accepted first; eligible buys are scaled *without* any
    # look-ahead reoptimization. Positive cash lower bound is respected.
    delta=result-previous
    sell=(-delta.clip(upper=0)).sum()
    buys=delta.clip(lower=0)
    buy_total=float(buys.sum())
    cost_buy=float(policy['cost_buy']);cost_sell=float(policy['cost_sell'])
    cash_min=float(policy['cash_lower'])
    base_cash=1-float(previous.sum())
    available=base_cash+(1-cost_sell)*float(sell)-cash_min
    if available < -1e-9:
        # Existing locked positions can already violate the cash constraint.
        # Never invent sales; disclose the actual resulting breach.
        scale=0.0
    else:
        scale=min(1.0,max(0.0,available/((1+cost_buy)*buy_total))) if buy_total>tolerance else 1.0
    if scale<1:result=result-buys*(1-scale)
    delta=result-previous
    fee=cost_buy*float(delta.clip(lower=0).sum())+cost_sell*float((-delta).clip(lower=0).sum())
    cash=1-float(result.sum())-fee
    if cash < -1e-8:raise ValueError('Reconciled execution has negative cash')
    ledger=[]
    for code in index:
        if not changed.loc[code]:continue
        order=float(goal.loc[code]-previous.loc[code]);filled=float(result.loc[code]-previous.loc[code])
        status=('rejected_no_proxy_volume' if blocked.loc[code] else
                ('partial_cash_capacity' if abs(order-filled)>tolerance else 'proxy_eligible'))
        ledger.append(dict(stock_code=code,planned_change=order,executed_change=filled,
                           unfilled_change=order-filled,turnover_proxy=float(vol.loc[code]) if pd.notna(vol.loc[code]) else np.nan,
                           fill_status=status))
    return result,cash,fee,ledger


def run(input_dir,stage2_dir,data_dir,out,timing='next_close',fallback='hold',window_limit=None):
    if fallback not in ['hold','repair']:raise ValueError('Unknown fallback')
    input_dir,stage2_dir,data_dir,out=map(Path,[input_dir,stage2_dir,data_dir,out])
    if out.exists() and any(out.iterdir()):raise FileExistsError('Use a fresh execution output directory')
    # Portable byte checks for the complete supplied Stage2 folder.
    source=json.loads((stage2_dir/'step2_run_metadata.json').read_text())
    if source.get('pipeline_model')!='ModelA' or source.get('status')!='complete_stage2_inputs':
        raise ValueError('Complete Model A Stage2 required')
    for item in source['outputs']:
        f=stage2_dir/item['path'].replace('\\','/').rsplit('/',1)[-1]
        if not f.is_file() or step1.sha256_file(f)!=item['sha256']:raise ValueError(f'Stage2 missing/hash mismatch: {f}')
    if source['code_sha256']!=step1.sha256_file(step2.__file__):raise ValueError('Use matching Step2 source')
    policy=step2.load_policy(stage2_dir/'lp_policy_lock.json')
    contexts,_,signs=step2.load_inputs(input_dir,policy)
    if window_limit:contexts=dict(list(contexts.items())[:window_limit])
    returns=step1.read_daily_matrix(input_dir/'full_return_matrix.csv',True)
    turnover=step1.read_daily_matrix(input_dir/'full_turnover_matrix.csv',True)
    rf=step1.read_daily_matrix(input_dir/'risk_free_rate.csv').iloc[:,0]
    selected=step2.select_experiments(step2.experiment_specs(policy),source['experiment_id'])
    specs=[s for s in step2.experiment_specs(policy) if s['experiment_id'] in selected]
    days=[];windows=[];assets=[];attempts=[];cross=[];trade_states=[];fill_ledger=[];started=time.perf_counter()
    calendar=pd.DatetimeIndex([date for c in contexts.values() for date in returns.loc[c['win'].test_start:c['win'].test_end].index])
    if calendar.has_duplicates or not calendar.is_monotonic_increasing:raise ValueError('Invalid calendar')
    if not returns.loc[calendar[0]:calendar[-1]].index.equals(calendar):raise ValueError('Holding calendar gaps')
    for spec in specs:
        previous=pd.Series(dtype=float);ew_previous=pd.Series(dtype=float);wealth=ew_wealth=1.
        for wid,c in contexts.items():
            dates=returns.loc[c['win'].test_start:c['win'].test_end].index
            if len(dates)!=60:raise ValueError('Need 60 holding days')
            original_previous=previous.copy()
            previous,prefix=before_trade(previous,dates,returns,rf,timing)
            codes,x0,mu,rho,locks,forced,sigma,ep,ki=step3.coefficients(c,spec,previous,returns,turnover,signs,policy)
            x,info=step2.solve_execution_modelA(mu,rho,c['rf_daily'],spec['target_annual'],x0,
                spec['exposure_floor'],policy,locks,forced,sigma,spec['scales'])
            identity=dict(experiment_id=spec['experiment_id'],model_id=spec['model_id'],window_id=int(wid))
            attempts.extend(dict(identity,**v) for v in info['attempts'])
            optimized=x is not None;repaired=False;repair_status='not_requested'
            if optimized:fee,cash,action=info['fee'],info['cash'],'optimized_rebalance'
            else:
                if fallback=='repair':
                    x,repair=repair_portfolio(x0,spec['exposure_floor'],policy,locks,forced)
                    repair_status=repair['status'];repaired=x is not None
                if repaired:fee,cash,action=repair['fee'],repair['cash'],'minimum_turnover_repair'
                else:x=x0.copy();fee=0.;cash=1-x.sum();action='hold_after_infeasible'
            target=pd.Series(x,index=codes,dtype=float)
            execution_pretrade=previous.copy()
            if timing=='next_close':
                target,cash,fee,records=reconcile_execution_fills(previous,target,turnover.loc[dates[0]],policy)
                fill_ledger.extend(dict(identity,execution_date=dates[0],benchmark=False,**r) for r in records)
                action += '_execution_checked'
            # All execution accounting is keyed by stock code. A cancelled sell
            # can leave a stock outside the current calibration universe held.
            target=target.astype(float)
            aligned_before=execution_pretrade.reindex(target.index.union(execution_pretrade.index),fill_value=0.)
            aligned_after=target.reindex(aligned_before.index,fill_value=0.)
            actual_change=aligned_after-aligned_before
            actual_turnover=float(.5*actual_change.abs().sum())
            x_model=target.reindex(codes,fill_value=0.).to_numpy(float)
            off_model=target.drop(labels=codes,errors='ignore')
            off_model_positive=bool((off_model>1e-12).any())
            actual_exposure=float(aligned_after.sum())
            if abs(actual_exposure+cash+fee-1)>1e-7:
                raise ValueError('Post-fill budget identity failed')
            # No original-optimum claim survives an execution-day rejected order.
            execution_changed=timing=='next_close' and any(r['fill_status']!='proxy_eligible' for r in (records if timing=='next_close' else []))
            daily,previous,cash_end=after_trade(target,cash,fee,dates,returns,rf,prefix)
            for key,value in identity.items():daily[key]=value
            daily['solver_status']=info['status'];daily['execution_action']=action
            daily['execution_fully_filled_proxy']=not execution_changed
            daily['optimized']=optimized;daily['direct_eligible']=info['status']=='Direct'
            daily['nav']=wealth*daily.nav_relative;wealth=float(daily.nav.iloc[-1]);days.append(daily)
            actual_weights=target.to_numpy(float)
            hhi=float((actual_weights/actual_exposure)@(actual_weights/actual_exposure)) if actual_exposure>0 else np.nan
            windows.append(dict(identity,risk_measure=spec['risk_measure'],k=spec['k'],solver_status=info['status'],
                execution_action=action,optimized=optimized,repaired=repaired,repair_status=repair_status,
                execution_fully_filled_proxy=not execution_changed,
                target_annual=spec['target_annual'],effective_target=info['target_annual_effective'],
                relaxation_scale=info['scale'],exposure_floor=spec['exposure_floor'],risky_exposure=actual_exposure,
                cash_weight=cash,trading_cost=fee,turnover=actual_turnover,initial_weight_sum=float(aligned_before.sum()),
                off_model_held_count=int((off_model>1e-12).sum()),
                post_execution_exposure_floor_gap=max(0.,float(spec['exposure_floor'])-actual_exposure),
                post_execution_weight_cap_gap=max(0.,float(target.max())-float(policy['weight_upper'])) if len(target) else 0.,
                forced_exit_sales=float(x0[forced].sum()) if optimized or repaired else 0.,locked_assets=int(np.isfinite(locks).sum()),
                expected_net_daily=info['expected_net_daily'],return_constraint_slack=info['return_slack'],
                max_constraint_violation=info['residual'],hhi_risky=hhi,effective_n=1/hhi if hhi>0 else np.nan,
                window_total_return=float(daily.nav_relative.iloc[-1]-1),hold_start=dates[0],hold_end=dates[-1],
                formation_date=c['win'].est_end,execution_date=dates[0] if timing=='next_close' else c['win'].est_end,
                trading_cost_denominator='execution pretrade NAV',execution_timing=timing))
            code_index={code:j for j,code in enumerate(codes)}
            for code in aligned_after.index:
                j=code_index.get(code)
                model_known=j is not None
                initial=float(aligned_before.loc[code]);post=float(aligned_after.loc[code])
                assets.append(dict(identity,stock_code=code,initial_weight=initial,
                    previous_window_terminal_weight=float(original_previous.get(code,0.)),
                    posttrade_amount=post,posttrade_weight=post/(1-fee),
                    terminal_weight=float(previous.get(code,0.)),
                    mu=float(mu[j]) if model_known else np.nan,
                    rho=float(rho[j]) if model_known else np.nan,
                    k_return=float(ki[j]) if model_known else np.nan,
                    a=float(ep[j,0]) if model_known else np.nan,
                    b=float(ep[j,1]) if model_known else np.nan,
                    c=float(ep[j,2]) if model_known else np.nan,
                    d=float(ep[j,3]) if model_known else np.nan,
                    buy=max(0.,post-initial),sell=max(0.,initial-post),
                    locked=bool(np.isfinite(locks[j])) if model_known else True,
                    outside_model_universe=not model_known))
            trade_states.append(dict(identity,formation_date=c['win'].est_end,execution_date=windows[-1]['execution_date'],
                signal_precedes_execution=timing=='next_close',repair_status=repair_status,
                interpretation='Prior signals with idealized close-value rebalance; turnover does not certify fills'))
            if spec['risk_measure']!='MV':
                _,risks=step2.three_measure_coefficients(ep,ki)
                row=dict(identity,k=spec['k'],solver_status=info['status'],optimized=optimized,
                         risk_comparable=not off_model_positive,
                         off_model_held_count=int((off_model>1e-12).sum()))
                row.update({'risk_'+label:(float(v@x_model-c['rf_daily']*cash+fee)
                                        if not off_model_positive else np.nan) for label,v in risks.items()})
                row['identity_residual']=(row['risk_CVaR20']-.25*row['risk_CVaR05']-.75*row['risk_RVaR']
                                         if not off_model_positive else np.nan)
                cross.append(row)
            if spec['risk_measure']=='RVaR':
                ew_previous,ew_prefix=before_trade(ew_previous,dates,returns,rf,timing)
                old=ew_previous.copy();desired=actual_exposure/(1-fee)
                ei=dict(identity,model_id=spec['model_id']+'_EMEW')
                ex,ec,ef,matched=step3.equal_weight_execution(old,c['stocks'],turnover.loc[c['win'].est_end],desired,policy)
                if timing=='next_close':
                    ex,ec,ef,ew_records=reconcile_execution_fills(old,ex,turnover.loc[dates[0]],policy)
                    fill_ledger.extend(dict(ei,execution_date=dates[0],benchmark=True,**r) for r in ew_records)
                    matched=bool(matched and abs(float(ex.sum()/(1-ef))-desired)<=1e-8)
                ed,ew_previous,_=after_trade(ex,ec,ef,dates,returns,rf,ew_prefix)
                for key,value in ei.items():ed[key]=value
                ed['solver_status']='Benchmark';ed['execution_action']='exposure_matched_equal_weight' if matched else 'hold_unmatched'
                ed['optimized']=False;ed['direct_eligible']=info['status']=='Direct' and matched
                ed['nav']=ew_wealth*ed.nav_relative;ew_wealth=float(ed.nav.iloc[-1]);days.append(ed)
                change=ex.subtract(old,fill_value=0.)
                windows.append(dict(ei,risk_measure='EMEW',k=spec['k'],solver_status='Benchmark',execution_action=ed.execution_action.iloc[0],
                    optimized=False,risky_exposure=ex.sum(),cash_weight=ec,trading_cost=ef,turnover=.5*change.abs().sum(),
                    window_total_return=float(ed.nav_relative.iloc[-1]-1),hold_start=dates[0],hold_end=dates[-1],
                    exposure_match_gap=float(ex.sum()/(1-ef)-desired),exposure_matched=matched))
                for code in ex.index:
                    assets.append(dict(ei,stock_code=code,initial_weight=float(old.get(code,0)),posttrade_amount=ex[code],
                        posttrade_weight=ex[code]/(1-ef),terminal_weight=float(ew_previous.get(code,0)),
                        buy=max(0.,change[code]),sell=max(0.,-change[code])))
        print(f'Execution {timing}/{fallback}: {spec["model_id"]}',flush=True)
    d=pd.concat(days,ignore_index=True);w=pd.DataFrame(windows)
    index=step1.read_daily_matrix(data_dir/'sh50_index_return.csv')['return'].reindex(calendar)
    if not np.isfinite(index).all():raise ValueError('Missing index data')
    lookup={date:wid for wid,c in contexts.items() for date in returns.loc[c['win'].test_start:c['win'].test_end].index}
    bench=pd.DataFrame(dict(date=calendar,net_return=index.to_numpy(),rf_daily=rf.reindex(calendar).to_numpy(),
        nav=np.cumprod(1+index.to_numpy()),model_id='SH50',experiment_id='BENCHMARK',optimized=True,
        execution_action='observed_index',solver_status='Direct',direct_eligible=True,window_id=[lookup[t] for t in calendar]))
    d=pd.concat([d,bench],ignore_index=True)
    checks=pd.DataFrame([dict(check='finite_returns',passed=bool(np.isfinite(d.net_return).all())),
        dict(check='same_complete_calendar',passed=bool(d.groupby(['experiment_id','model_id']).size().eq(len(calendar)).all())),
        dict(check='budget',passed=bool(np.allclose(w.risky_exposure+w.cash_weight+w.trading_cost,1,atol=1e-7,rtol=0))),
        dict(check='accepted_original_constraints',passed=bool(w.loc[w.optimized,'max_constraint_violation'].le(policy['constraint_tolerance']).all()))])
    if not checks.passed.all():raise ValueError('Execution sensitivity audit failed')
    out.mkdir(parents=True,exist_ok=True);files=[]
    for name,frame in [('step3_daily_returns.parquet',d),('step3_assets.parquet',pd.DataFrame(assets)),
        ('step3_window_backtests.csv',w),('step3_attempts.csv',pd.DataFrame(attempts)),
        ('step3_cross_risks.csv',pd.DataFrame(cross)),('step3_acceptance_checks.csv',checks),
        ('execution_timing_audit.csv',pd.DataFrame(trade_states)),
        ('execution_fill_ledger.csv',pd.DataFrame(fill_ledger,columns=['experiment_id','model_id','window_id','execution_date','benchmark','stock_code','planned_change','executed_change','unfilled_change','turnover_proxy','fill_status']))]:
        path=out/name
        if path.suffix=='.parquet':frame.to_parquet(path,index=False)
        else:frame.to_csv(path,index=False,encoding='utf-8-sig')
        files.append(path)
    return step3.finish_metadata(out,'step3_run_metadata.json',dict(stage=3,status='accepted_modelA_stage3',
        pipeline_model='ModelA',pipeline_revision='review_completion_v1',scope='pilot' if window_limit else 'full',
        policy=policy,execution_variant=timing,fallback_policy=fallback,
        source_stage2_metadata_sha256=step1.sha256_file(stage2_dir/'step2_run_metadata.json'),
        code_sha256=step1.sha256_file(__file__),failed_target_windows=int(w.solver_status.eq('Failed').sum()),
        repaired_windows=int(w.execution_action.isin(['minimum_turnover_repair','minimum_turnover_repair_execution_checked']).sum()),elapsed_seconds=time.perf_counter()-started,
        caveats=['Close-value rebalancing idealization; not actual share/order execution',
                 'Repair preserves original pretrade-NAV bounds, not continuous-time/postfee hard limits',
                 'Execution-day turnover rejects ineligible orders, eligible buys cash-scaled; no reoptimization using that observation',
                 'Conditional bootstrap module must not value delayed trades as formation-close trades']),files)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['input','stage2','data','out']:p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--timing',choices=['formation_close','next_close'],default='next_close')
    p.add_argument('--fallback',choices=['hold','repair'],default='hold');p.add_argument('--window-limit',type=int)
    a=p.parse_args();print(run(a.input,a.stage2,a.data,a.out,a.timing,a.fallback,a.window_limit)['status'])


if __name__=='__main__':main()
