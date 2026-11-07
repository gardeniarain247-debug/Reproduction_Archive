"""Read-only completion of Model A inference, execution audits and reviewer evidence.

Accepts a normal run root or the flattened examine archive. Never changes source
files. A computed diagnostic is not a certificate of real-world execution.
"""
from pathlib import Path
import argparse
import hashlib
import json
import numpy as np
import pandas as pd
import empirical_statistics as es
import step4

P_COLUMNS = ['hac_p', 'annual_return_bootstrap_p', 'sharpe_bootstrap_p',
             'ES95_bootstrap_p', 'ES99_bootstrap_p', 'threshold_loss_bootstrap_p',
             'threshold_probability_bootstrap_p', 'threshold_loss_hac_p']


def locate(root, name, required=True):
    root = Path(root)
    choices = [root/name, *[root/x/name for x in
               ['results2', 'results3', 'results4', 'results_remaining',
                'results_remaining/shape_calibration']]]
    found = [p for p in choices if p.is_file()]
    if len(found) > 1:
        raise ValueError(f'Ambiguous source: {name}')
    if found:
        return found[0]
    if required:
        raise FileNotFoundError(f'{root}: missing {name}')
    return None


def verify_sources(root):
    """Check all supplied manifest-listed bytes, and disclose omitted artifacts."""
    records = []
    for name in ['step2_run_metadata.json', 'step3_run_metadata.json',
                 'step4_run_metadata.json', 'remaining_run_metadata.json']:
        path = locate(root, name, required=name.startswith('step3'))
        if path is None:
            records.append(dict(manifest=name, file=name, status='not_supplied'))
            continue
        meta = json.loads(path.read_text(encoding='utf-8'))
        if meta.get('pipeline_model', 'ModelA') != 'ModelA':
            raise ValueError('Model A source required')
        if name == 'step3_run_metadata.json' and meta.get('status') != 'accepted_modelA_stage3':
            raise ValueError('Accepted Step3 source required')
        for item in meta.get('outputs', []):
            basename = item['path'].replace('\\', '/').rsplit('/', 1)[-1]
            f = locate(root, basename, False)
            status = 'not_supplied'
            if f is not None:
                status = 'matched' if hashlib.sha256(f.read_bytes()).hexdigest() == item['sha256'] else 'hash_mismatch'
                if status == 'hash_mismatch':
                    raise ValueError(f'Source hash mismatch: {f}')
            records.append(dict(manifest=name, file=basename, status=status))
    # Metadata links are checked even when the archive omits other output files.
    m3 = locate(root, 'step3_run_metadata.json')
    m2 = locate(root, 'step2_run_metadata.json', False)
    if m2 is not None:
        expected=json.loads(m3.read_text()).get('source_stage2_metadata_sha256')
        if expected and expected!=hashlib.sha256(m2.read_bytes()).hexdigest():
            raise ValueError('Inconsistent Stage2/3 source chain')
    for name, field in [('step4_run_metadata.json', 'source_stage3_metadata_sha256'),
                        ('remaining_run_metadata.json', 'source_stage3_metadata_sha256')]:
        f = locate(root, name, False)
        if f:
            expected = json.loads(f.read_text()).get(field)
            if expected and expected != hashlib.sha256(m3.read_bytes()).hexdigest():
                raise ValueError(f'Inconsistent source chain: {name}')
    return pd.DataFrame(records)


def sample_masks(both):
    good = []
    for side in ['left', 'right']:
        status = both['solver_status_'+side]
        action = both['execution_action_'+side]
        good.append(status.isin(['Direct', 'Relaxed', 'Benchmark']) &
                    ~action.isin(['hold_after_infeasible', 'hold_unmatched']))
    return {'full_policy': np.ones(len(both), bool),
            'both_accepted': good[0] & good[1],
            'both_direct_only': both.direct_eligible_left.fillna(False) & both.direct_eligible_right.fillna(False)}


def paired_tables(daily, replicates, block, seed, existing=None):
    d = daily.copy()
    d['date'] = pd.to_datetime(d.date)
    calendar = pd.Index(sorted(d.date.unique()))
    d['position'] = calendar.get_indexer(d.date)
    definitions = step4.pair_definitions(d)
    pairs, individual, cache = [], [], {}
    for rec in definitions.to_dict('records'):
        left = d[d.experiment_id.eq(rec['left_experiment']) & d.model_id.eq(rec['left_model'])]
        right = d[d.experiment_id.eq(rec['right_experiment']) & d.model_id.eq(rec['right_model'])]
        both = left.merge(right, on='date', suffixes=('_left', '_right'), validate='one_to_one').sort_values('date')
        for scope, mask in sample_masks(both).items():
            z = both.loc[mask]
            a, b, rf, pos = [z[c].to_numpy() for c in
                             ['net_return_left', 'net_return_right', 'rf_daily_left', 'position_left']]
            if not np.allclose(rf, z.rf_daily_right, atol=1e-14, rtol=0):
                raise ValueError('Paired cash rates differ')
            row = dict(rec, scope=scope, n_days=len(z), n_windows=z.window_id_left.nunique(),
                       n_segments=len(es.segments(pos)) if len(pos) else 0,
                       correlation=float(np.corrcoef(a, b)[0, 1]) if len(a)>1 and min(np.std(a), np.std(b))>0 else np.nan,
                       interpretation='full implemented policy' if scope=='full_policy' else 'selected implementation subset; not unconditional performance')
            if len(z):
                for side, values in [('left', a), ('right', b)]:
                    metric = step4.metrics(values, rf, pos)
                    individual.append(dict(rec, scope=scope, side=side, n_days=len(z),
                        n_windows=z.window_id_left.nunique(), n_segments=row['n_segments'],
                        mean_daily_risky_exposure=float(z['risky_exposure_'+side].mean()),
                        mean_daily_cash_weight=float(z['cash_weight_'+side].mean()), **metric))
                ma, mb = step4.metrics(a, rf, pos), step4.metrics(b, rf, pos)
                row.update({'delta_'+key: ma[key]-mb[key] for key in ma})
                win = z.groupby('window_id_left')[['net_return_left', 'net_return_right']].apply(
                    lambda g: np.prod(1+g.net_return_left)-np.prod(1+g.net_return_right))
                row['window_win_rate'] = float(win.gt(0).mean())
            key = es.fingerprint(a,b,rf,pos,replicates=replicates,block=block,seed=seed)
            if key not in cache:
                old = existing[existing.inference_key.eq(key)] if existing is not None else pd.DataFrame()
                if not old.empty:
                    # Reuse only statistic/inference fields with an exact data/settings key.
                    columns = [c for c in old if c.startswith(('delta_','hac_','threshold_','annual_return_',
                               'sharpe_', 'ES95_', 'ES99_', 'inference_')) and not c.endswith('_holm')]
                    cache[key] = old.iloc[0][columns].to_dict()
                    cache[key]['block_length'] = block
                else:
                    cache[key] = es.paired_inference(a,b,rf,pos,replicates,block,seed)
            row.update(cache[key])
            pairs.append(row)
        print(f'Completion {rec["pair_id"]}: {rec["left_model"]}', flush=True)
    frame = es.adjust_unique(pd.DataFrame(pairs), P_COLUMNS)
    return frame, pd.DataFrame(individual)


def longest_run(values):
    best = current = 0
    for value in values:
        current = current+1 if value else 0
        best = max(best,current)
    return best


def constraint_audit(w, assets, daily, policy):
    keys = ['experiment_id','model_id','window_id']
    a = assets.copy()
    cap = policy['weight_upper']; tol = policy['constraint_tolerance']
    a['cap_pretrade_excess'] = np.maximum(a.posttrade_amount-cap, 0.)
    a['cap_postfee_excess'] = np.maximum(a.posttrade_weight-cap, 0.)
    a['cap_terminal_excess'] = np.maximum(a.terminal_weight-cap, 0.)
    a['fee_from_trades'] = policy['cost_buy']*a.buy+policy['cost_sell']*a.sell
    locked = a.locked.eq(True) if 'locked' in a else pd.Series(False,index=a.index)
    a['locked_trade'] = locked & ((a.posttrade_amount-a.initial_weight).abs()>tol)
    ag = a.groupby(keys).agg(max_cap_pretrade_excess=('cap_pretrade_excess','max'),
        max_cap_postfee_excess=('cap_postfee_excess','max'),max_cap_terminal_excess=('cap_terminal_excess','max'),
        fee_recomputed=('fee_from_trades','sum'),asset_amount=('posttrade_amount','sum'),locked_trades=('locked_trade','sum'))
    dg = daily.groupby(keys).agg(min_daily_exposure=('risky_exposure','min'),max_daily_exposure=('risky_exposure','max'))
    out = w.merge(ag,on=keys,validate='one_to_one').merge(dg,on=keys,validate='one_to_one')
    out['budget_error'] = abs(out.risky_exposure+out.cash_weight+out.trading_cost-1)
    out['fee_error'] = abs(out.trading_cost-out.fee_recomputed)
    out['floor_shortfall_pretrade'] = np.maximum(out.exposure_floor-out.risky_exposure,0.)
    out['floor_shortfall_daily'] = np.maximum(out.exposure_floor-out.min_daily_exposure,0.)
    out['floor_breach'] = out.floor_shortfall_pretrade.gt(tol)
    out['cap_pretrade_breach'] = out.max_cap_pretrade_excess.gt(tol)
    out['cap_postfee_breach'] = out.max_cap_postfee_excess.gt(tol)
    out['cap_terminal_breach'] = out.max_cap_terminal_excess.gt(tol)
    if out.budget_error.gt(tol).any() or out.fee_error.gt(tol).any() or out.locked_trades.gt(0).any():
        raise ValueError('Accounting/locked trade audit failed')
    summaries = []
    for identity,g in out.sort_values(keys).groupby(['experiment_id','model_id']):
        f = g.solver_status.eq('Failed')
        summaries.append(dict(experiment_id=identity[0],model_id=identity[1],windows=len(g),
            failed_windows=int(f.sum()),failed_floor_breaches=int((f&g.floor_breach).sum()),
            failed_cap_breaches=int((f&g.cap_pretrade_breach).sum()),
            postfee_cap_breach_windows=int(g.cap_postfee_breach.sum()),terminal_cap_breach_windows=int(g.cap_terminal_breach.sum()),
            longest_failed_run=longest_run(f),longest_floor_breach_run=longest_run(g.floor_breach),
            max_floor_shortfall=float(g.floor_shortfall_pretrade.max()),
            max_cap_pretrade_excess=float(g.max_cap_pretrade_excess.max()),
            interpretation='Decision-state audit; daily per-asset maxima need original asset returns. Postfee and pretrade denominators differ.'))
    return out,pd.DataFrame(summaries)


def source_evidence(input_dir=None):
    rows=[]
    for name in ['step1_run_metadata.json','input_data_audit.csv','implementation_parameters.csv',
                 'risk_free_alignment_audit.csv','universe_asset_audit.csv']:
        f=Path(input_dir)/name if input_dir else None
        rows.append(dict(item=name,status='provided_not_independently_certified' if f and f.is_file() else 'not_supplied',
            note='File existence is not proof of raw corporate actions or actual execution.'))
    rows.extend([dict(item='corporate_action_and_adjustment_ledger',status='not_certified',
                      note='Requires raw prices, adjustment flags and event/reconciliation evidence.'),
                 dict(item='real_close_order_fills',status='not_certified',
                      note='Positive turnover is a proxy; no actual order ledger supplied.')])
    return pd.DataFrame(rows)


def auxiliary_tables(source):
    tables={}
    path=locate(source,'step3_cross_risks.csv',False)
    if path:
        c=pd.read_csv(path)
        order=(c.risk_CVaR05>=c.risk_CVaR20-1e-10)&(c.risk_CVaR20>=c.risk_RVaR-1e-10)
        residual=abs(c.risk_CVaR20-.25*c.risk_CVaR05-.75*c.risk_RVaR)
        if not order.all() or residual.max()>1e-10:raise ValueError('Cross-risk theory check failed')
        tables['completion_theory_checks.csv']=pd.DataFrame([dict(check='same_portfolio_tail_order',
            records=len(c),passed=bool(order.all()),max_residual=np.nan),
            dict(check='CVaR20_decomposition',records=len(c),passed=True,max_residual=float(residual.max()))])
    path=locate(source,'remaining_endpoint_draws.parquet',False)
    if path:
        b=pd.read_parquet(path)
        g=b.groupby(['model_id','window_id','block_length'])
        summary=g.agg(draws=('replicate','size'),accepted=('solver_status',lambda x:x.isin(['Direct','Relaxed']).sum()),
            median_accepted_weight_l1=('weight_l1','median'),complete_valuations=('conditional_valuation_status',lambda x:x.eq('complete').sum())).reset_index()
        summary['failure_rate']=1-summary.accepted/summary.draws
        summary['interpretation']='Conditional fixed-reference one-window draws; intervals condition on accepted optimizations'
        tables['completion_endpoint_window_failures.csv']=summary
    path=locate(source,'ki_asset_estimates.csv',False)
    if path:
        k=pd.read_csv(path);rows=[]
        for variant,g in k.groupby('variant'):
            valid=g[g.eligible.fillna(False)]
            rows.append(dict(variant=variant,records=len(g),eligible=len(valid),median_k=valid.k.median(),
                median_loss=valid.loss.median(),median_free_loss=valid.free_loss.median(),
                median_restriction_loss=valid.restriction_loss.median(),
                median_free_left_power=valid.free_left_power.median(),median_free_right_power=valid.free_right_power.median(),
                interpretation='Conditional target-curve fit, not behavior or validation of the common shape family'))
        tables['completion_shape_fit.csv']=pd.DataFrame(rows)
        a=k[k.variant.eq('median_balanced')].set_index(['window_id','stock_code']).k
        b=k[k.variant.eq('median_equal_shoulders')].set_index(['window_id','stock_code']).k
        common=a.dropna().index.intersection(b.dropna().index)
        if len(common):
            tables['completion_shape_convention_check.csv']=pd.DataFrame([dict(
                variants='median_balanced vs median_equal_shoulders',records=len(common),
                max_k_difference=float(abs(a.loc[common]-b.loc[common]).max()),
                independent_shape_checks=False,
                reason='Core loss is k-independent; equal rescaling of both shoulder losses leaves argmin unchanged')])
    return tables


def run(source,out,replicates=2000,block=10,seed=42,input_dir=None,pilot=False):
    if replicates<2000 and not pilot:
        raise ValueError('Formal completion needs >=2000 replicates; use --pilot for a smoke test')
    out=Path(out)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError('Use a fresh completion output directory')
    integrity=verify_sources(source)
    meta=json.loads(locate(source,'step3_run_metadata.json').read_text())
    if meta.get('scope')!='full' and not pilot:
        raise ValueError('Pilot Stage3 cannot be reported as formal completion; add --pilot')
    daily=pd.read_parquet(locate(source,'step3_daily_returns.parquet'))
    assets=pd.read_parquet(locate(source,'step3_assets.parquet'))
    windows=pd.read_csv(locate(source,'step3_window_backtests.csv'))
    if daily.duplicated(['experiment_id','model_id','date']).any() or not np.isfinite(daily.net_return).all():
        raise ValueError('Invalid daily source')
    for n in ['solver_status','execution_action','direct_eligible']:
        if n not in daily:raise ValueError('New Step3 execution records required')
    existing_path=locate(source,'step4_paired_comparisons.csv',False)
    existing=pd.read_csv(existing_path) if existing_path else None
    previous_metadata=locate(source,'step4_run_metadata.json',False)
    if previous_metadata is None or json.loads(previous_metadata.read_text()).get('statistics_code_sha256')!=hashlib.sha256(Path(es.__file__).read_bytes()).hexdigest():
        existing=None
    pairs,individual=paired_tables(daily,replicates,block,seed,existing)
    audit,summary=constraint_audit(windows,assets,daily,meta['policy'])
    attempts=pd.read_csv(locate(source,'step3_attempts.csv'))
    timing=attempts.groupby(['experiment_id','solver_code']).solver_time_sec.agg(['count','median','max']).reset_index()
    dims=assets[assets.mu.notna()].groupby(['experiment_id','model_id','window_id']).size().rename('n_assets').reset_index()
    dims['n_variables']=3*dims.n_assets;dims['core_inequality_rows']=3
    dims['trade_identity_rows']=dims.n_assets
    dims['dimension_basis']='Step3 x,buy,sell: 3 budget/exposure/return inequalities, n trade equalities; variable bounds/locks separate; MV is QP'
    evidence=source_evidence(input_dir)
    tables={'completion_pairs.csv':pairs,'completion_paired_performance.csv':individual,
            'completion_neutral_k1.csv':pairs[pairs.left_experiment.eq('K1_NEUTRAL')&pairs.purpose.eq('rvar_cvar')],
            'completion_constraint_windows.csv':audit,'completion_constraint_summary.csv':summary,
            'completion_solver_timing.csv':timing,'completion_model_dimensions.csv':dims,
            'completion_source_integrity.csv':integrity,'completion_input_evidence.csv':evidence}
    tables.update(auxiliary_tables(source))
    out.mkdir(parents=True,exist_ok=True)
    for name,frame in tables.items():frame.to_csv(out/name,index=False,encoding='utf-8-sig')
    metadata=dict(status='pilot_diagnostics_only' if pilot else 'computed_completion_with_open_execution_and_data_evidence',
        pipeline_model='ModelA',replicates=replicates,block_length=block,seed=seed,
        source_step3_sha256=hashlib.sha256(locate(source,'step3_run_metadata.json').read_bytes()).hexdigest(),
        complete_raw_data_certificate=False,actual_execution_certificate=False,
        completion_code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        statistics_code_sha256=hashlib.sha256(Path(es.__file__).read_bytes()).hexdigest(),
        scope_rule='both_accepted includes Direct and Relaxed; failures excluded; full_policy retained separately',
        multiplicity='Holm within purpose x scope x statistic, deduplicated; includes annual-return bootstrap',
        limitations=['Subset comparisons are not causal or unconditional whole-history estimates',
                     'Original same-close execution remains an assumption until separately evaluated',
                     'No inference of investor preferences from conditional shape projections'],
        outputs={name:hashlib.sha256((out/name).read_bytes()).hexdigest() for name in tables})
    (out/'completion_metadata.json').write_text(json.dumps(metadata,indent=2),encoding='utf-8')
    return metadata


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--input',type=Path);p.add_argument('--replicates',type=int,default=2000)
    p.add_argument('--block',type=int,default=10);p.add_argument('--seed',type=int,default=42)
    p.add_argument('--pilot',action='store_true')
    a=p.parse_args();print(run(a.source,a.out,a.replicates,a.block,a.seed,a.input,a.pilot)['status'])


if __name__=='__main__':main()
