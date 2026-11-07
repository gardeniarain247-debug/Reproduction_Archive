"""Reconcile supplied Stage1 matrices with cached prices; no fabricated event ledger."""
from pathlib import Path
import argparse,json,hashlib
import numpy as np
import pandas as pd
import step1


def run(input_dir,data_dir,out,cache_dir=None):
    input_dir,data_dir,out=map(Path,[input_dir,data_dir,out])
    if out.exists() and any(out.iterdir()):raise FileExistsError('Use a fresh data-audit directory')
    source=json.loads((input_dir/'step1_run_metadata.json').read_text())
    settings=source['settings'];cache=Path(cache_dir) if cache_dir else input_dir/'baostock_cache'
    records=[]
    for item in source['outputs']:
        f=input_dir/item['path'].replace('\\','/').rsplit('/',1)[-1]
        if not f.is_file() or step1.sha256_file(f)!=item['sha256']:raise ValueError(f'Stage1 hash mismatch: {f}')
    observed=step1.read_daily_matrix(input_dir/'full_return_matrix.csv',True)
    turn=step1.read_daily_matrix(input_dir/'full_turnover_matrix.csv',True)
    for code in observed.columns:
        f=cache/(str(code)+'.csv')
        if not f.is_file():
            records.append(dict(stock_code=code,status='cache_not_supplied'));continue
        raw=pd.read_csv(f,parse_dates=['date']).set_index('date').sort_index()
        if raw.index.has_duplicates:raise ValueError(f'Duplicate cached dates: {code}')
        raw=raw.loc[settings['start_date']:settings['end_date']]
        prices=pd.to_numeric(raw.close,errors='raise').reindex(observed.index.union(raw.index).sort_values())
        expected=prices.pct_change(fill_method=None).reindex(observed.index)
        actual=observed[code]
        missing=int((expected.isna()!=actual.isna()).sum())
        delta=float((expected-actual).abs().max()) if (expected.notna()&actual.notna()).any() else np.nan
        rawturn=pd.to_numeric(raw.turn,errors='raise').reindex(turn.index)
        td=float((rawturn-turn[code]).abs().max())
        turn_missing=int((rawturn.isna()!=turn[code].isna()).sum())
        status='matched' if not missing and not turn_missing and (not np.isfinite(delta) or delta<=1e-10) and (not np.isfinite(td) or td<=1e-10) else 'mismatch'
        records.append(dict(stock_code=code,status=status,return_nan_pattern_differences=missing,
            max_return_error=delta,max_turnover_error=td,turnover_nan_pattern_differences=turn_missing,cache_sha256=hashlib.sha256(f.read_bytes()).hexdigest(),
            cache_adjustment_flag=';'.join(map(str,raw.adjustflag.dropna().unique())) if 'adjustflag' in raw else 'not_stored'))
    frame=pd.DataFrame(records)
    if frame.status.eq('mismatch').any():raise ValueError('Cached-price/matrix reconciliation failed')
    rf=step1.read_daily_matrix(input_dir/'risk_free_rate.csv').iloc[:,0]
    result=dict(status='reconciled_available_caches_with_open_provenance',declared_adjustflag=settings.get('adjustflag'),
        cached_assets_matched=int(frame.status.eq('matched').sum()),assets=len(frame),
        missing_cache_assets=int(frame.status.eq('cache_not_supplied').sum()),
        rf_daily_min=float(rf.min()),rf_daily_max=float(rf.max()),rf_unique=int(rf.nunique()),
        cash_rate_is_time_varying=bool(rf.nunique()>1),
        corporate_actions_independently_reconciled=False,index_total_return_comparability_verified=False,
        limitations=['Cached close values alone cannot prove which adjustment option was originally used',
                     'An original query log/vendor flag and event reconciliation are still needed',
                     'Price-index versus dividend-adjusted stock-return comparability needs index provenance'])
    out.mkdir(parents=True,exist_ok=True);frame.to_csv(out/'data_cache_reconciliation.csv',index=False)
    (out/'data_evidence_metadata.json').write_text(json.dumps(result,indent=2))
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for n in ['input','data','out']:p.add_argument('--'+n,type=Path,required=True)
    p.add_argument('--cache',type=Path)
    a=p.parse_args();print(run(a.input,a.data,a.out,a.cache)['status'])
