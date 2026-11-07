"""Self-contained synthetic fixtures; never load or bless old Model B outputs."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import step1,step2


def make_inputs(root,n_windows=2):
    root=Path(root);r1=root/'results1';r1.mkdir(parents=True);data=root/'data';data.mkdir()
    dates=pd.bdate_range('2019-01-01',periods=250+n_windows*60)
    codes=[f'{600000+i:06d}' for i in range(7)];rng=np.random.default_rng(12)
    returns=pd.DataFrame(rng.normal(.007,.012,(len(dates),7)),index=dates,columns=codes)
    turn=pd.DataFrame(1.,index=dates,columns=codes)
    if n_windows>1:turn.loc[dates[309],codes[0]]=0.
    rf=pd.DataFrame({'rf':.0001},index=dates)
    windows=pd.DataFrame([dict(window_id=j,est_start=dates[j*60],est_end=dates[249+j*60],test_start=dates[250+j*60],test_end=dates[309+j*60]) for j in range(n_windows)])
    universe=[];params=[];endpoints=[]
    for win in windows.itertuples(index=False):
        pool=codes[:6] if win.window_id==0 else codes[1:]
        for code in pool:
            universe.append(dict(window_id=win.window_id,stock_code=code))
            for scheme,prob in step1.ENDPOINT_SCHEMES.items():
                ep=np.quantile(returns.loc[win.est_start:win.est_end,code],prob)
                row=dict(window_id=win.window_id,stock_code=code,endpoint_scheme=scheme,**dict(zip('abcd',ep)))
                endpoints.append(row)
                if scheme=='baseline':params.append({k:v for k,v in row.items() if k!='endpoint_scheme'})
    returns.to_csv(r1/'full_return_matrix.csv');turn.to_csv(r1/'full_turnover_matrix.csv');rf.to_csv(r1/'risk_free_rate.csv')
    for name,frame in [('rolling_windows.csv',windows),('windows_universe.csv',pd.DataFrame(universe)),('fuzzy_parameters.csv',pd.DataFrame(params)),('endpoint_calibration.csv',pd.DataFrame(endpoints))]:frame.to_csv(r1/name,index=False)
    settings=step1.load_settings();settings.update(expected_windows=n_windows,bootstrap_replicates=3)
    np.savez_compressed(r1/'bootstrap_indices.npz',**step1.bootstrap_window_indices(windows,returns,settings))
    files=list(r1.iterdir());meta=dict(status='audit_pass_with_review_items',audit_counts={'failed':0},run_id='synthetic',synthetic_fixture=True,settings=settings,outputs=[dict(path=str(p),sha256=step1.sha256_file(p)) for p in files])
    (r1/'step1_run_metadata.json').write_text(json.dumps(meta))
    pd.DataFrame({'return':returns.mean(axis=1)}).to_csv(data/'sh50_index_return.csv')
    policy=step2.load_policy();policy['expected_windows']=n_windows;policy['bootstrap_replicates']=3
    return r1,data,policy
