"""Complete a previously computed Model A run without overwriting it.

Post-processing works with examine's flattened files. Execution sensitivities
additionally require the full original results1, results2 and index data.
"""
from pathlib import Path
import argparse,json
import pandas as pd
import review_completion,controlled_diagnostics,data_evidence_audit,execution_sensitivity
import step4,step5


def run(source,out,input_dir=None,data_dir=None,cache_dir=None,with_execution=False,
        replicates=2000,pilot=False):
    source,out=Path(source),Path(out)
    if out.exists() and any(out.iterdir()):raise FileExistsError('Use a new output root; original outputs are never overwritten')
    if with_execution and (input_dir is None or data_dir is None):
        raise ValueError('--with-execution requires --input and --data')
    if not pilot and replicates<2000:raise ValueError('Formal inference requires >=2000 replicates')
    # Validate execution inputs before starting expensive calculations.
    stage2=None
    if with_execution:
        stage2=review_completion.locate(source,'step2_run_metadata.json').parent
        required=['step2_windows.csv','lp_policy_lock.json']
        for name in required:
            if not (stage2/name).is_file():raise FileNotFoundError('Full results2 needed, flattened examine archive is insufficient: '+name)
        for name in ['full_return_matrix.csv','full_turnover_matrix.csv','step1_run_metadata.json']:
            if not (Path(input_dir)/name).is_file():raise FileNotFoundError('Missing real Stage1 input: '+name)
    out.mkdir(parents=True,exist_ok=True)
    base=review_completion.run(source,out/'baseline_completion',replicates=replicates,
                               input_dir=input_dir,pilot=pilot)
    source_windows=pd.read_csv(review_completion.locate(source,'step3_window_backtests.csv'))
    experiments=set(source_windows.experiment_id)
    checks=[dict(comment='R2-M2 / R4-M2',item='conditional_k_dispersion',
                 status='computed_conditional_only' if review_completion.locate(source,'ki_conditional_dispersion.csv',False) else 'not_supplied'),
            dict(comment='R2-M3 / R4-M6',item='endpoint_and_interval_sensitivity',
                 status='recorded_in_source_check_tables' if {'ENDPOINT_ALT','RVAR_INTERVAL'}<=experiments else 'not_supplied_or_not_run'),
            dict(comment='R4-M4',item='k1_both_accepted_sample',status='computed'),
            dict(comment='R2-M6 / R3',item='failure_constraint_deviations',status='computed'),
            dict(comment='R2-M4',item='tail_substitution_and_real_losses',status='computed_in_source_and_completion'),
            dict(comment='R2-M7',item='formal_state_contrasts',status='supplied' if review_completion.locate(source,'step4_state_contrasts.csv',False) else 'not_supplied'),
            dict(comment='R2-Minor / R4-Minor3',item='solver_dimensions_and_timing',status='computed'),
            dict(comment='R2-M3 consider',item='EVT_or_parametric_calibration',status='not_implemented_optional'),
            dict(comment='R3 if_feasible',item='second_market',status='not_implemented_optional'),
            dict(comment='R3',item='public_code_archive',status='requires_public_release'),
            dict(comment='R2-Minor',item='corporate_action_ledger',status='requires_external_evidence')]
    if input_dir is not None and data_dir is not None:
        data_evidence_audit.run(input_dir,data_dir,out/'data_audit',cache_dir)
        checks.append(dict(comment='R2-Minor',item='available_raw_cache_reconciliation',status='computed_see_coverage'))
    else:
        checks.append(dict(comment='R2-Minor',item='available_raw_cache_reconciliation',status='not_run_missing_inputs'))
    if with_execution:
        controlled_diagnostics.run(input_dir,stage2,out/'controlled_diagnostics')
        bd=pd.read_parquet(review_completion.locate(source,'step3_daily_returns.parquet'))
        baseline={key:step4.metrics(g.net_return,g.rf_daily) for key,g in bd.groupby(['experiment_id','model_id'])}
        comparisons=[]
        for timing,fallback in [('formation_close','repair'),('next_close','hold'),('next_close','repair')]:
            name=timing+'_'+fallback;root=out/name
            execution_sensitivity.run(input_dir,stage2,data_dir,root/'results3',timing,fallback,2 if pilot else None)
            step4.run(root/'results3',root/'results4',replicates=replicates,allow_pilot=pilot)
            step5.run(root/'results3',root/'results4',root/'results5',allow_pilot=pilot)
            review_completion.run(root,root/'completion',replicates=replicates,input_dir=input_dir,pilot=pilot)
            pf=pd.read_csv(root/'results4/step4_performance.csv')
            ww=pd.read_csv(root/'results3/step3_window_backtests.csv')
            for record in pf.to_dict('records'):
                key=(record['experiment_id'],record['model_id'])
                if key not in baseline:continue
                z=ww[ww.experiment_id.eq(key[0])&ww.model_id.eq(key[1])]
                comparisons.append(dict(variant=name,experiment_id=key[0],model_id=key[1],
                    n_days=record['n_days'],failed_target_windows=int(z.solver_status.eq('Failed').sum()),
                    repaired_windows=int(z.execution_action.isin(['minimum_turnover_repair','minimum_turnover_repair_execution_checked']).sum()),
                    **{'delta_'+k:record[k]-baseline[key][k] for k in ['annual_return','sharpe','max_drawdown','ES95']},
                    interpretation='Descriptive execution-policy sensitivity; not causal and not a significance test'))
        pd.DataFrame(comparisons).to_csv(out/'execution_policy_comparison.csv',index=False)
        checks.append(dict(comment='execution_audit',item='delayed_and_repair_sensitivities',status='computed_under_disclosed_close_value_assumption'))
    else:
        checks.append(dict(comment='execution_audit',item='delayed_and_repair_sensitivities',status='not_run_requires_full_inputs_and_flag'))
    pd.DataFrame(checks).to_csv(out/'reviewer_evidence_checklist.csv',index=False,encoding='utf-8-sig')
    metadata=dict(status='pilot_only' if pilot else 'computed_requested_diagnostics_with_disclosed_open_evidence',
        with_execution=with_execution,all_reviewer_comments_certified=False,
        actual_fills_certified=False,corporate_actions_certified=False,
        note='Theory, narrative, public release and real-world execution cannot be certified by output existence.')
    (out/'finish_metadata.json').write_text(json.dumps(metadata,indent=2))
    return metadata


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--input',type=Path);p.add_argument('--data',type=Path);p.add_argument('--cache',type=Path)
    p.add_argument('--with-execution',action='store_true');p.add_argument('--replicates',type=int,default=2000)
    p.add_argument('--pilot',action='store_true')
    a=p.parse_args();print(run(a.source,a.out,a.input,a.data,a.cache,a.with_execution,a.replicates,a.pilot)['status'])


if __name__=='__main__':main()
