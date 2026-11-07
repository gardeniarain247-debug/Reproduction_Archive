"""One entry point for Model A reviewer experiments; never fabricates absent data."""
import argparse
from pathlib import Path
import json
import subprocess
import sys
import time
import pandas as pd
import step1,step2,step3,step4,step5,remaining_experiments
BASE=Path(__file__).resolve().parent


def run(input_dir,data_dir,out_root,pilot=False,overwrite=False,replicates=2000,endpoint_replicates=300):
    input_dir=Path(input_dir);data_dir=Path(data_dir);out=Path(out_root)
    required=[*step1.CORE_FILES,'endpoint_calibration.csv','step1_run_metadata.json','bootstrap_indices.npz']
    missing=[str(input_dir/name) for name in required if not (input_dir/name).is_file()]
    if missing:raise FileNotFoundError('Required real Stage1 inputs absent: '+', '.join(missing))
    if not (data_dir/'sh50_index_return.csv').is_file():raise FileNotFoundError('Index returns required for reviewer market-state diagnostics')
    source=step3.checked_manifest(input_dir,'step1_run_metadata.json','audit_pass_with_review_items')
    if not pilot and source.get('synthetic_fixture'):raise ValueError('Synthetic fixture cannot certify real empirical completion')
    if not pilot and (replicates<2000 or endpoint_replicates!=300):raise ValueError('Formal default: >=2000 paired draws and 300 endpoint draws')
    policy=step2.load_policy()
    if not pilot and source.get('settings',{}).get('expected_windows',40)!=40:raise ValueError('Formal study requires 40 windows')
    # A pilot starts with a full static source run only if it exists; otherwise use
    # small-limit Step2 outputs and a separate explicit pilot-compatible execution.
    out.mkdir(parents=True,exist_ok=True)
    # Run the regular reference set, not the old static-bootstrap trajectory.
    # Actual-state conditional bootstrap is performed below by remaining_experiments.
    step2.run_stage2(input_dir,out/'results2',out/'results2',policy,'standard',overwrite)
    step3.run_stage3(input_dir,out/'results2',data_dir,out/'results3',overwrite,2 if pilot else None)
    remaining_experiments.run(input_dir,out/'results2',out/'results3',out/'results_remaining',overwrite,
                              endpoint_replicates,2 if pilot else None)
    step4.run(out/'results3',out/'results4',overwrite,replicates,allow_pilot=pilot)
    step5.run(out/'results3',out/'results4',out/'results5',overwrite,pilot)
    windows=pd.read_csv(out/'results3/step3_window_backtests.csv')
    experiments=set(windows.experiment_id);required_e={'BASELINE_AUDIT','CVAR02_ROBUSTNESS','KPURE_CONTROL','K1_NEUTRAL','K_HETEROG','ENDPOINT_ALT','RVAR_INTERVAL'}
    checks=[dict(task=e,computed=e in experiments) for e in sorted(required_e)]
    required_files=['step4_structure_summary.csv','step4_state_performance.csv','step4_state_contrasts.csv','step4_neutral_reference.csv','step4_heterogeneity_mechanisms.csv','step4_inference_sensitivity.csv']
    checks.extend(dict(task=name,computed=(out/'results4'/name).is_file()) for name in required_files)
    frame=pd.DataFrame(checks);frame.to_csv(out/'review_completion_checklist.csv',index=False)
    status='pilot_computed_not_formal_evidence' if pilot else ('real_data_calculations_completed_with_disclosed_limits' if frame.computed.all() else 'incomplete')
    metadata=dict(status=status,pipeline_model='ModelA',real_data_required=True,pilot=pilot,
        limits=['Conditional ki projection is not identified investor preference dispersion','Turnover proxy does not certify close-price execution or corporate-action accounting','Statistical computation does not guarantee significant findings'],
        source_stage1_metadata_sha256=step1.sha256_file(input_dir/'step1_run_metadata.json'),checks=checks)
    (out/'review_completion.json').write_text(json.dumps(metadata,indent=2))
    return metadata


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--input',type=Path,default=BASE/'results1');p.add_argument('--data',type=Path,default=BASE/'data')
    p.add_argument('--out-root',type=Path,default=BASE/'modelA_full_results');p.add_argument('--tests-only',action='store_true')
    p.add_argument('--pilot',action='store_true');p.add_argument('--overwrite',action='store_true');p.add_argument('--replicates',type=int,default=2000);p.add_argument('--endpoint-replicates',type=int,default=300)
    a=p.parse_args(argv)
    if a.tests_only:return subprocess.call([sys.executable,'-m','unittest','discover','-s',str(BASE),'-p','test_*.py','-v'],cwd=BASE)
    result=run(a.input,a.data,a.out_root,a.pilot,a.overwrite,a.replicates,a.endpoint_replicates);print(result['status']);return 0 if result['status']!='incomplete' else 2
if __name__=='__main__':raise SystemExit(main())
