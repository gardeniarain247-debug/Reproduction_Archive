"""Unified Model A baseline and completion entry point."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

import baseline_pipeline
import finish_all

BASE = Path(__file__).resolve().parent


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', type=Path)
    p.add_argument('--data', type=Path)
    p.add_argument('--cache', type=Path)
    p.add_argument('--source', type=Path, help='Existing results: skip Step2–5 and complete these results')
    p.add_argument('--out-root', type=Path, default=BASE/'modelA_unified_results')
    p.add_argument('--with-execution', action='store_true')
    p.add_argument('--tests-only', action='store_true')
    p.add_argument('--pilot', action='store_true')
    p.add_argument('--replicates', type=int, default=2000)
    p.add_argument('--endpoint-replicates', type=int, default=300)
    a = p.parse_args(argv)
    if a.tests_only:
        return subprocess.call([sys.executable, '-m', 'unittest', 'discover',
                                '-s', str(BASE), '-p', 'test_*.py', '-v'], cwd=BASE)
    if not a.pilot and a.replicates < 2000:
        p.error('Formal inference requires >=2000 replicates')
    if a.source is not None and (a.input is None) != (a.data is None):
        p.error('Supply --input and --data together, or omit both for statistics only')
    if a.source is not None and a.with_execution and (a.input is None or a.data is None):
        p.error('--source with --with-execution requires --input and --data')
    out = a.out_root.resolve()
    if out.exists() and any(out.iterdir()):
        p.error('Use a new/empty --out-root to preserve existing results')
    if a.source is not None and not a.source.is_dir():
        p.error('--source must be an existing results directory')
    if a.source is not None and (out == a.source.resolve() or a.source.resolve().is_relative_to(out)):
        p.error('--out-root must not contain or equal --source')
    input_dir = a.input
    data_dir = a.data
    source = a.source
    if source is None:
        input_dir = input_dir or BASE/'results1'
        data_dir = data_dir or BASE/'data'
        print('Step 1/2: baseline Step2–5 and original reviewer experiments', flush=True)
        baseline = baseline_pipeline.run(input_dir, data_dir, out, pilot=a.pilot,
            replicates=a.replicates, endpoint_replicates=a.endpoint_replicates)
        expected = 'pilot_computed_not_formal_evidence' if a.pilot else 'real_data_calculations_completed_with_disclosed_limits'
        if baseline['status'] != expected:
            raise RuntimeError('Baseline incomplete; completion not certified. Inspect review_completion.json')
        source = out
    else:
        print('Using existing results; Step2–5 will not be rerun', flush=True)
    print('Step 2/2: completion statistics and requested execution diagnostics', flush=True)
    completion = finish_all.run(source, out/'completion', input_dir, data_dir, a.cache,
        with_execution=a.with_execution, replicates=a.replicates, pilot=a.pilot)
    manifest = dict(status=completion['status'], source=str(Path(source).resolve()),
        baseline_rerun=a.source is None, with_execution=a.with_execution,
        completion_directory=str(out/'completion'), all_reviewer_comments_certified=False)
    (out/'unified_run_metadata.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print('UNIFIED RUN FINISHED: '+manifest['status'], flush=True)
    if not a.with_execution:
        print('Execution variants not requested. Use --with-execution with full inputs to compute them.', flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
