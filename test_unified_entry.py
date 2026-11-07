"""Check actual unified entry dispatch, rather than output-file existence."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import run_all


class UnifiedEntryTests(unittest.TestCase):
    def test_full_pipeline_automatically_calls_completion(self):
        with tempfile.TemporaryDirectory() as td:
            out = Path(td)/'out'
            def baseline(*args, **kwargs):
                out.mkdir()
                return {'status': 'real_data_calculations_completed_with_disclosed_limits'}
            with patch.object(run_all.baseline_pipeline, 'run', side_effect=baseline) as b, \
                 patch.object(run_all.finish_all, 'run', return_value={'status':'computed'}) as f:
                self.assertEqual(run_all.main(['--out-root',str(out),'--with-execution']),0)
                b.assert_called_once()
                self.assertEqual(f.call_args.args[0].resolve(),out.resolve())
                self.assertEqual(f.call_args.args[1].resolve(),(out/'completion').resolve())
                self.assertTrue(f.call_args.kwargs['with_execution'])

    def test_existing_results_skip_baseline(self):
        with tempfile.TemporaryDirectory() as td:
            source = Path(td)/'source'; source.mkdir()
            out = Path(td)/'out'
            def complete(*args, **kwargs):
                (out/'completion').mkdir(parents=True)
                return {'status':'computed'}
            with patch.object(run_all.baseline_pipeline,'run') as b, \
                 patch.object(run_all.finish_all,'run',side_effect=complete) as f:
                self.assertEqual(run_all.main(['--source',str(source),'--out-root',str(out)]),0)
                b.assert_not_called()
                self.assertFalse(f.call_args.kwargs['with_execution'])

    def test_execution_without_inputs_rejected_before_work(self):
        with tempfile.TemporaryDirectory() as td:
            with patch.object(run_all.finish_all,'run') as f:
                with self.assertRaises(SystemExit):
                    run_all.main(['--source',td,'--with-execution'])
                f.assert_not_called()

    def test_nonempty_output_rejected_before_baseline(self):
        with tempfile.TemporaryDirectory() as td:
            (Path(td)/'keep.txt').write_text('keep')
            with patch.object(run_all.baseline_pipeline,'run') as b:
                with self.assertRaises(SystemExit):
                    run_all.main(['--out-root',td])
                b.assert_not_called()
