"""Isolated synthetic end-to-end checks, with production gates intact."""
import contextlib,io,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import pandas as pd
import step2,step3,step4,step5,remaining_experiments
from test_support import make_inputs

class PipelineTests(unittest.TestCase):
    def test_all_regular_experiments_uncertainty_inference_and_figures(self):
        with tempfile.TemporaryDirectory() as td,contextlib.redirect_stdout(io.StringIO()):
            root=Path(td);r1,data,policy=make_inputs(root)
            original=step2.experiment_specs
            def specs(p):
                s=original(p);return s+[dict(s[0],model_id='CVaR_forced_failure',target_annual=100.)]
            with patch.object(step2,'experiment_specs',side_effect=specs):
                m2=step2.run_stage2(r1,root/'results2',root/'results2',policy,'standard')
                self.assertEqual(m2['status'],'complete_stage2_inputs')
                m3=step3.run_stage3(r1,root/'results2',data,root/'results3')
                self.assertEqual(m3['status'],'accepted_modelA_stage3')
                daily=pd.read_parquet(root/'results3/step3_daily_returns.parquet')
                f=daily[daily.model_id.eq('CVaR_forced_failure')]
                np.testing.assert_allclose(f.net_return,f.rf_daily,rtol=0,atol=1e-12);self.assertEqual(len(f),120)
                windows=pd.read_csv(root/'results3/step3_window_backtests.csv')
                self.assertTrue({'K1_NEUTRAL','K_HETEROG','ENDPOINT_ALT','RVAR_INTERVAL'}<=set(windows.experiment_id))
                m=remaining_experiments.run(r1,root/'results2',root/'results3',root/'remaining',replicates=2,window_limit=1)
                self.assertEqual(m['scope'],'pilot')
            m4=step4.run(root/'results3',root/'results4',replicates=100)
            self.assertEqual(m4['status'],'accepted_modelA_stage4')
            paired=pd.read_csv(root/'results4/step4_paired_comparisons.csv')
            main=paired[paired.left_model.eq('RVaR_neu')&paired.right_model.eq('CVaR_neu')&paired.scope.eq('full_policy')].iloc[0]
            duplicate=paired[paired.left_model.eq('RVaR_kpure08')&paired.right_model.eq('CVaR_kpure08')&paired.scope.eq('full_policy')].iloc[0]
            self.assertEqual(main.inference_key,duplicate.inference_key)
            self.assertEqual(main.sharpe_bootstrap_p,duplicate.sharpe_bootstrap_p)
            self.assertFalse(pd.read_csv(root/'results4/step4_neutral_reference.csv').empty)
            self.assertFalse(pd.read_csv(root/'results4/step4_heterogeneity_mechanisms.csv').empty)
            self.assertIn('ES95_bootstrap_p',paired)
            m5=step5.run(root/'results3',root/'results4',root/'results5',allow_pilot=True)
            self.assertEqual(m5['status'],'accepted_modelA_stage5')
            self.assertTrue((root/'results5/app_heterogeneity.pdf').exists())
            with (root/'results3/step3_window_backtests.csv').open('a') as h:h.write('\n')
            with self.assertRaises(ValueError):step4.run(root/'results3',root/'tampered',replicates=100)

if __name__=='__main__':unittest.main()
