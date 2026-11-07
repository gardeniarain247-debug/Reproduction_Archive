"""Independent numeric oracles for inference, not self-matching expected output."""
import unittest
import numpy as np
import pandas as pd
from scipy.stats import norm
import step4
import empirical_statistics as es

class StatisticsTests(unittest.TestCase):
    def test_hac_matches_dense_calendar_kernel(self):
        x=np.array([.01,-.02,.02,-.01,.03,-.03,.02,.01,-.02,.01]);pos=np.array([0,1,2,8,9,10,11,12,18,19]);q=3
        u=x-x.mean();kernel=np.maximum(0,1-abs(pos[:,None]-pos[None,:])/(q+1))
        expected=np.sqrt(u@kernel@u)/len(x);mean,se,p=es.hac_mean(x,pos,q)
        self.assertAlmostEqual(se,expected,14);self.assertAlmostEqual(p,2*norm.sf(abs(x.mean()/expected)))
    def test_hac_does_not_close_calendar_gaps(self):
        x=np.arange(12.)**2;pos=np.r_[np.arange(6),np.arange(20,26)]
        self.assertNotEqual(es.hac_mean(x,pos,3)[1],es.hac_mean(x,np.arange(12),3)[1])
    def test_duplicate_and_nonfinite_inputs_rejected(self):
        with self.assertRaises(ValueError):es.hac_mean([1.,2.],[0,0])
        with self.assertRaises(ValueError):es.hac_mean([np.nan],[0])
    def test_zero_difference_not_claimed_significant(self):
        self.assertTrue(np.isnan(es.hac_mean(np.zeros(60),np.arange(60))[2]))
    def test_short_segment_not_silently_dropped_or_shortened(self):
        with self.assertRaisesRegex(ValueError,'insufficient_blocks'):es.block_indices([0,1,10,11],3,np.random.default_rng(1))
    def test_bootstrap_preserves_segment_lengths(self):
        pos=np.r_[np.arange(10),np.arange(100,110)];ix=es.block_indices(pos,4,np.random.default_rng(1))
        self.assertEqual(len(ix),20);self.assertTrue((ix[:10]<10).all());self.assertTrue((ix[10:]>=10).all())
        for part in [ix[:10],ix[10:]]:
            np.testing.assert_array_equal(np.diff(part[:4]),1)
    def test_repeated_comparison_same_random_stream(self):
        rng=np.random.default_rng(4);a=rng.normal(.001,.01,60);b=a+rng.normal(0,.001,60);rf=np.zeros(60);pos=np.arange(60)
        x=es.paired_inference(a,b,rf,pos,100,5,42);es.paired_inference(a,b*.8,rf,pos,100,5,42);y=es.paired_inference(a,b,rf,pos,100,5,42)
        self.assertEqual(x,y)
    def test_identical_series_flagged_not_p_zero(self):
        a=np.linspace(-.01,.02,60);r=es.paired_inference(a,a,np.zeros(60),np.arange(60),100,5,42)
        self.assertEqual(r['inference_status'],'identical_series');self.assertTrue(np.isnan(r['sharpe_bootstrap_p']))
    def test_matching_centered_interval_and_test(self):
        draws=.5+np.random.default_rng(2).normal(0,.1,2000);r=es.interval_test(.5,draws)
        self.assertLess(r['p'],.05);self.assertGreater(r['low'],0)
        r=es.interval_test(.01,draws-.49);self.assertGreater(r['p'],.05);self.assertLess(r['low'],0);self.assertGreater(r['high'],0)
    def test_unique_hypotheses_not_double_corrected(self):
        f=pd.DataFrame(dict(purpose=['p']*3,scope=['s']*3,inference_key=['a','a','b'],p=[.01,.01,.04]))
        out=es.adjust_unique(f,['p']);np.testing.assert_allclose(out.p_holm,[.02,.02,.04])
    def test_holm_na_and_known_answer(self):np.testing.assert_allclose(es.holm([.01,.04,np.nan,.03]),[.03,.06,np.nan,.06],equal_nan=True)
    def test_sharpe_definition_matches_current_net_return_sd(self):
        r=np.array([-.1,.05,-.02,.03]);rf=np.array([.001,.002,.003,.004]);m=step4.metrics(r,rf)
        self.assertAlmostEqual(m['sharpe'],np.sqrt(252)*(r-rf).mean()/r.std(ddof=1))
    def test_drawdown_initial_nav_and_gaps(self):
        m=step4.metrics([-.1,.05,-.2,.1],[0]*4,[0,1,10,11]);self.assertAlmostEqual(m['max_drawdown'],.2)
    def test_fractional_es_hand_example(self):
        loss=np.arange(1,11.)
        self.assertAlmostEqual(es.upper_es(loss,.15),(10+.5*9)/1.5)
    def test_tail_integral_identity(self):
        loss=np.arange(20.)
        self.assertAlmostEqual(step4.tail_average(loss,.8,1),.25*step4.tail_average(loss,.95,1)+.75*step4.tail_average(loss,.8,.95))
    def test_state_contrast_hand_mean_difference(self):
        y=np.r_[np.arange(10)*.01,np.arange(10)*.01+.02];state=np.r_[np.zeros(10),np.ones(10)]
        result=es.window_state_contrast(y,state,np.arange(20));self.assertAlmostEqual(result['contrast'],.02)
    def test_state_insufficient_group_not_fake_zero(self):
        r=es.window_state_contrast(np.arange(12.),np.r_[np.zeros(11),1],np.arange(12));self.assertTrue(np.isnan(r['p']))
    def test_neutral_and_heterogeneity_pair_definitions(self):
        ids=[('K1_NEUTRAL','RVaR_neutral_k1'),('K1_NEUTRAL','CVaR_neutral_k1'),('K_HETEROG','RVaR_heterog_all_20'),('K_HETEROG','RVaR_heterog_control')]
        d=pd.DataFrame(ids,columns=['experiment_id','model_id']);p=step4.pair_definitions(d)
        self.assertIn('heterogeneous_vs_control',p.purpose.tolist());self.assertTrue(p.left_experiment.eq('K1_NEUTRAL').any())

if __name__=='__main__':unittest.main()
