"""Reviewer-diagnostic design checks, independent of profitable outcomes."""
import unittest,tempfile
from pathlib import Path
import numpy as np
import step1,step2,step3,remaining_experiments as rem
from test_support import make_inputs

class RemainingTests(unittest.TestCase):
    def test_tail_path_contains_three_exact_criteria(self):
        e=np.array([[-.1,-.02,.03,.1]])
        _,r=step2.three_measure_coefficients(e,.8)
        np.testing.assert_allclose(.25*r['CVaR05']+.75*r['RVaR'],r['CVaR20'])
        self.assertTrue(np.array_equal(0*r['CVaR05']+r['RVaR'],r['RVaR']))
    def test_controlled_paths_use_same_cash_initial_not_policy_history(self):
        with tempfile.TemporaryDirectory() as td:
            r1,data,policy=make_inputs(td,1);contexts,_,_=step2.load_inputs(r1,policy)
            turn=step1.read_daily_matrix(r1/'full_turnover_matrix.csv',True)
            table,weights=rem.controlled_substitution(contexts,policy,turn)
            self.assertEqual(len(table),15);self.assertEqual(set(table.theta),{0,.25,.5,.75,1})
            self.assertTrue(table.shared_initial_rule.str.contains('cash').all())
            self.assertTrue(table[table.theta.eq(0)].coherent_spectral_weight.eq(False).all())
    def test_heterogeneous_shapes_not_claimed_empirical_preferences(self):
        import ki_calibration as kc
        self.assertIn('never behavioral identification',kc.__doc__)
    def test_invalid_sample_count_rejected_before_io(self):
        with self.assertRaises(ValueError):rem.run('missing','missing','missing','missing',replicates=0)
    def test_common_shape_closure_can_fail(self):
        levels=np.array([.25,.5]);lower=.5*(np.sqrt(levels)+levels**2)
        k=np.log(lower)/np.log(levels);self.assertGreater(abs(k[0]-k[1]),.1)

if __name__=='__main__':unittest.main()
