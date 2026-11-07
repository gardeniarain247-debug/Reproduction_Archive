"""Model A algebra, scenario completeness and cost invariants."""
import unittest
import tempfile
from pathlib import Path
from unittest.mock import patch
import numpy as np
import step2


class Stage2Tests(unittest.TestCase):
    def setUp(self):self.p=step2.load_policy()
    def test_modelB_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            f=Path(td)/'bad.json';f.write_text('{"objective":"legacy_scalarized"}')
            with self.assertRaises(ValueError):step2.load_policy(f)
    def test_asymmetric_fee_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            f=Path(td)/'bad.json';f.write_text('{"cost_buy":0.0025}')
            with self.assertRaises(ValueError):step2.load_policy(f)
    def test_risk_tail_mapping(self):self.assertTrue(step2.risk_tail_mapping_check().passed.all())
    def test_common_object_identity_and_ordering(self):
        e=np.array([[-.1,-.02,.04,.2],[-.04,-.01,.03,.1]])
        for k in [.5,.8,1.,1.1]:
            mu,risks=step2.three_measure_coefficients(e,k)
            np.testing.assert_allclose(risks['CVaR20'],.25*risks['CVaR05']+.75*risks['RVaR'],atol=1e-12)
            self.assertTrue((risks['CVaR05']>=risks['CVaR20']-1e-12).all())
            self.assertTrue((risks['CVaR20']>=risks['RVaR']-1e-12).all())
    def test_interval_changes_risk_not_mean(self):
        e=np.array([[-.1,-.01,.02,.1]])
        m,r=step2.portfolio_coefficients(e,.8);mm,rr=step2.portfolio_coefficients(e,.8,interval=(.10,.20))
        np.testing.assert_array_equal(m,mm);self.assertNotEqual(r[0],rr[0])
    def test_assetwise_heterogeneous_coefficients(self):
        e=np.array([[-.1,-.02,.03,.1],[-.2,-.05,.1,.4]])
        mu,rho=step2.portfolio_coefficients(e,[.7,.9])
        for j,k in enumerate([.7,.9]):
            m,r=step2.portfolio_coefficients(e[j:j+1],k);self.assertAlmostEqual(mu[j],m[0]);self.assertAlmostEqual(rho[j],r[0])
    def test_partial_heterogeneity_only_chosen_assets(self):
        codes=[f'{j:06d}' for j in range(20)];spec=dict(k=.8,heterog_scope='partial',delta=.2);signs={c:1 if j%2 else -1 for j,c in enumerate(codes)}
        k=step2.heterogeneous_shapes(codes,spec,self.p,signs)
        self.assertEqual(np.count_nonzero(k!=.8),10);np.testing.assert_array_equal(k[10:],.8)
    def test_standard_has_all_required_regular_experiments(self):
        specs=step2.experiment_specs(self.p);selected=step2.select_experiments(specs,'standard')
        self.assertTrue({'K1_NEUTRAL','K_HETEROG','ENDPOINT_ALT','RVAR_INTERVAL','CVAR02_ROBUSTNESS','KPURE_CONTROL'}<=selected)
        self.assertNotIn('ENDPOINT_BOOT',selected)
        k1=[s for s in specs if s['experiment_id']=='K1_NEUTRAL'];self.assertEqual(len(k1),2)
        self.assertTrue(all(s['k']==1 and s['target_annual']==.08 and s['exposure_floor']==.5 for s in k1))
    def test_relaxation_scales_unified(self):
        self.assertTrue(all(s['scales']==[1,.95,.9,.85] for s in step2.experiment_specs(self.p)))
    def test_no_mean_reward_in_objective(self):
        args=([.01]*4,[.02]*4,.0001,0,[0.]*4,.5,self.p)
        a=step2.build_lp(*args)[0];b=step2.build_lp([.02]*4,*args[1:])[0]
        np.testing.assert_array_equal(a,b)
    def test_data_relocation_hash_validation(self):
        from test_support import make_inputs
        with tempfile.TemporaryDirectory() as td:
            r1,data,p=make_inputs(td);step2.load_inputs(r1,p)
            with (r1/'full_return_matrix.csv').open('a') as f:f.write('\n')
            with self.assertRaises(ValueError):step2.load_inputs(r1,p)

if __name__=='__main__':unittest.main()
