"""Execution invariants migrated from old correction module to current Model A."""
import unittest
from unittest.mock import patch
import numpy as np
from scipy.optimize import OptimizeResult
import step2

class ExecutionTests(unittest.TestCase):
    def setUp(self):self.p=step2.load_policy()
    def test_cash_fee_budget_and_fees_both_sides(self):
        old=np.array([.2,.2,0.,0.]);mu=np.full(4,.02)
        x,i=step2.solve_execution_modelA(mu,np.array([.08,.04,.02,.01]),.0001,0,old,.9,self.p)
        self.assertIsNotNone(x);fee=.003*(np.maximum(x-old,0).sum()+np.maximum(old-x,0).sum())
        self.assertAlmostEqual(i['fee'],fee,12);self.assertAlmostEqual(x.sum()+i['cash']+fee,1,12)
        self.assertGreaterEqual(i['return_slack'],-1e-7)
    def test_locked_asset_above_cap_and_forced_exit(self):
        old=np.array([.6,.1,0.,0.]);locks=np.array([.6,np.nan,np.nan,np.nan])
        x,i=step2.solve_execution_modelA(np.full(4,.02),np.full(4,.01),.0001,0,old,.5,self.p,locks,np.array([False,True,False,False]))
        self.assertIsNotNone(x);self.assertAlmostEqual(x[0],.6);self.assertAlmostEqual(x[1],0);self.assertGreaterEqual(i['fee'],.0003-1e-9)
    def test_zero_position_untradable_has_no_purchase(self):
        locks=np.array([0.,np.nan,np.nan,np.nan])
        x,i=step2.solve_execution_modelA(np.full(4,.02),np.array([-.1,0.,.01,.02]),0,0,np.zeros(4),.5,self.p,locks=locks)
        self.assertAlmostEqual(x[0],0)
    def test_infeasible_preserves_failure(self):
        x,i=step2.solve_execution_modelA(np.zeros(4),np.ones(4),0,100,np.zeros(4),.5,self.p)
        self.assertIsNone(x);self.assertEqual(i['status'],'Failed');self.assertEqual(len(i['attempts']),4)
    def test_numeric_error_is_not_target_relaxation(self):
        with patch.object(step2,'linprog',return_value=OptimizeResult(status=4,success=False,message='numerical')) as call:
            with self.assertRaises(RuntimeError):step2.solve_execution_modelA(np.ones(4),np.ones(4),0,.08,np.zeros(4),.5,self.p)
            self.assertEqual(call.call_count,1)
    def test_osqp_no_version_specific_polishing_keyword(self):
        import cvxpy as cp
        original=cp.Problem.solve
        def guarded(obj,*a,**kw):
            self.assertNotIn('polishing',kw);return original(obj,*a,**kw)
        with patch.object(cp.Problem,'solve',guarded):
            x,i=step2.solve_execution_modelA(np.full(4,.02),np.zeros(4),.0001,.08,np.zeros(4),.6,self.p,sigma=np.eye(4)*.01)
        self.assertIsNotNone(x);self.assertLessEqual(i['residual'],1e-7)
    def test_return_and_objective_reconstructed(self):
        mu=np.full(4,.02);rho=np.array([.01,.02,.03,.04]);rf=.0001
        x,i=step2.solve_execution_modelA(mu,rho,rf,.08,np.zeros(4),.5,self.p)
        self.assertAlmostEqual(i['expected_net_daily'],mu@x+rf*i['cash']-i['fee'])
        self.assertAlmostEqual(i['objective'],rho@x-rf*i['cash']+i['fee'])

if __name__=='__main__':unittest.main()
