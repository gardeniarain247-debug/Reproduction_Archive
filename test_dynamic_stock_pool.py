"""Regression: suspended old holdings survive changes in eligible stock pool."""
import unittest
import pandas as pd
from execution_sensitivity import reconcile_execution_fills

P={'cost_buy':.003,'cost_sell':.003,'cash_lower':0.}
class DynamicPoolTests(unittest.TestCase):
    def test_old_suspended_stock_is_preserved_and_aligned(self):
        old=pd.Series({'OLD':.2,'A':.1,'B':0.0})
        plan=pd.Series({'A':.3,'NEW':.2})
        vol=pd.Series({'OLD':float('nan'),'A':1.0,'NEW':1.0})
        actual,cash,fee,log=reconcile_execution_fills(old,plan,vol,P)
        old_keyed=old.reindex(actual.index,fill_value=0.)
        self.assertAlmostEqual(actual.loc['OLD'],.2)
        self.assertAlmostEqual(actual.sum()+cash+fee,1.,places=9)
        self.assertAlmostEqual(.5*(actual-old_keyed).abs().sum(),.2,places=9)
        self.assertTrue(any(e['stock_code']=='OLD' and e['fill_status']=='rejected_no_proxy_volume' for e in log))
if __name__=='__main__':unittest.main()
