import unittest
import pandas as pd
from execution_sensitivity import reconcile_execution_fills
P={'cost_buy':.003,'cost_sell':.003,'cash_lower':0.}
class SuspensionFillTests(unittest.TestCase):
 def test_suspended_sell_remains_held(self):
  old=pd.Series({'600030':.2,'A':.1}); target=pd.Series({'600030':0.,'A':.5})
  x,c,f,log=reconcile_execution_fills(old,target,pd.Series({'600030':0.,'A':1.}),P)
  self.assertAlmostEqual(x['600030'],.2)
  self.assertAlmostEqual(x['A'],.5)
  self.assertAlmostEqual(x.sum()+c+f,1)
  self.assertEqual(log[0]['fill_status'],'rejected_no_proxy_volume')
 def test_suspended_buy_is_rejected(self):
  old=pd.Series({'600030':0.,'A':.1}); target=pd.Series({'600030':.3,'A':.1})
  x,c,f,log=reconcile_execution_fills(old,target,pd.Series({'600030':float('nan'),'A':1.}),P)
  self.assertAlmostEqual(x['600030'],0.)
  self.assertAlmostEqual(f,0.)
 def test_rejected_sale_scales_buys_no_leverage(self):
  old=pd.Series({'600030':.8,'A':.2}); target=pd.Series({'600030':0.,'A':1.})
  x,c,f,log=reconcile_execution_fills(old,target,pd.Series({'600030':0.,'A':1.}),P)
  self.assertAlmostEqual(x['600030'],.8)
  self.assertGreaterEqual(c,-1e-9)
  self.assertAlmostEqual(x.sum()+c+f,1.)
  self.assertTrue(any(z['fill_status']=='partial_cash_capacity' for z in log))
if __name__=='__main__':unittest.main()
