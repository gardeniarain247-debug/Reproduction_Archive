"""Actual wealth recursion, missing data, EMEW and no-trade fallback."""
import unittest
from pathlib import Path
import tempfile,json
import numpy as np
import pandas as pd
import step2,step3

class Stage3Tests(unittest.TestCase):
    def context(self):
        dates=pd.bdate_range('2020-01-01',periods=3)
        return dates,pd.DataFrame({'a':[.1,-.05,.02],'b':[-.1,.05,0.]},index=dates),pd.Series(.001,index=dates)
    def test_buy_and_hold_matches_share_value(self):
        dates,r,rf=self.context();x=pd.Series({'a':.3,'b':.2});fee=.0015;cash=1-x.sum()-fee
        d,end,c=step3.advance_holdings(x,cash,r,rf,fee)
        expected=.3*np.prod(1+r.a)+.2*np.prod(1+r.b)+cash*np.prod(1+rf)
        self.assertAlmostEqual(np.prod(1+d.net_return),expected);self.assertAlmostEqual(end.sum()+c,1)
        self.assertAlmostEqual(d.trading_cost.sum(),fee)
    def test_price_drift_changes_next_initial_weights(self):
        dates,r,rf=self.context();d,end,c=step3.advance_holdings(pd.Series({'a':.5,'b':0.}),.5,r,rf)
        self.assertNotAlmostEqual(end.a,.5)
    def test_cash_fallback_earns_rf_not_zero(self):
        dates,r,rf=self.context();d,end,c=step3.advance_holdings(pd.Series(dtype=float),1.,r,rf)
        np.testing.assert_allclose(d.net_return,rf,rtol=0,atol=1e-12)
    def test_no_trade_hold_continues_actual_positions(self):
        dates,r,rf=self.context();first,prev,c=step3.advance_holdings(pd.Series({'a':.3,'b':.2}),.5,r.iloc[:1],rf)
        second,end,cash=step3.advance_holdings(prev,c,r.iloc[1:],rf)
        whole,ee,cc=step3.advance_holdings(pd.Series({'a':.3,'b':.2}),.5,r,rf)
        np.testing.assert_allclose(second.net_return,whole.net_return.iloc[1:]);self.assertAlmostEqual(end.a,ee.a)
    def test_missing_unheld_asset_does_not_poison(self):
        dates,r,rf=self.context();r['b']=np.nan
        d,_,_=step3.advance_holdings(pd.Series({'a':.5,'b':0}),.5,r,rf);self.assertTrue(np.isfinite(d.net_return).all())
    def test_missing_held_asset_aborts(self):
        dates,r,rf=self.context();r.loc[dates[1],'a']=np.nan
        with self.assertRaisesRegex(ValueError,'no zero imputation'):step3.advance_holdings(pd.Series({'a':.5}),.5,r,rf)
    def test_emew_uses_own_fees_and_exact_postfee_exposure(self):
        x,c,fee,ok=step3.equal_weight_execution(pd.Series({'a':.1}),['a','b'],pd.Series({'a':1,'b':1}),.9,step2.load_policy())
        self.assertTrue(ok);self.assertAlmostEqual(x.sum()/(1-fee),.9);self.assertAlmostEqual(x.sum()+c+fee,1)
        self.assertAlmostEqual(fee,.003*(x.subtract(pd.Series({'a':.1}),fill_value=0).abs().sum()))
    def test_emew_preserves_locked_holding(self):
        x,c,fee,ok=step3.equal_weight_execution(pd.Series({'a':.2}),['a','b'],pd.Series({'a':0,'b':1}),.6,step2.load_policy())
        self.assertTrue(ok);self.assertAlmostEqual(x.a,.2)
    def test_zero_position_unavailable_gets_no_equal_weight(self):
        x,c,fee,ok=step3.equal_weight_execution(pd.Series(dtype=float),['a','b'],pd.Series({'a':0,'b':1}),.5,step2.load_policy())
        self.assertAlmostEqual(x.a,0);self.assertGreater(x.b,0)
    def test_heterogeneity_current_pool_and_locked_exit(self):
        from test_support import make_inputs
        with tempfile.TemporaryDirectory() as td:
            r1,data,policy=make_inputs(td);contexts,_,signs=step2.load_inputs(r1,policy)
            ret=step3.step1.read_daily_matrix(r1/'full_return_matrix.csv',True);turn=step3.step1.read_daily_matrix(r1/'full_turnover_matrix.csv',True)
            context=contexts[1];previous=pd.Series({'600000':.2})
            specs=step2.experiment_specs(policy)
            for scope in ['partial','all']:
                spec=next(x for x in specs if x['model_id']=='RVaR_heterog_'+scope+'_20')
                codes,*rest=step3.coefficients(context,spec,previous,ret,turn,signs,policy)
                ki=rest[-1];j=codes.index('600000')
                expected=.8*(1+.2*signs['600000']) if scope=='all' else .8
                self.assertAlmostEqual(ki[j],expected)
    def test_hash_tamper_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            folder=Path(td);p=folder/'data.csv';p.write_text('original')
            step3.finish_metadata(folder,'manifest.json',dict(status='ok'),[p]);step3.checked_manifest(folder,'manifest.json','ok')
            p.write_text('changed')
            with self.assertRaises(ValueError):step3.checked_manifest(folder,'manifest.json','ok')
    def test_existing_outputs_not_overwritten(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/'step3_sentinel.csv';p.write_text('keep')
            with self.assertRaises(FileExistsError):step3.prepare_output(td,'step3_',False)
            self.assertEqual(p.read_text(),'keep')

if __name__=='__main__':unittest.main()
