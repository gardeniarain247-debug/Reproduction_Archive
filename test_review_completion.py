"""Independent checks of sample selection, delayed accounting and repair limits."""
import contextlib,io,json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import pandas as pd
import step2,step3,step4
import review_completion as rc
import execution_sensitivity as ex
import controlled_diagnostics as cd
import data_evidence_audit as da
from test_support import make_inputs


class CompletionTests(unittest.TestCase):
    def test_accepted_includes_relaxed_but_never_failed(self):
        z=pd.DataFrame(dict(solver_status_left=['Direct','Relaxed','Failed','Direct'],
            solver_status_right=['Direct','Direct','Direct','Benchmark'],
            execution_action_left=['optimized_rebalance']*2+['hold_after_infeasible','optimized_rebalance'],
            execution_action_right=['optimized_rebalance']*3+['hold_unmatched'],
            direct_eligible_left=[True,False,False,True],direct_eligible_right=[True,True,True,False]))
        masks=rc.sample_masks(z)
        self.assertEqual(list(masks['both_accepted']),[True,True,False,False])
        self.assertEqual(list(masks['both_direct_only']),[True,False,False,False])

    def test_delayed_new_buy_does_not_earn_first_day(self):
        dates=pd.bdate_range('2020-01-01',periods=2)
        ret=pd.DataFrame({'a':[.9,-.1]},index=dates);rf=pd.Series(0.,index=dates)
        old,prefix=ex.before_trade(pd.Series(dtype=float),dates,ret,rf,'next_close')
        d,end,cash=ex.after_trade(pd.Series({'a':.6}),.397,.003,dates,ret,rf,prefix)
        self.assertAlmostEqual(d.net_return.iloc[0],-.003)
        self.assertAlmostEqual(np.prod(1+d.net_return),.6*.9+.397)
        self.assertAlmostEqual(end.sum()+cash,1)

    def test_delayed_old_holdings_and_fee_use_execution_nav(self):
        dates=pd.bdate_range('2020-01-01',periods=2)
        ret=pd.DataFrame({'a':[.1,-.1]},index=dates);rf=pd.Series(0.,index=dates)
        old,prefix=ex.before_trade(pd.Series({'a':.5}),dates,ret,rf,'next_close')
        self.assertAlmostEqual(old.a,.55/1.05)
        d,_,_=ex.after_trade(pd.Series({'a':.6}),.397,.003,dates,ret,rf,prefix)
        self.assertAlmostEqual(np.prod(1+d.net_return),1.05*(.6*.9+.397))
        self.assertAlmostEqual(d.nav_relative.iloc[-1],np.prod(1+d.net_return))

    def test_proxy_rejection_aborts_instead_of_using_future_reoptimization(self):
        old=pd.Series({'a':0.0})
        planned=pd.Series({'a':0.5})
        unavailable=pd.Series({'a':0.0})
        target,cash,fee,ledger=ex.reconcile_execution_fills(
            old,planned,unavailable,step2.load_policy())
        self.assertAlmostEqual(target['a'],old['a'])
        self.assertAlmostEqual(cash,1.0)
        self.assertAlmostEqual(fee,0.0)
        self.assertEqual(len(ledger),1)
        self.assertEqual(ledger[0]['fill_status'],'rejected_no_proxy_volume')
        self.assertAlmostEqual(ledger[0]['executed_change'],0.0)

    def test_repair_respects_budget_and_floor_but_not_original_target(self):
        p=step2.load_policy();old=np.zeros(3)
        x,info=ex.repair_portfolio(old,.5,p,np.full(3,np.nan),np.zeros(3,bool))
        self.assertAlmostEqual(x.sum(),.5);self.assertTrue((x<=.3+1e-8).all())
        self.assertAlmostEqual(x.sum()+info['cash']+info['fee'],1)
        self.assertEqual(info['status'],'repaired_without_return_target')

    def test_repair_does_not_unlock_impossible_positions(self):
        p=step2.load_policy()
        x,info=ex.repair_portfolio(np.zeros(2),.5,p,np.zeros(2),np.zeros(2,bool))
        self.assertIsNone(x);self.assertEqual(info['status'],'repair_infeasible')

    def test_cap_audit_separates_pretrade_from_postfee_denominator(self):
        fee=.003*.9
        identity=dict(experiment_id='TEST',model_id='A',window_id=0)
        assets=pd.DataFrame([dict(identity,posttrade_amount=.3,posttrade_weight=.3/(1-fee),
            terminal_weight=.3/(1-fee),buy=.3,sell=0.,initial_weight=0.,locked=False) for _ in range(3)])
        w=pd.DataFrame([dict(identity,solver_status='Direct',optimized=True,risky_exposure=.9,
            cash_weight=.1-fee,trading_cost=fee,exposure_floor=.9)])
        daily=pd.DataFrame([dict(identity,risky_exposure=.9/(1-fee))])
        audit,summary=rc.constraint_audit(w,assets,daily,step2.load_policy())
        self.assertFalse(audit.cap_pretrade_breach.iloc[0]);self.assertTrue(audit.cap_postfee_breach.iloc[0])

    def test_closure_example_preserves_additivity(self):
        frame,meta=cd.closure_example()
        self.assertGreater(abs(frame.left_sum-frame.single_left).max(),1e-4)
        self.assertLess(meta['quantile_integral_additivity_error'],1e-10)

    def test_flat_windows_manifest_and_tamper(self):
        import hashlib
        with tempfile.TemporaryDirectory() as td:
            p=Path(td);f=p/'table.csv';f.write_text('one')
            m=dict(pipeline_model='ModelA',status='accepted_modelA_stage3',
                outputs=[dict(path=r'D:\old\table.csv',sha256=hashlib.sha256(f.read_bytes()).hexdigest())])
            (p/'step3_run_metadata.json').write_text(json.dumps(m))
            result=rc.verify_sources(p);self.assertTrue(result.status.eq('matched').any())
            f.write_text('two')
            with self.assertRaisesRegex(ValueError,'hash mismatch'):rc.verify_sources(p)

    def test_data_cache_reconciliation_and_no_false_event_certificate(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);r1,data,policy=make_inputs(root,1)
            ret=step3.step1.read_daily_matrix(r1/'full_return_matrix.csv',True)
            turn=step3.step1.read_daily_matrix(r1/'full_turnover_matrix.csv',True)
            cache=r1/'baostock_cache';cache.mkdir()
            for code in ret:
                dates=ret.index.insert(0,ret.index[0]-pd.offsets.BDay())
                prices=np.r_[100.,100*np.cumprod(1+ret[code].to_numpy())]
                pd.DataFrame(dict(date=dates,close=prices,turn=np.r_[1.,turn[code]])).to_csv(cache/(code+'.csv'),index=False)
            meta=da.run(r1,data,root/'audit')
            self.assertEqual(meta['cached_assets_matched'],len(ret.columns))
            self.assertFalse(meta['corporate_actions_independently_reconciled'])

    def test_delayed_end_to_end_and_failed_repair_stays_failed(self):
        with tempfile.TemporaryDirectory() as td,contextlib.redirect_stdout(io.StringIO()):
            root=Path(td);r1,data,policy=make_inputs(root,1)
            original=step2.experiment_specs
            def specs(p):
                base=original(p)
                return [base[0],base[1],dict(base[0],model_id='CVaR_forced_failure',target_annual=100.)]
            with patch.object(step2,'experiment_specs',side_effect=specs):
                step2.run_stage2(r1,root/'results2',root/'results2',policy,'standard')
                ex.run(r1,root/'results2',data,root/'delayed','next_close','repair',1)
            w=pd.read_csv(root/'delayed/step3_window_backtests.csv')
            f=w[w.model_id.eq('CVaR_forced_failure')].iloc[0]
            self.assertEqual(f.solver_status,'Failed');self.assertFalse(f.optimized)
            self.assertEqual(f.execution_action,'minimum_turnover_repair_execution_checked')
            step4.run(root/'delayed',root/'results4',replicates=100,allow_pilot=True,sensitivity=False)
            m=rc.run(root/'delayed',root/'completion',100,pilot=True)
            self.assertEqual(m['status'],'pilot_diagnostics_only')
            pairs=pd.read_csv(root/'completion/completion_pairs.csv')
            self.assertIn('both_accepted',pairs.scope.tolist())

    def test_finish_entry_runs_all_three_separate_execution_policies(self):
        import finish_all
        with tempfile.TemporaryDirectory() as td,contextlib.redirect_stdout(io.StringIO()):
            root=Path(td);r1,data,policy=make_inputs(root,1)
            original=step2.experiment_specs
            def specs(p):
                all_specs=original(p)
                return [s for s in all_specs if s['model_id'] in ['CVaR_opt','RVaR_opt',
                    'RVaR_neutral_k1','CVaR_neutral_k1','RVaR_heterog_control','CVaR_heterog_control']]
            with patch.object(step2,'experiment_specs',side_effect=specs):
                source=root/'original'
                step2.run_stage2(r1,source/'results2',source/'results2',policy,'standard')
                step3.run_stage3(r1,source/'results2',data,source/'results3')
                meta=finish_all.run(source,root/'finished',r1,data,with_execution=True,replicates=100,pilot=True)
            self.assertFalse(meta['all_reviewer_comments_certified'])
            comparison=pd.read_csv(root/'finished/execution_policy_comparison.csv')
            self.assertEqual(set(comparison.variant),{'formation_close_repair','next_close_hold','next_close_repair'})


if __name__=='__main__':unittest.main()
