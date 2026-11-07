"""Focused stage-1 regression checks: python -m unittest test_step1 -v."""
import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

import step1


class Stage1Tests(unittest.TestCase):
    def small_case(self):
        settings = step1.load_settings()
        settings.update(estimation_days=4, holding_days=3, step_days=3,
                        expected_windows=1, min_observations=3,
                        bootstrap_block_lengths=[2], bootstrap_replicates=5)
        dates = pd.bdate_range("2020-01-01", periods=7)
        returns = pd.DataFrame({"600000": [0.01, 0.02, np.nan, -0.01, 0.2, -0.3, 0.4]}, index=dates)
        members = pd.DataFrame({"stock_code": ["600000", "600001"],
                                "period_start": [dates[0], dates[0]],
                                "period_end": [dates[-1], dates[-1]]})
        windows = step1.build_windows(dates, 4, 3)
        return settings, returns, members, windows

    def test_import_does_not_read_write_or_load_network(self):
        spec = importlib.util.spec_from_file_location("step1_safe_import", step1.__file__)
        with patch.object(pd, "read_csv", side_effect=AssertionError("I/O during import")), \
             patch.object(pd, "read_excel", side_effect=AssertionError("I/O during import")), \
             patch.object(Path, "mkdir", side_effect=AssertionError("mkdir during import")):
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
        self.assertTrue(callable(module.main))

    def test_complete_windows_and_no_future_overlap(self):
        dates = pd.bdate_range("2015-01-01", periods=2673)
        windows = step1.build_windows(dates)
        self.assertEqual(len(windows), 40)
        self.assertEqual(windows.iloc[0].est_end, dates[249])
        self.assertEqual(windows.iloc[0].test_start, dates[250])
        self.assertEqual(windows.iloc[1].est_start, dates[60])
        self.assertTrue((windows.est_end < windows.test_start).all())
        with self.assertRaises(ValueError):
            step1.build_windows(dates.append(dates[:1]))

    def test_calibration_is_independent_of_holding_returns(self):
        settings, returns, members, windows = self.small_case()
        original = step1.calibrate_windows(returns, windows, members, settings)
        changed = returns.copy()
        changed.loc[windows.iloc[0].test_start:] = np.nan
        revised = step1.calibrate_windows(changed, windows, members, settings)
        pd.testing.assert_frame_equal(original[0], revised[0])
        pd.testing.assert_frame_equal(original[1], revised[1])
        self.assertEqual(original[2].reason.tolist(), ["included", "missing_return_column"])
        self.assertEqual(original[1].groupby("endpoint_scheme").size().tolist(), [1, 1, 1])
        self.assertEqual(original[1].obs.tolist(), [3, 3, 3])

    def test_quantiles_and_degeneracy_are_not_silently_adjusted(self):
        result = step1.fuzzy_params([0.0, 1.0, np.nan, 2.0, 3.0])
        np.testing.assert_allclose([result[k] for k in "abcd"], [0.15, 0.75, 2.25, 2.85])
        self.assertEqual(result["obs"], 4)
        self.assertEqual(step1.endpoint_status(step1.fuzzy_params([1.0] * 10)), "degenerate")
        with self.assertRaises(ValueError):
            step1.fuzzy_params([0, np.inf])
        with self.assertRaises(ValueError):
            step1.fuzzy_params([0, 1], [0.05, 0.75, 0.25, 0.95])

    def test_bootstrap_uses_contiguous_training_blocks_shared_across_assets(self):
        settings, returns, members, windows = self.small_case()
        first = step1.bootstrap_window_indices(windows, returns, settings)
        second = step1.bootstrap_window_indices(windows, returns, settings)
        indices = first["indices_win0_block2"]
        np.testing.assert_array_equal(indices, second["indices_win0_block2"])
        self.assertEqual(indices.shape, (5, 4))
        self.assertGreaterEqual(indices.min(), 0)
        self.assertLess(indices.max(), 4)
        np.testing.assert_array_equal(indices[:, 1::2] - indices[:, ::2], 1)
        self.assertLess(first["dates_win0"].max(), windows.iloc[0].test_start.to_datetime64())
        vector = np.column_stack([np.arange(4), 10 + 2 * np.arange(4)])
        sample = vector[indices]
        np.testing.assert_array_equal(sample[:, :, 1], 10 + 2 * sample[:, :, 0])

    def test_bootstrap_handles_partial_final_block(self):
        indices = step1.moving_block_indices(7, 3, 8, np.random.default_rng(42))
        self.assertEqual(indices.shape, (8, 7))
        self.assertLess(indices.max(), 7)
        np.testing.assert_array_equal(np.diff(indices[:, :3]), np.ones((8, 2)))

    def test_causal_risk_free_does_not_use_future_observations(self):
        dates = pd.date_range("2020-01-01", periods=4)
        bond = pd.Series([0.1, 0.3], index=dates[[1, 3]])
        legacy = step1.align_risk_free(bond, dates, "legacy")
        causal = step1.align_risk_free(bond, dates)
        self.assertTrue(legacy.iloc[0].used_future_value)
        self.assertTrue(pd.isna(causal.iloc[0].rf_daily))
        self.assertEqual(causal.iloc[2].rf_daily, 0.1)
        self.assertEqual(causal.iloc[2].source_date, dates[1])
        # A source date omitted from the target axis still supplies an as-of value.
        sparse = step1.align_risk_free(bond, dates[[2]])
        self.assertEqual(sparse.iloc[0].rf_daily, 0.1)

    def test_offline_missing_cache_never_downloads(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(FileNotFoundError):
                step1.download_one_stock("600000", directory)

    def test_no_overwrite_and_build_cannot_target_baseline(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            existing = folder / "windows_master_40.csv"
            existing.write_text("preserve", encoding="utf-8")
            with self.assertRaises(FileExistsError):
                step1.run_stage1(folder, folder, folder, step1.load_settings())
            with self.assertRaises(ValueError):
                step1.run_stage1(folder, folder, folder, step1.load_settings(), mode="build", overwrite=True)
            self.assertEqual(existing.read_text(), "preserve")
            self.assertEqual(list(folder.iterdir()), [existing])

    def test_configuration_validation(self):
        settings = step1.load_settings()
        settings["step_days"] = 1
        with self.assertRaises(ValueError):
            step1.validate_settings(settings)
        settings = step1.load_settings()
        settings["bootstrap_block_lengths"] = [251]
        with self.assertRaises(ValueError):
            step1.validate_settings(settings)

    def test_offline_build_writes_compatible_outputs_and_failed_checks_block_core(self):
        settings = step1.load_settings()
        settings.update(estimation_days=4, holding_days=3, step_days=3,
                        expected_windows=1, min_observations=3,
                        bootstrap_block_lengths=[2], bootstrap_replicates=5)
        dates = pd.bdate_range("2020-01-01", periods=8)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data, baseline, cache = root / "data", root / "baseline", root / "cache"
            for folder in [data, baseline, cache]:
                folder.mkdir()
            pd.DataFrame({"period_start": [20190101], "period_end": [20201231],
                          "stock_code": ["600000"]}).to_excel(data / settings["constituent_file"], index=False)
            pd.DataFrame({"date": dates, "returns": [2.0] * 8}).to_csv(data / settings["bond_file"], index=False)
            pd.DataFrame({"date": dates, "close": [10, 11, 10, 12, 11, 13, 12, 14],
                          "turn": [1.0] * 8}).to_csv(cache / "600000.csv", index=False)
            output = root / "build"
            metadata = step1.run_stage1(data, baseline, output, settings, mode="build",
                                       cache_dir=cache, generate_bootstrap=False)
            self.assertNotIn("failed", metadata["audit_counts"])
            self.assertTrue(all((output / name).is_file() for name in step1.CORE_FILES))
            self.assertEqual(len(pd.read_csv(output / "rolling_windows.csv")), 1)
            self.assertEqual(pd.read_csv(output / "fuzzy_parameters.csv").columns.tolist(),
                             ["window_id", "stock_code", "a", "b", "c", "d", "mean", "obs"])
            self.assertEqual(len(pd.read_csv(output / "run_registry.csv")), 1)
            self.assertEqual(list(baseline.iterdir()), [])
            settings["expected_windows"] = 2
            failed_output = root / "failed"
            metadata = step1.run_stage1(data, baseline, failed_output, settings, mode="build",
                                       cache_dir=cache, generate_bootstrap=False)
            self.assertEqual(metadata["status"], "failed_checks")
            self.assertTrue((failed_output / "input_data_audit.csv").is_file())
            self.assertFalse(any((failed_output / name).exists() for name in step1.CORE_FILES))


if __name__ == "__main__":
    unittest.main()
