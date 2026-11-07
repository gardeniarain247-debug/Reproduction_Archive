"""Mathematical and edge-case checks for the isolated calibration pilot."""
import unittest
import numpy as np

import ki_calibration as kc


class ShapeCalibrationTests(unittest.TestCase):
    def setUp(self):
        self.policy = kc.default_policy()
        self.e = np.array([-2., -.5, .5, 2.])
        self.spec = self.policy["variants"][0]

    def test_neutral_curve_hand_calculation(self):
        x = np.array([-3, -2, -1.25, -.5, 0, .5, 1.25, 2, 3])
        np.testing.assert_allclose(kc.membership(x, self.e, 1), [0, 0, .5, 1, 1, 1, .5, 0, 0])

    def test_reciprocal_exponents(self):
        x = np.array([-1.25, 1.25])
        np.testing.assert_allclose(kc.membership(x, self.e, 2), [np.sqrt(.5), .25])

    def test_acf_ties_and_median(self):
        curve = kc.construct_acf([0, 0, 0, 1, 2, 5])
        self.assertEqual(curve.core_probability, .5)
        self.assertAlmostEqual(float(curve(curve.core)), 1)
        self.assertTrue(np.all(np.diff(curve.probabilities) >= 0))
        self.assertEqual(float(curve(-1)), 0)
        self.assertEqual(float(curve(7)), 0)

    def test_mean_core_is_not_same_convention(self):
        values = [0, 1, 2, 3, 20]
        median, mean = kc.construct_acf(values), kc.construct_acf(values, "mean")
        self.assertNotAlmostEqual(mean.core_probability, median.core_probability)
        self.assertAlmostEqual(float(mean(mean.core)), 1)

    def test_midrank_curve_against_four_hand_values(self):
        curve = kc.construct_acf([0, 1, 2, 3])
        np.testing.assert_allclose(curve.cdf([0, 1, 1.5, 2, 3]), [.125, .375, .5, .625, .875])
        np.testing.assert_allclose(curve([0, 1, 1.5, 2, 3]), [.25, .75, 1, .75, .25])

    def test_infinite_returns_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "infinite"):
            kc.construct_acf([0, 1, np.inf])

    def test_affine_invariance_of_projection(self):
        values = np.linspace(-3, 3, 250) ** 3
        e = np.quantile(values, [.05, .25, .75, .95])
        a, _ = kc.analyse(values, e, self.spec, self.policy, free=False)
        b, _ = kc.analyse(4 * values + 10, 4 * e + 10, self.spec, self.policy, free=False)
        self.assertAlmostEqual(a["k"], b["k"], places=5)
        self.assertAlmostEqual(a["loss"], b["loss"], places=8)

    def test_constant_and_degenerate_not_imputed(self):
        with self.assertRaises(ValueError):
            kc.construct_acf(np.zeros(250))
        ans, obj = kc.analyse(np.zeros(250), np.zeros(4), self.spec, self.policy)
        self.assertFalse(ans["eligible"])
        self.assertTrue(np.isnan(ans["k"]))
        self.assertIsNone(obj)

    def test_mean_inverse_and_outside_cases(self):
        for true in [.1, .5, 1, 2, 10]:
            a, b, c, d = self.e
            m = (true * (a + c) + b + d) / (2 * (true + 1))
            result = kc.mean_match(self.e, m)
            self.assertAlmostEqual(result["k_mean"], true)
        result = kc.mean_match(self.e, 2)
        self.assertTrue(np.isnan(result["k_mean"]))
        self.assertEqual(result["mean_status"], "no_positive_finite_solution")

    def test_near_mean_boundary_flag(self):
        lower, upper = (self.e[0] + self.e[2]) / 2, (self.e[1] + self.e[3]) / 2
        result = kc.mean_match(self.e, lower + 1e-4 * (upper - lower))
        self.assertEqual(result["mean_status"], "near_boundary")
        self.assertGreater(result["k_mean"], 1000)

    def test_global_search_finds_two_equal_minima(self):
        result = kc.fit_scalar(lambda th: (np.asarray(th) ** 2 - 1) ** 2, self.policy)
        self.assertEqual(result["status"], "multiple_minima")
        self.assertEqual(result["minima"], 2)
        self.assertAlmostEqual(abs(result["theta"]), 1, places=5)

    def test_flat_objective_is_not_identified(self):
        result = kc.fit_scalar(lambda th: np.zeros_like(th) + .25, self.policy)
        self.assertEqual(result["status"], "flat_objective")
        self.assertFalse(result["eligible"])
        self.assertTrue(np.isnan(result["k"]))

    def test_boundary_solution_kept_and_flagged(self):
        result = kc.fit_scalar(lambda th: -np.asarray(th), self.policy)
        self.assertEqual(result["status"], "boundary")
        self.assertFalse(result["eligible"])
        self.assertAlmostEqual(result["k"], 20)

    def test_weak_profile_is_flagged(self):
        result = kc.fit_scalar(lambda th: 1e-6 * np.asarray(th) ** 2, self.policy)
        self.assertEqual(result["status"], "weak_profile")
        self.assertFalse(result["eligible"])

    def test_incompatible_free_shape_has_nonzero_restriction_cost(self):
        target = lambda x: kc.membership(x, self.e, powers=(2, 2))
        objective = kc.ShapeObjective(self.e, target)
        constrained = kc.fit_scalar(objective, self.policy)
        p = kc.fit_scalar(objective.exponent_objective("left"), self.policy)
        q = kc.fit_scalar(objective.exponent_objective("right"), self.policy)
        self.assertAlmostEqual(p["k"], 2, places=5)
        self.assertAlmostEqual(q["k"], 2, places=5)
        self.assertGreater(constrained["loss"], p["loss"] + q["loss"] + 1e-3)

    def test_core_residual_cannot_be_removed_by_k(self):
        target = lambda x: np.zeros_like(x) + .5
        obj = kc.ShapeObjective(self.e, target)
        self.assertAlmostEqual(obj.core_loss, .25 * 1 / 4)


if __name__ == "__main__":
    unittest.main()
