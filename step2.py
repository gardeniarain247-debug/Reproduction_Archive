"""Stage 2: Model A minimum-downside optimization and revision experiments.

Place beside the project's step1.py; inputs default to results1.
Default core: main scenarios, CVaR20 substitution, and fixed-rule k controls.
--experiments standard: all regular revision experiments (no bootstrap).
--experiments all: also run endpoint bootstrap using Stage1 bootstrap indices.
Both purchases and sales cost 0.30%; Model B scalarization is not supported.
Failures remain NaN/ineligible. These are optimization trajectories, not an
execution backtest: Stage3 must reconcile actual holdings and failed windows.
"""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import platform
import sys
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import scipy
from scipy.optimize import linprog
import step1

BASE_DIR = Path(__file__).resolve().parent


def _module_version(name):
    """Return a module's __version__ without failing when the distribution lacks dist-info."""
    try:
        module = __import__(name)
    except ImportError:
        return "not_installed"
    return str(getattr(module, "__version__", "unknown"))


ALPHA, BETA = 0.80, 0.95

P_CVAR = 0.95

P_CVAR02 = 0.80

TRADING_DAYS = 252

COST_BUY = 0.0030

COST_SELL = 0.0030

WEIGHT_UPPER = 0.30

RISKY_SUM_LOWER_BY_MODEL = {
    "MV": 0.60,
    "CVaR_opt": 0.90,
    "CVaR_neu": 0.50,
    "CVaR_pes": 0.10,
    "RVaR_opt": 0.90,
    "RVaR_neu": 0.50,
    "RVaR_pes": 0.10,
    "CVaR02_k05": 0.90,
    "CVaR02_k08": 0.50,
    "CVaR02_k11": 0.10,
    "CVaR_kpure05": 0.50,
    "RVaR_kpure05": 0.50,
    "CVaR_kpure08": 0.50,
    "RVaR_kpure08": 0.50,
    "CVaR_kpure11": 0.50,
    "RVaR_kpure11": 0.50,
}

RISK_FREE_LOWER = 0.00

RISKY_SUM_UPPER = 1.0 - RISK_FREE_LOWER

# Main scenarios follow the revision document:
# k = 0.5 / 0.8 / 1.1 with annual targets 12% / 8% / 5%
# and risky floors 90% / 50% / 10%.
ATTITUDES = {
    "CVaR_opt": {"k": 0.5, "target_annual": 0.12, "attitude_label": "optimistic"},
    "RVaR_opt": {"k": 0.5, "target_annual": 0.12, "attitude_label": "optimistic"},
    "CVaR_neu": {"k": 0.8, "target_annual": 0.08, "attitude_label": "mildly_optimistic"},
    "RVaR_neu": {"k": 0.8, "target_annual": 0.08, "attitude_label": "mildly_optimistic"},
    "CVaR_pes": {"k": 1.1, "target_annual": 0.05, "attitude_label": "mildly_conservative"},
    "RVaR_pes": {"k": 1.1, "target_annual": 0.05, "attitude_label": "mildly_conservative"},
    "CVaR02_k05": {"k": 0.5, "target_annual": 0.12, "attitude_label": "optimistic", "cvar_p": P_CVAR02},
    "CVaR02_k08": {"k": 0.8, "target_annual": 0.08, "attitude_label": "mildly_optimistic", "cvar_p": P_CVAR02},
    "CVaR02_k11": {"k": 1.1, "target_annual": 0.05, "attitude_label": "mildly_conservative", "cvar_p": P_CVAR02},
    "CVaR_kpure05": {"k": 0.5, "target_annual": 0.08, "attitude_label": "low_k"},
    "RVaR_kpure05": {"k": 0.5, "target_annual": 0.08, "attitude_label": "low_k"},
    "CVaR_kpure08": {"k": 0.8, "target_annual": 0.08, "attitude_label": "middle_k"},
    "RVaR_kpure08": {"k": 0.8, "target_annual": 0.08, "attitude_label": "middle_k"},
    "CVaR_kpure11": {"k": 1.1, "target_annual": 0.08, "attitude_label": "high_k"},
    "RVaR_kpure11": {"k": 1.1, "target_annual": 0.08, "attitude_label": "high_k"},
}

MAIN_MODEL_LIST = ["MV", "CVaR_opt", "CVaR_neu", "CVaR_pes", "RVaR_opt", "RVaR_neu", "RVaR_pes"]

CVaR02_ROBUSTNESS_MODEL_LIST = ["CVaR02_k05", "CVaR02_k08", "CVaR02_k11"]

KPURE_MODEL_LIST = [
    "CVaR_kpure05",
    "RVaR_kpure05",
    "CVaR_kpure08",
    "RVaR_kpure08",
    "CVaR_kpure11",
    "RVaR_kpure11",
]

# Stable historical aliases are documented, but no duplicate strategies are solved.
SUBSTITUTION_MODEL_ALIASES = {
    "CVaR_k05": "CVaR_opt", "RVaR_k05": "RVaR_opt",
    "CVaR_k08": "CVaR_neu", "RVaR_k08": "RVaR_neu",
    "CVaR_k11": "CVaR_pes", "RVaR_k11": "RVaR_pes",
}
SUBSTITUTION_MODEL_LIST = list(SUBSTITUTION_MODEL_ALIASES)
MODEL_LIST = MAIN_MODEL_LIST + CVaR02_ROBUSTNESS_MODEL_LIST + KPURE_MODEL_LIST

MV_K = 1.0

MV_TARGET_ANNUAL = 0.08

def annual_to_daily(annual_rate):
    """Convert an annual decimal return target to a daily decimal return."""
    return annual_rate / TRADING_DAYS

def ret_to_loss(a, b, c, d):
    """
    Convert fuzzy return parameters to fuzzy loss parameters.

    If R = (a, b, c, d)_k, then L = -R has support/core parameters
    (-d, -c, -b, -a). Because left and right branches are exchanged,
    its attitude parameter must be changed from k to 1/k when computing
    loss-side VaR, CVaR, or RVaR.
    """
    return -d, -c, -b, -a

def fuzzy_expectation(a, b, c, d, k=1.0):
    """Compute the fuzzy expected return under return-side attitude parameter k."""
    return (k * a + b + k * c + d) / (2 * (k + 1))

def fuzzy_cvar_loss(la, lb, lc, ld, k_loss, p=P_CVAR):
    """
    Compute CVaR_p(L) for a fuzzy loss variable L=(la,lb,lc,ld)_{k_loss}.
    """
    if 0 < p <= 0.5:
        t1 = la
        t2 = (2 * p) ** k_loss * (lb - la)
        t3 = (lb - la) * (1 - (2 * p) ** k_loss) / (1 - p)
        t4 = ((lc - lb) / 2 + (ld - lc) / (2 * (k_loss + 1))) / (1 - p)
        t5 = (
            k_loss
            * (lb - la)
            * (1 - (2 * p) ** (k_loss + 1))
            / (2 * (1 - p) * (k_loss + 1))
        )
        return t1 + t2 + t3 + t4 - t5

    if 0.5 < p < 1:
        t1 = ld
        t2 = -(ld - lc) * (2 * (1 - p)) ** (1 / k_loss)
        t3 = (
            (ld - lc)
            * (2 * (1 - p)) ** ((k_loss + 1) / k_loss)
            / (2 * (1 - p) * (k_loss + 1))
        )
        return t1 + t2 + t3

    return np.nan

def fuzzy_rvar_loss(la, lb, lc, ld, k_loss, alpha=ALPHA, beta=BETA):
    """
    Compute RVaR_{alpha,beta}(L) for a fuzzy loss variable L=(la,lb,lc,ld)_{k_loss}.

    The right-tail case 0.5 < alpha < beta < 1 must use a minus sign:
        d - [k(d-c)2^{1/k}/(k+1)] * average((1-p)^{1/k}).
    """
    if 0 < alpha < beta <= 0.5:
        t1 = la
        t2 = (lb - la) * (2**k_loss) / (k_loss + 1)
        t3 = (beta ** (k_loss + 1) - alpha ** (k_loss + 1)) / (beta - alpha)
        return t1 + t2 * t3

    if 0.5 < alpha < beta < 1:
        t1 = ld
        t2 = (ld - lc) * k_loss * (2 ** (1 / k_loss)) / (k_loss + 1)
        t3 = ((1 - alpha) ** (1 + 1 / k_loss) - (1 - beta) ** (1 + 1 / k_loss)) / (beta - alpha)
        return t1 - t2 * t3

    if 0 < alpha <= 0.5 < beta < 1:
        p1 = (k_loss * la + lb) / (2 * (k_loss + 1))
        p2 = -la * alpha
        p3 = -(lb - la) * (2**k_loss) / (k_loss + 1) * (alpha ** (k_loss + 1))
        p4 = ld * beta
        p5 = (
            (ld - lc)
            * k_loss
            * (2 ** (1 / k_loss))
            / (k_loss + 1)
            * ((1 - beta) ** (1 + 1 / k_loss))
        )
        p6 = -((2 * k_loss + 1) * ld - k_loss * lc) / (2 * (k_loss + 1))
        return (p1 + p2 + p3 + p4 + p5 + p6) / (beta - alpha)

    return np.nan

def single_sided_turnover(current, previous):
    """Compute single-sided turnover after aligning current and previous weights by stock code."""
    if previous is None:
        return 0.5 * current.sum()
    all_codes = previous.index.union(current.index)
    diff = current.reindex(all_codes, fill_value=0.0) - previous.reindex(all_codes, fill_value=0.0)
    return 0.5 * np.abs(diff).sum()


# Policies are values in the run configuration, never implicit k branches.
# The empirical model is the minimum-downside LP; the scalarized objective
# of the earlier revision is not retained.
DEFAULT_POLICY = {
    "objective": "pure_downside",
    "cost_buy": COST_BUY, "cost_sell": COST_SELL, "weight_upper": WEIGHT_UPPER,
    "cash_lower": RISK_FREE_LOWER, "trading_days": TRADING_DAYS,
    "revision_relaxation_scales": [1.0, 0.95, 0.9, 0.85],
    "solver_method": "highs",
    "solver_options": {"primal_feasibility_tolerance": 1e-7,
                       "dual_feasibility_tolerance": 1e-7, "ipm_optimality_tolerance": 1e-8},
    "constraint_tolerance": 1e-7, "baseline_weight_tolerance": 1e-6,
    "initial_weight_rule": "cash_first_then_own_last_success_current_pool",
    "exit_asset_cost": "include_forced_liquidations",
    "degenerate_endpoint_policy": "preserve_algebraic_limit_and_flag",
    "mv_reference_policy": "recompute_mv_as_separate_benchmark",
    "heterogeneous_k0": 0.8, "heterogeneous_deltas": [0.1, 0.2],
    "heterogeneous_partial_assets": 10, "bootstrap_block_lengths": [5, 10],
    "bootstrap_replicates": 300, "bootstrap_seed": 42,
    "expected_windows": 40,
}


def load_policy(path=None):
    policy = json.loads(json.dumps(DEFAULT_POLICY))
    if path:
        incoming = json.loads(Path(path).read_text(encoding="utf-8-sig"))
        incoming = incoming.get("policy", incoming)
        unknown = set(incoming) - set(policy)
        if unknown:
            raise ValueError(f"Unknown policy keys: {sorted(unknown)}")
        policy.update(incoming)
    if policy["objective"] != "pure_downside":
        raise ValueError(
            "The adopted empirical model is minimum-downside risk minimization; "
            "the risk--mean scalarized objective is not retained"
        )
    for name in ["weight_upper", "constraint_tolerance", "baseline_weight_tolerance"]:
        if not np.isfinite(policy[name]) or policy[name] <= 0:
            raise ValueError(f"{name} must be positive")
    if not 0 <= policy["cash_lower"] < 1 or not 0 < policy["weight_upper"] <= 1:
        raise ValueError("Invalid cash or box bounds")
    if any(not np.isfinite(policy[n]) or policy[n] < 0 for n in ["cost_buy", "cost_sell"]):
        raise ValueError("Transaction costs must be nonnegative")
    if not (np.isclose(policy["cost_buy"], 0.003, rtol=0, atol=1e-12)
            and np.isclose(policy["cost_sell"], 0.003, rtol=0, atol=1e-12)):
        raise ValueError("Original Model A requires 0.30% for BOTH purchases and sales")
    scales = policy["revision_relaxation_scales"]
    if not scales or scales[0] != 1.0 or any(not 0 < x <= 1 for x in scales) or any(a <= b for a, b in zip(scales, scales[1:])):
        raise ValueError("Relaxation scales must begin at 1 and strictly decrease")
    for name in ["trading_days", "heterogeneous_partial_assets", "bootstrap_replicates", "expected_windows"]:
        if isinstance(policy[name], bool) or not isinstance(policy[name], int) or policy[name] < 1:
            raise ValueError(f"{name} must be a positive integer")
    if policy["initial_weight_rule"] != DEFAULT_POLICY["initial_weight_rule"] or policy["exit_asset_cost"] != DEFAULT_POLICY["exit_asset_cost"]:
        raise ValueError("Historical position/cost policy changes require a separate audited implementation")
    if policy["degenerate_endpoint_policy"] != DEFAULT_POLICY["degenerate_endpoint_policy"]:
        raise ValueError("No silent exclusion or jitter of degenerate endpoints")
    if policy["mv_reference_policy"] != DEFAULT_POLICY["mv_reference_policy"]:
        raise ValueError("MV baseline changes require a separate reviewed run")
    if not np.isfinite(policy["heterogeneous_k0"]) or policy["heterogeneous_k0"] <= 0:
        raise ValueError("heterogeneous_k0 must be positive")
    if any(not 0 < d < 1 for d in policy["heterogeneous_deltas"]):
        raise ValueError("Heterogeneous deltas must be inside (0, 1)")
    if policy["bootstrap_seed"] != 42:
        raise ValueError("Use the frozen stage1 bootstrap seed (42)")
    if not policy["bootstrap_block_lengths"] or len(set(policy["bootstrap_block_lengths"])) != len(policy["bootstrap_block_lengths"]):
        raise ValueError("Bootstrap block lengths must be nonempty and unique")
    if any(n not in {5, 10} for n in policy["bootstrap_block_lengths"]) or policy["bootstrap_replicates"] > 300:
        raise ValueError("Bootstrap configuration exceeds frozen stage1 inputs")
    return policy


def loss_attitude_parameter(k_return):
    k = np.asarray(k_return, dtype=float)
    if not np.all(np.isfinite(k)) or np.any(k <= 0):
        raise ValueError("Return-side shape parameters must be finite and positive")
    return 1.0 / k


def portfolio_coefficients(endpoints, k_return, measure="RVaR", interval=(0.05, 0.20), gamma=0.05):
    """Compute separate asset-level coefficients, including heterogeneous ki."""
    endpoints = np.asarray(endpoints, dtype=float)
    if endpoints.ndim != 2 or endpoints.shape[1] != 4 or not np.isfinite(endpoints).all():
        raise ValueError("Endpoints must be a finite n-by-4 array")
    if np.any(np.diff(endpoints, axis=1) < 0):
        raise ValueError("Endpoints are not ordered")
    k = np.broadcast_to(np.asarray(k_return, dtype=float), (len(endpoints),))
    k_loss = loss_attitude_parameter(k)
    a, b, c, d = endpoints.T
    mu = fuzzy_expectation(a, b, c, d, k)
    if measure == "CVaR":
        if not 0 < gamma <= 0.5:
            raise ValueError("Invalid return-side CVaR tail")
        rho = fuzzy_cvar_loss(-d, -c, -b, -a, k_loss, p=1 - gamma)
    elif measure == "RVaR":
        alpha, beta = interval
        if not 0 < alpha < beta <= 0.5:
            raise ValueError("Invalid return-side RVaR interval")
        rho = fuzzy_rvar_loss(-d, -c, -b, -a, k_loss, alpha=1 - beta, beta=1 - alpha)
    elif measure == "MV":
        rho = np.zeros(len(mu))
    else:
        raise ValueError(f"Unknown risk measure: {measure}")
    return mu, rho


def build_lp(mu, rho, rf_daily, target_daily, x0, exposure_floor, policy, *, exit_cost=0.0, locked_weights=None):
    """3n variables: risky weights, purchases and sales; original cost units."""
    if policy["objective"] != "pure_downside":
        raise ValueError("build_lp only supports the minimum-downside objective")
    mu, rho, x0 = [np.asarray(v, dtype=float) for v in (mu, rho, x0)]
    n = len(mu)
    if n == 0 or rho.shape != mu.shape or x0.shape != mu.shape or not all(np.isfinite(v).all() for v in (mu, rho, x0)):
        raise ValueError("Invalid LP coefficient/initial-weight vectors")
    if not np.isfinite(rf_daily) or not np.isfinite(target_daily) or target_daily < 0:
        raise ValueError("Invalid interest rate/target")
    if np.any(x0 < 0) or not 0 <= exposure_floor <= 1 - policy["cash_lower"]:
        raise ValueError("Invalid initial weights/exposure floor")
    # Risk-minimization objective: sum_i x_i rho_i - x_f r_f + C(x).
    # With x_f = 1 - sum_i x_i this is sum_i (rho_i + r_f) x_i + C(x) - r_f,
    # where the constant -r_f is dropped from the LP objective.
    risk_part = rho + rf_daily
    objective = np.r_[risk_part, np.full(n, policy["cost_buy"]), np.full(n, policy["cost_sell"])]
    matrix = np.zeros((2 * n + 3, 3 * n))
    identity = np.eye(n)
    matrix[:2*n:2] = np.c_[identity, -identity, identity]
    matrix[1:2*n:2] = -matrix[:2*n:2]
    rhs = np.r_[np.column_stack([x0, -x0]).ravel(), -(target_daily - rf_daily),
                1 - policy["cash_lower"], -exposure_floor]
    matrix[2*n] = np.r_[-(mu - rf_daily), np.full(n, policy["cost_buy"]), np.full(n, policy["cost_sell"])]
    matrix[2*n+1, :n] = 1
    matrix[2*n+2, :n] = -1
    bounds = [(0, policy["weight_upper"])] * n + [(0, None)] * (2*n)
    if not np.isfinite(exit_cost) or exit_cost < 0:
        raise ValueError("Invalid executable exit cost")
    rhs[2*n] -= exit_cost
    if locked_weights is not None:
        locked = np.asarray(locked_weights,dtype=float)
        if locked.shape != (n,):
            raise ValueError("Invalid locked weights")
        for j in np.flatnonzero(np.isfinite(locked)):
            if locked[j] < 0 or locked[j] > policy["weight_upper"] or not np.isclose(locked[j],x0[j],rtol=0,atol=1e-12):
                raise ValueError("Locked weights must equal previous holdings within box limits")
            bounds[j] = (float(locked[j]),float(locked[j]))
            bounds[n+j] = bounds[2*n+j] = (0.0,0.0)
    return objective, matrix, rhs, bounds


def solve_fuzzy_lp(mu_k, rho_k, rf_daily, target_annual, x0, n=None,
                    k_attitude=None, risky_sum_lower=0.5, *, policy=None, scales=None, exit_cost=0.0, locked_weights=None):
    """Solve with explicit scales; preserve failures and each attempted target."""
    policy = load_policy() if policy is None else policy
    if n is not None and n != len(mu_k):
        raise ValueError("n disagrees with coefficients")
    scales = policy["revision_relaxation_scales"] if scales is None else scales
    target_daily = target_annual / policy["trading_days"]
    objective, matrix, rhs, bounds = build_lp(mu_k, rho_k, rf_daily, target_daily, x0, risky_sum_lower, policy,
                                            exit_cost=exit_cost,locked_weights=locked_weights)
    n = len(mu_k)
    attempts = []
    info = {"status": "Failed", "initial_success": False, "relaxed": False,
            "final_success": False, "fallback_used": False, "relaxation_scale": np.nan,
            "objective_value": np.nan, "return_constraint_slack": np.nan,
            "risky_lower_slack": np.nan, "risky_upper_slack": np.nan,
            "max_constraint_violation": np.nan, "failure_reason": "all_targets_failed",
            "n_assets": n, "n_variables": 3*n, "n_inequalities": 2*n+3,
            "n_equalities": 0, "n_finite_lower_bounds": 3*n,
            "n_finite_upper_bounds":sum(upper is not None for lower,upper in bounds)}
    for level, scale in enumerate(scales):
        adjusted = rhs.copy()
        adjusted[2*n] = -(target_daily * scale - rf_daily + exit_cost)
        start = time.perf_counter()
        result = linprog(objective, A_ub=matrix, b_ub=adjusted, bounds=bounds,
                         method=policy["solver_method"], options=policy["solver_options"])
        elapsed = time.perf_counter() - start
        violation = np.nan
        if result.success:
            candidate = result.x.copy()
            # Only remove numerical boundary noise; then recheck the returned solution.
            candidate[:n] = np.clip(candidate[:n], 0, policy["weight_upper"])
            violation = max(0.0, float(np.max(matrix @ candidate - adjusted)),
                            float(np.max(-candidate)), float(np.max(candidate[:n] - policy["weight_upper"])))
            for j,(lower,upper) in enumerate(bounds):
                violation = max(violation,lower-candidate[j],candidate[j]-upper if upper is not None else 0.0)
        accepted = bool(result.success and violation <= policy["constraint_tolerance"])
        attempts.append({"attempt": level, "scale": scale, "target_annual": target_annual * scale,
                         "solver_code": result.status, "solver_success": bool(result.success),
                         "accepted": accepted, "solver_time_sec": elapsed,
                         "max_constraint_violation": violation, "message": result.message})
        if not accepted:
            info["failure_reason"] = "constraint_residual" if result.success else result.message
            # Only infeasibility permits lowering the target; not time/numerical errors.
            if result.status != 2:
                break
            continue
        weights = candidate[:n]
        slacks = adjusted - matrix @ candidate
        # objective @ x = sum_i (rho_i + r_f) x_i + C(x); subtract r_f to recover
        # the actual risk-minimization value sum_i rho_i x_i - x_f r_f + C(x).
        info.update(status="Direct" if level == 0 else "Relaxed", initial_success=level == 0,
                    relaxed=level > 0, final_success=True, relaxation_scale=scale,
                    objective_value=float(objective @ candidate) - rf_daily + exit_cost,
                    return_constraint_slack=float(slacks[2*n]),
                    risky_lower_slack=float(slacks[-1]), risky_upper_slack=float(slacks[-2]),
                    max_constraint_violation=violation, failure_reason="", relax_level=level)
        info["attempts"] = attempts
        info["solver_time_sec"] = sum(a["solver_time_sec"] for a in attempts)
        return weights, True, info
    info.update(attempts=attempts, solver_time_sec=sum(a["solver_time_sec"] for a in attempts), relax_level=len(attempts)-1)
    return np.full(n, np.nan), False, info


def solve_mv(mu, sigma, rf_daily, x0, target_annual, floor, policy, *, exit_cost=0.0):
    """Historical MV audit only; OSQP remains separate from LP diagnostics."""
    import cvxpy as cp
    n = len(mu)
    x, buy, sell = cp.Variable(n), cp.Variable(n), cp.Variable(n)
    objective = cp.Minimize(cp.quad_form(x, sigma) + policy["cost_buy"] * cp.sum(buy) + policy["cost_sell"] * cp.sum(sell) + exit_cost)
    attempts = []
    for level, scale in enumerate([1.0, 0.95, 0.9, 0.85, 0.8, 0.7]):
        constraints = [x >= 0, x <= policy["weight_upper"], cp.sum(x) >= floor,
                       cp.sum(x) <= 1-policy["cash_lower"], x-x0 == buy-sell, buy >= 0, sell >= 0,
                       cp.sum(cp.multiply(mu-rf_daily, x))-policy["cost_buy"]*cp.sum(buy)-policy["cost_sell"]*cp.sum(sell)
                       >= target_annual/policy["trading_days"]*scale-rf_daily+exit_cost]
        start = time.perf_counter()
        problem = cp.Problem(objective, constraints)
        try:
            problem.solve(solver=cp.OSQP, verbose=False)
            ok = problem.status in {cp.OPTIMAL,cp.OPTIMAL_INACCURATE} and x.value is not None and np.isfinite(x.value).all()
            message = str(problem.status)
        except cp.error.SolverError as exc:
            ok, message = False, str(exc)
        elapsed = time.perf_counter()-start
        attempts.append({"attempt": level, "scale": scale, "target_annual": target_annual*scale,
                         "solver_code": 0 if ok else 2, "solver_success": ok, "accepted": ok,
                         "solver_time_sec": elapsed, "max_constraint_violation": np.nan, "message": message})
        if ok:
            weights = np.clip(x.value, 0, policy["weight_upper"])
            # Audit the returned (including clipped) weights, not only solver presence.
            realized_cost = policy["cost_buy"]*np.maximum(weights-x0,0).sum()+policy["cost_sell"]*np.maximum(x0-weights,0).sum()+exit_cost
            return_slack = float((mu-rf_daily)@weights-realized_cost-(target_annual/policy["trading_days"]*scale-rf_daily))
            residual = max(0.0,-return_slack,floor-float(weights.sum()),float(weights.sum())-(1-policy["cash_lower"]))
            attempts[-1]["max_constraint_violation"] = residual
            # This diagnostic uses the CVXPY OSQP interface's default 1e-5 tolerance.
            # Numerical residual failure must not be treated as infeasibility relaxation.
            if residual > 1e-5:
                attempts[-1]["accepted"] = False
                break
            return weights, True, {"status": "Direct" if level == 0 else "Relaxed", "relaxation_scale": scale,
                                  "objective_value": float(problem.value), "failure_reason": "", "relax_level": level,
                                   "max_constraint_violation":residual,"return_constraint_slack":return_slack,
                                  "attempts": attempts, "solver_time_sec": sum(t["solver_time_sec"] for t in attempts)}
        if message not in {cp.INFEASIBLE,cp.INFEASIBLE_INACCURATE}:
            break
    return np.full(n, np.nan), False, {"status": "Failed", "relaxation_scale": np.nan, "objective_value": np.nan,
                                     "failure_reason": message, "relax_level": level, "attempts": attempts,
                                     "solver_time_sec": sum(t["solver_time_sec"] for t in attempts)}


def experiment_specs(policy):
    """Each specification has its own rolling position history."""
    specs = []

    def add(experiment, name, measure, k, target, floor, **extra):
        specs.append(dict(experiment_id=experiment, model_id=name, risk_measure=measure, k=k,
                          target_annual=target, exposure_floor=floor, endpoint_scheme="baseline",
                          interval=[0.05, 0.20], gamma=0.05,
                          scales=policy["revision_relaxation_scales"], **extra))

    for name, original in ATTITUDES.items():
        measure = "CVaR" if "CVaR" in name else "RVaR"
        if name in MAIN_MODEL_LIST:
            experiment = "BASELINE_AUDIT"
        elif name in CVaR02_ROBUSTNESS_MODEL_LIST:
            experiment = "CVAR02_ROBUSTNESS"
        elif name in KPURE_MODEL_LIST:
            experiment = "KPURE_CONTROL"
        else:
            raise ValueError(f"Unclassified Model A specification: {name}")
        specs.append(dict(experiment_id=experiment, model_id=name,
                          risk_measure=measure, k=original["k"],
                          target_annual=original["target_annual"],
                          exposure_floor=RISKY_SUM_LOWER_BY_MODEL[name],
                          endpoint_scheme="baseline", interval=[0.05, 0.20],
                          gamma=1-original.get("cvar_p", 0.95),
                          scales=policy["revision_relaxation_scales"]))
    add("BASELINE_AUDIT", "MV", "MV", MV_K, MV_TARGET_ANNUAL,
        RISKY_SUM_LOWER_BY_MODEL["MV"])
    for k, suffix in [(0.5, "05"), (0.8, "08"), (1.1, "11")]:
        for measure in ["RVaR", "CVaR"]:
            add("FIXED_RULE_CONTROL", f"{measure}_fixed_k{suffix}", measure, k, 0.08, 0.5)
            add("ENDPOINT_BOOT_REFERENCE", f"{measure}_boot_k{suffix}", measure, k,
                {0.5: 0.12, 0.8: 0.08, 1.1: 0.05}[k], {0.5: 0.9, 0.8: 0.5, 1.1: 0.1}[k])
    for measure in ["RVaR", "CVaR"]:
        add("K1_NEUTRAL", f"{measure}_neutral_k1", measure, 1.0, 0.08, 0.5)
        add("K_HETEROG", f"{measure}_heterog_control", measure, policy["heterogeneous_k0"], 0.08, 0.5,
            heterog_scope="none", delta=0.0)
        for scope in ["partial", "all"]:
            for delta in policy["heterogeneous_deltas"]:
                add("K_HETEROG", f"{measure}_heterog_{scope}_{int(delta*100)}", measure,
                    policy["heterogeneous_k0"], 0.08, 0.5, heterog_scope=scope, delta=delta)
    for k, suffix, target, floor in [(0.5, "05", 0.12, 0.9), (0.8, "08", 0.08, 0.5), (1.1, "11", 0.05, 0.1)]:
        for measure in ["RVaR", "CVaR"]:
            for scheme in step1.ENDPOINT_SCHEMES:
                add("ENDPOINT_ALT", f"{measure}_endpoint_{scheme}_k{suffix}", measure, k, target, floor)
                specs[-1]["endpoint_scheme"] = scheme
        for label, interval in [("baseline", [0.05, 0.20]), ("a", [0.05, 0.15]), ("b", [0.10, 0.20])]:
            add("RVAR_INTERVAL", f"RVaR_interval_{label}_k{suffix}", "RVaR", k, target, floor)
            specs[-1]["interval"] = interval
        add("RVAR_INTERVAL", f"CVaR_interval_control_k{suffix}", "CVaR", k, target, floor)
    return specs


def heterogeneous_shapes(stocks, spec, policy, global_signs):
    k = np.full(len(stocks), spec["k"], dtype=float)
    scope = spec.get("heterog_scope", "none")
    chosen = set(sorted(stocks)[:policy["heterogeneous_partial_assets"]]) if scope == "partial" else set(stocks)
    if scope != "none":
        for position, code in enumerate(stocks):
            if code in chosen:
                k[position] *= 1 + spec["delta"] * global_signs[code]
    return k


class ParquetSink:
    """Write bounded batches to one compressed file; avoid per-window files."""
    def __init__(self, path):
        import pyarrow.parquet as pq
        self.path, self.writer, self.pq = Path(path), None, pq

    def write(self, frame):
        import pyarrow as pa
        if frame.empty:
            return
        # Stable schema even when a whole batch failed or consists of one risk type.
        table = pa.Table.from_pandas(frame, preserve_index=False)
        if self.writer is None:
            self.writer = self.pq.ParquetWriter(self.path, table.schema, compression="zstd")
        self.writer.write_table(table.cast(self.writer.schema))

    def close(self):
        if self.writer is not None:
            self.writer.close()
            self.writer = None


def load_inputs(input_dir, policy):
    input_dir = Path(input_dir)
    metadata = json.loads((input_dir / "step1_run_metadata.json").read_text(encoding="utf-8"))
    if metadata["status"] != "audit_pass_with_review_items" or metadata["audit_counts"].get("failed", 0):
        raise ValueError("Stage1 integrity audit must pass first")
    # The copied results1 folder is portable. Verify the current local generated
    # files by SHA256, never reopen an old project's absolute path. External raw
    # inputs are represented by their stored provenance only.
    for item in metadata.get("outputs", []):
        local_path = input_dir / Path(item["path"]).name
        if not local_path.is_file() or step1.sha256_file(local_path) != item["sha256"]:
            raise ValueError(f"Stage1 output missing/hash mismatch after relocation: {local_path}")
    returns = step1.read_daily_matrix(input_dir / "full_return_matrix.csv", True)
    windows = step1.read_result_table(input_dir / "rolling_windows.csv")
    for name in step1.WINDOW_COLUMNS[1:]:
        windows[name] = pd.to_datetime(windows[name])
    if len(windows) != policy["expected_windows"] or windows.window_id.duplicated().any():
        raise ValueError("Unexpected or duplicate rolling windows")
    universe = step1.read_result_table(input_dir / "windows_universe.csv", True)
    if universe.duplicated(["window_id", "stock_code"]).any():
        raise ValueError("Duplicate universe entries")
    endpoints = step1.read_result_table(input_dir / "endpoint_calibration.csv", True)
    old_fuzzy = step1.read_result_table(input_dir / "fuzzy_parameters.csv", True)
    baseline = endpoints.endpoint_scheme.eq("baseline")
    # Exact historical floating-point inputs are retained for reproducibility.
    for column in list("abcd"):
        values = old_fuzzy.set_index(["window_id", "stock_code"])[column]
        index = pd.MultiIndex.from_frame(endpoints.loc[baseline, ["window_id", "stock_code"]])
        endpoints.loc[baseline, column] = values.reindex(index).to_numpy()
    rf = step1.read_daily_matrix(input_dir / "risk_free_rate.csv").iloc[:, 0]
    if not windows.est_end.isin(rf.index).all():
        raise ValueError("Missing decision-date risk-free rate; no full-sample mean fallback")
    prepared = {}
    for win in windows.itertuples(index=False):
        stocks = universe.loc[universe.window_id.eq(win.window_id), "stock_code"].tolist()
        training = returns.loc[win.est_start:win.est_end, stocks]
        if len(training) != 250 or not len(stocks):
            raise ValueError(f"Invalid frozen estimation window: {win.window_id}")
        schemes = {}
        for scheme in step1.ENDPOINT_SCHEMES:
            selected = endpoints.loc[endpoints.window_id.eq(win.window_id) & endpoints.endpoint_scheme.eq(scheme)]
            if selected.stock_code.duplicated().any():
                raise ValueError("Duplicate calibration entries")
            schemes[scheme] = selected.set_index("stock_code").loc[stocks, list("abcd")].to_numpy(dtype=float)
        prepared[win.window_id] = dict(stocks=stocks, training=training, schemes=schemes,
                                      rf_daily=float(rf.loc[win.est_end]), win=win)
    signs = {code: -1 if i % 2 == 0 else 1 for i, code in enumerate(sorted(universe.stock_code.unique()))}
    return prepared, metadata, signs


def baseline_comparison(window_id, spec, stocks, weights, ok, baseline_dir, policy):
    """No cross-model weight comparison: Model B is excluded by design."""
    return {"window_id": window_id, "model_id": spec["model_id"],
            "k": spec["k"], "risk_measure": spec["risk_measure"],
            "gate_required": False, "source_file": "",
            "status_match": False, "max_weight_difference": np.nan,
            "matches": False, "reason": "model_A_fresh_solution_no_legacy_weight_comparison"}


def solve_records(context, spec, policy, previous, signs, endpoints=None, replicate=-1, block=0, reference=None):
    """Return stable window/asset/attempt tables and the accepted portfolio."""
    start = time.perf_counter()
    stocks, win = context["stocks"], context["win"]
    n = len(stocks)
    endpoints = context["schemes"][spec["endpoint_scheme"]] if endpoints is None else endpoints
    x0 = np.zeros(n) if previous is None else previous.reindex(stocks, fill_value=0).to_numpy(dtype=float)
    ki = heterogeneous_shapes(stocks, spec, policy, signs)
    coefficient_start = time.perf_counter()
    coefficient_error = ""
    try:
        mu, rho = portfolio_coefficients(endpoints, ki, spec["risk_measure"], spec["interval"], spec["gamma"])
    except ValueError as exc:
        if replicate < 0:
            raise
        mu, rho = np.full(n, np.nan), np.full(n, np.nan)
        coefficient_error = str(exc)
    coefficient_seconds = time.perf_counter()-coefficient_start
    local = dict(policy)  # The same Model A cost/constraint configuration applies to ALL experiments.
    dropped = 0.0 if previous is None else float(previous.loc[~previous.index.isin(stocks)].sum())
    exit_cost = dropped * local["cost_sell"]
    if coefficient_error:
        weights, ok = np.full(n, np.nan), False
        info = dict(status="Failed", relaxation_scale=np.nan, objective_value=np.nan,
                    failure_reason="bootstrap_calibration: " + coefficient_error, relax_level=-1,
                    attempts=[], solver_time_sec=0.0)
    elif spec["risk_measure"] == "MV":
        sigma = context["training"].cov().to_numpy() + np.eye(n)*1e-8
        weights, ok, info = solve_mv(mu, sigma, context["rf_daily"], x0, spec["target_annual"], spec["exposure_floor"], local, exit_cost=exit_cost)
    else:
        weights, ok, info = solve_fuzzy_lp(mu, rho, context["rf_daily"], spec["target_annual"], x0,
                                         risky_sum_lower=spec["exposure_floor"], policy=local, scales=spec["scales"], exit_cost=exit_cost)
    exposure = float(weights.sum()) if ok else np.nan
    normalized = weights/exposure if ok and exposure > 0 else np.full(n, np.nan)
    hhi = float(normalized @ normalized) if ok and exposure > 0 else np.nan
    trade_cost = float(local["cost_buy"]*np.maximum(weights-x0, 0).sum()+local["cost_sell"]*np.maximum(x0-weights, 0).sum()+exit_cost) if ok else np.nan
    reference_l1 = float(np.sum(np.abs(weights-reference))) if ok and reference is not None and np.isfinite(reference).all() else np.nan
    identity = dict(experiment_id=spec["experiment_id"], model_id=spec["model_id"], window_id=int(win.window_id),
                    replicate=int(replicate), block_length=int(block))
    row = dict(identity, risk_measure=spec["risk_measure"], k_setting=float(spec["k"]),
               backtest_eligible=bool(ok),
               endpoint_scheme=spec["endpoint_scheme"], return_alpha=float(spec["interval"][0]),
               return_beta=float(spec["interval"][1]), cvar_return_gamma=float(spec["gamma"]),
               train_start=win.est_start, train_end=win.est_end, hold_start=win.test_start, hold_end=win.test_end,
               objective_policy=local["objective"], target_annual=float(spec["target_annual"]),
               exposure_floor=float(spec["exposure_floor"]), rf_daily=context["rf_daily"], n_assets=n,
               solver_status=info["status"], relaxation_scale=float(info["relaxation_scale"]),
               relax_level=int(info["relax_level"]), objective_value=float(info["objective_value"]),
               solver_time_sec=float(info["solver_time_sec"]), coefficient_time_sec=coefficient_seconds,
               n_variables=3*n, n_inequalities=3 if spec["risk_measure"] == "MV" else 2*n+3,
               n_equalities=n if spec["risk_measure"] == "MV" else 0, n_finite_lower_bounds=3*n,
               n_finite_upper_bounds=n, risky_exposure=exposure, cash_weight=1-exposure,
               hhi_risky=hhi, effective_n=1/hhi if hhi > 0 else np.nan,
               max_normalized_weight=float(np.nanmax(normalized)) if ok and exposure > 0 else np.nan,
               reference_weight_l1=reference_l1, mu_mean=float(mu.mean()), rho_mean=float(rho.mean()),
               initial_weight_sum=float(x0.sum()), dropped_previous_exposure=dropped,
               lp_cost=trade_cost, omitted_exit_cost=0.0, forced_exit_cost=exit_cost,
               max_constraint_violation=float(info.get("max_constraint_violation", np.nan)),
               return_constraint_slack=float(info.get("return_constraint_slack", np.nan)),
               failure_reason=info["failure_reason"], degenerate_endpoints=int(np.sum((endpoints[:,0]>=endpoints[:,1]) | (endpoints[:,2]>=endpoints[:,3]))),
               window_time_sec=time.perf_counter()-start)
    assets = pd.DataFrame({**{k: [v]*n for k, v in identity.items()}, "stock_code": stocks,
                           "backtest_eligible": bool(ok),
                           "k_return": ki, "k_loss": 1/ki, "a": endpoints[:,0], "b": endpoints[:,1],
                           "c": endpoints[:,2], "d": endpoints[:,3], "mu": mu, "rho": rho,
                           "initial_weight": x0, "weight": weights})
    attempts = [dict(identity, risk_measure=spec["risk_measure"], solver_method="OSQP" if spec["risk_measure"] == "MV" else local["solver_method"], **attempt)
                for attempt in info["attempts"]]
    current = pd.Series(weights, index=stocks) if ok else previous
    return row, assets, attempts, current, weights, ok



# Same-portfolio three-measure diagnostic. Each measure is evaluated on the SAME
# fuzzy return and the SAME candidate allocation. This is distinct from a
# realized-out-of-sample comparison (which belongs to stages 3 and 4).
SUBSTITUTION_TRIOS = (
    (0.5, "05", "CVaR_opt", "RVaR_opt", "CVaR02_k05"),
    (0.8, "08", "CVaR_neu", "RVaR_neu", "CVaR02_k08"),
    (1.1, "11", "CVaR_pes", "RVaR_pes", "CVaR02_k11"),
)


def three_measure_coefficients(endpoints, k_return):
    """Return coefficients for CVaR(.05), RVaR(.05,.20), CVaR(.20).

    The identity is pointwise and only applies to a common fuzzy return:
    CVaR(.20) = .25 CVaR(.05) + .75 RVaR(.05,.20).
    """
    mu, c05 = portfolio_coefficients(endpoints, k_return, "CVaR", gamma=0.05)
    mu_r, rvar = portfolio_coefficients(endpoints, k_return, "RVaR", interval=(0.05, 0.20))
    mu_b, c20 = portfolio_coefficients(endpoints, k_return, "CVaR", gamma=0.20)
    if not np.allclose(mu, mu_r, atol=1e-13, rtol=0) or not np.allclose(mu, mu_b, atol=1e-13, rtol=0):
        raise ValueError("Substitution diagnostic: means differ under identical k")
    if not np.allclose(c20, 0.25*c05 + 0.75*rvar, atol=1e-12, rtol=0):
        raise ValueError("Substitution diagnostic: quantile-weight identity failed")
    if np.any(c05 < c20-1e-12) or np.any(c20 < rvar-1e-12):
        raise ValueError("Substitution diagnostic: tail-risk ordering failed")
    return mu, {"CVaR05": c05, "RVaR": rvar, "CVaR20": c20}


def substitution_diagnostics(windows, assets, policy):
    """Cross-evaluate all three *optimized* portfolios with all three risks.

    Returns only optimization-stage model-implied coefficients, trading costs and
    allocation differences. Do NOT interpret these as actual daily returns.
    """
    if not (np.isclose(policy["cost_buy"], 0.003, rtol=0, atol=1e-12)
            and np.isclose(policy["cost_sell"], 0.003, rtol=0, atol=1e-12)):
        raise ValueError("Substitution diagnostic requires 0.30% on both purchases and sales")
    output, pairs = [], []
    if windows.empty or assets.empty:
        return pd.DataFrame(), pd.DataFrame()
    for k, suffix, c05_id, rv_id, c20_id in SUBSTITUTION_TRIOS:
        model_ids = (c05_id, rv_id, c20_id)
        for window_id in sorted(windows.window_id.unique()):
            w = windows.loc[(windows.window_id == window_id) &
                            (windows.experiment_id.isin(["BASELINE_AUDIT", "CVAR02_ROBUSTNESS"])) &
                            (windows.model_id.isin(model_ids))]
            if set(w.model_id) != set(model_ids) or len(w) != 3:
                continue  # Partial experiments do not silently create incomplete results.
            weight_maps = {}
            for model_id in model_ids:
                state = w.loc[w.model_id == model_id].iloc[0]
                sample = assets.loc[(assets.window_id == window_id) & (assets.model_id == model_id) &
                                    (assets.experiment_id == state.experiment_id)]
                if len(sample) != int(state.n_assets):
                    raise ValueError(f"Incomplete asset rows for {model_id} in window {window_id}")
                weights = sample.set_index("stock_code").weight.astype(float)
                weight_maps[model_id] = weights if state.solver_status != "Failed" else None
                # All the three optimized strategies must be calibrated using the same endpoints.
                mu, risk_values = three_measure_coefficients(sample[list("abcd")].to_numpy(float), k)
                if not np.allclose(mu, sample.mu.to_numpy(float), atol=1e-12, rtol=0):
                    raise ValueError(f"Coefficient disagreement: {model_id} win {window_id}")
                if state.solver_status == "Failed":
                    risks = {key: np.nan for key in risk_values}
                    expected_net = np.nan
                else:
                    x = weights.to_numpy(float)
                    cash = 1.0 - x.sum()
                    # Same portfolio, same fees, same cash return; only the risk metric changes.
                    risks = {key: float(x @ val - cash*state.rf_daily + state.lp_cost)
                             for key, val in risk_values.items()}
                    expected_net = float(x @ mu + cash*state.rf_daily - state.lp_cost)
                    if not np.isclose(risks["CVaR20"], 0.25*risks["CVaR05"]+0.75*risks["RVaR"], atol=1e-11, rtol=0):
                        raise ValueError("Portfolio-level three-measure identity failed")
                output.append(dict(window_id=int(window_id), k=float(k), model_id=model_id,
                                   experiment_id=state.experiment_id, solver_status=state.solver_status,
                                   **{f"cross_risk_{key}": value for key, value in risks.items()},
                                   model_expected_net_return=expected_net, risky_exposure=state.risky_exposure,
                                   cash_weight=state.cash_weight, trading_cost=state.lp_cost,
                                   theoretical_identity_checked=state.solver_status != "Failed",
                                   diagnostic_scope="in_sample_model_implied_not_realized_performance"))
            for left_id, right_id, label in ((rv_id,c05_id,"RVaR_vs_CVaR05"),
                                             (rv_id,c20_id,"RVaR_vs_CVaR20"),
                                             (c20_id,c05_id,"CVaR20_vs_CVaR05")):
                left, right = weight_maps[left_id], weight_maps[right_id]
                if left is None or right is None:
                    l1 = np.nan
                    common = False
                else:
                    index = left.index.union(right.index)
                    l1 = float(np.abs(left.reindex(index,fill_value=0)-right.reindex(index,fill_value=0)).sum())
                    common = True
                pairs.append(dict(window_id=int(window_id), k=float(k), comparison=label,
                                  left_model_id=left_id, right_model_id=right_id,
                                  both_solver_accepted=common, weight_l1_distance=l1,
                                  diagnostic_scope="in_sample_allocations_not_out_of_sample_inference"))
    return pd.DataFrame(output), pd.DataFrame(pairs)

def risk_tail_mapping_check():
    rows = []
    endpoints = np.array([[-0.06, -0.015, 0.01, 0.05]])
    a, b, c, d = endpoints[0]
    for k in [0.5, 0.8, 1.0, 1.1]:
        for alpha, beta in [(0.05, 0.20), (0.05, 0.15), (0.10, 0.20)]:
            omega = 2**k*(beta**(k+1)-alpha**(k+1))/((k+1)*(beta-alpha))
            direct = -(1-omega)*a-omega*b
            mu, rho = portfolio_coefficients(endpoints, k, interval=(alpha, beta))
            rows.append(dict(k_return=k, k_loss=1/k, measure="RVaR", return_alpha=alpha,
                             return_beta=beta, loss_alpha=1-beta, loss_beta=1-alpha,
                             return_formula_value=direct, code_loss_value=float(rho[0]),
                             absolute_difference=abs(direct-rho[0]),
                             source_function="portfolio_coefficients -> fuzzy_rvar_loss",
                             mathematical_definition="-average return quantile over [alpha,beta]"))
        for gamma in [0.05, 0.20]:
            omega = (2*gamma)**k/(k+1)
            direct = -(1-omega)*a-omega*b
            mu, rho = portfolio_coefficients(endpoints, k, "CVaR", gamma=gamma)
            rows.append(dict(k_return=k, k_loss=1/k, measure="CVaR", return_alpha=0.0,
                             return_beta=gamma, loss_alpha=1-gamma, loss_beta=1.0,
                             return_formula_value=direct, code_loss_value=float(rho[0]),
                             absolute_difference=abs(direct-rho[0]),
                             source_function="portfolio_coefficients -> fuzzy_cvar_loss",
                             mathematical_definition="-average return quantile over (0,gamma]"))
    frame = pd.DataFrame(rows)
    frame["passed"] = frame.absolute_difference <= 1e-12
    return frame


def run_bootstrap(prepared, reference_specs, policy, signs, initial_states, reference_weights,
                  input_dir, sinks, log):
    total = 0
    asset_buffer, window_buffer, attempt_buffer = [], [], []

    def flush():
        if window_buffer:
            sinks["windows"].write(pd.DataFrame(window_buffer))
            sinks["assets"].write(pd.concat(asset_buffer, ignore_index=True))
            if attempt_buffer:
                sinks["attempts"].write(pd.DataFrame(attempt_buffer))
            asset_buffer.clear()
            window_buffer.clear()
            attempt_buffer.clear()

    stage1_meta = json.loads((Path(input_dir)/"step1_run_metadata.json").read_text(encoding="utf-8"))
    frozen = stage1_meta["settings"]
    if policy["bootstrap_seed"] != frozen["random_seed"] or policy["bootstrap_replicates"] > frozen["bootstrap_replicates"]:
        raise ValueError("Bootstrap policy disagrees with frozen stage1 indices")
    with np.load(Path(input_dir)/"bootstrap_indices.npz", allow_pickle=False) as arrays:
        for window_id, context in prepared.items():
            training = context["training"].to_numpy(dtype=float)
            np.testing.assert_array_equal(arrays[f"dates_win{window_id}"], context["training"].index.to_numpy(dtype="datetime64[ns]"))
            for block in policy["bootstrap_block_lengths"]:
                indices = arrays[f"indices_win{window_id}_block{block}"][:policy["bootstrap_replicates"]]
                if indices.shape != (policy["bootstrap_replicates"], len(training)) or indices.min() < 0 or indices.max() >= len(training):
                    raise ValueError("Invalid frozen bootstrap indices")
                for replicate, sample_indices in enumerate(indices):
                    calibration_start = time.perf_counter()
                    sampled = training[sample_indices]
                    # Date vectors are synchronized; missing values remain per asset.
                    endpoints = np.nanquantile(sampled, step1.ENDPOINT_SCHEMES["baseline"], axis=0, method="linear").T
                    endpoint_seconds = time.perf_counter()-calibration_start
                    for reference_spec in reference_specs:
                        spec = dict(reference_spec, experiment_id="ENDPOINT_BOOT")
                        key = (reference_spec["model_id"], window_id)
                        row, assets, attempts, current, weights, ok = solve_records(
                            context, spec, policy, initial_states[key], signs, endpoints=endpoints,
                            replicate=replicate, block=block, reference=reference_weights[key])
                        row["endpoint_time_sec"] = endpoint_seconds/len(reference_specs)
                        window_buffer.append(row)
                        asset_buffer.append(assets)
                        attempt_buffer.extend(attempts)
                        total += 1
                    if (replicate+1) % 50 == 0:
                        flush()
                flush()
                log(f"ENDPOINT_BOOT window {window_id:02d} block {block}: {len(indices)} samples x {len(reference_specs)} models")
    return total


def solve_summary(frame):
    return frame.groupby(["experiment_id", "model_id"], sort=False).agg(
        evaluations=("solver_status", "size"),
        direct=("solver_status", lambda s: int(s.eq("Direct").sum())),
        relaxed=("solver_status", lambda s: int(s.eq("Relaxed").sum())),
        failed=("solver_status", lambda s: int(s.eq("Failed").sum())),
        n_assets_min=("n_assets", "min"), n_assets_max=("n_assets", "max"),
        solver_seconds_min=("solver_time_sec", "min"), solver_seconds_median=("solver_time_sec", "median"),
        solver_seconds_mean=("solver_time_sec", "mean"), solver_seconds_p95=("solver_time_sec", lambda s: s.quantile(0.95)),
        solver_seconds_max=("solver_time_sec", "max"), reference_l1_median=("reference_weight_l1", "median"),
        mean_risky_exposure=("risky_exposure", "mean"), degenerate_endpoint_evaluations=("degenerate_endpoints", lambda s: int(s.gt(0).sum())),
    ).reset_index()


def solution_audit(regular, policy):
    """Use the same acceptance tolerance as each solver; failures stay explicit."""
    accepted = regular.solver_status.isin(["Direct", "Relaxed"])
    tolerance = pd.Series(float(policy["constraint_tolerance"]), index=regular.index)
    # solve_mv explicitly accepts OSQP residuals up to 1e-5.
    tolerance.loc[regular.risk_measure.eq("MV")] = 1e-5
    valid = (np.isfinite(regular.max_constraint_violation)
             & regular.max_constraint_violation.le(tolerance)
             & np.isfinite(regular.objective_value)
             & np.isfinite(regular.lp_cost))
    invalid = accepted & ~valid
    return bool(accepted.any() and not invalid.any()), invalid, tolerance


def select_experiments(specs, experiments):
    """Resolve presets without loading data or enabling bootstrap implicitly."""
    known = {s["experiment_id"] for s in specs} | {"ENDPOINT_BOOT"}
    presets = {
        "audit": {"BASELINE_AUDIT"},
        "substitution": {"BASELINE_AUDIT", "CVAR02_ROBUSTNESS"},
        "core": {"BASELINE_AUDIT", "CVAR02_ROBUSTNESS", "KPURE_CONTROL"},
        "standard": known - {"ENDPOINT_BOOT", "ENDPOINT_BOOT_REFERENCE"},
        "all": known,
    }
    requested = set(presets[experiments]) if experiments in presets else set(experiments.split(","))
    if not requested <= known:
        raise ValueError(f"Unknown experiments: {requested-known}")
    requested.add("BASELINE_AUDIT")
    if "ENDPOINT_BOOT" in requested:
        requested.add("ENDPOINT_BOOT_REFERENCE")
    if "K1_NEUTRAL" in requested:
        requested.add("FIXED_RULE_CONTROL")
    return requested


def run_stage2(input_dir, output_dir, baseline_dir, policy, experiments="core", overwrite=False,
               window_limit=None, prefix="step2"):
    input_dir, output_dir, baseline_dir = [Path(p).resolve() for p in (input_dir, output_dir, baseline_dir)]
    specs = experiment_specs(policy)
    requested = select_experiments(specs, experiments)
    specs = [s for s in specs if s["experiment_id"] in requested]
    names = [f"{prefix}_substitution_cross_risks.csv", f"{prefix}_substitution_weight_pairs.csv",
             f"{prefix}_windows.csv", f"{prefix}_assets.parquet", f"{prefix}_attempts.csv",
             f"{prefix}_baseline_reproduction.csv", f"{prefix}_solve_summary.csv",
             f"{prefix}_run_metadata.json", f"{prefix}_run.log",
             "risk_tail_mapping_check.csv", "lp_policy_lock.json", "feasibility_protocol.md"]
    if "ENDPOINT_BOOT" in requested:
        names += [f"{prefix}_bootstrap_{kind}.parquet" for kind in ["assets", "windows", "attempts"]]
    conflicts = [output_dir/n for n in names if (output_dir/n).exists()]
    if conflicts and not overwrite:
        raise FileExistsError(f"Existing stage2 outputs; explicit --overwrite required: {conflicts}")
    output_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    metadata = dict(run_id="step2_"+datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ"),
                    stage=2, experiment_id=experiments, started_utc=datetime.now(timezone.utc).isoformat(),
                    code_sha256=step1.sha256_file(__file__), policy=policy, settings=policy,
                    config_sha256=hashlib.sha256(json.dumps(policy, sort_keys=True).encode()).hexdigest(),
                    scope="pilot" if window_limit else "full", selected_experiments=sorted(requested),
                    python=platform.python_version(), scipy=scipy.__version__, numpy=np.__version__,
                    pandas=pd.__version__, pyarrow=_module_version("pyarrow"), cvxpy=_module_version("cvxpy"),
                    platform=platform.platform(), processor=platform.processor(),
                    solver_backend="SciPy bundled HiGHS; resolved through installed SciPy",
                    performance_backtest_gate="pending stage3/4; stage2 only audits numerical solutions",
                    pipeline_model="ModelA", execution_solver_available=True,
                    model_aliases=SUBSTITUTION_MODEL_ALIASES,
                    trajectory_scope="previous accepted target weights, not market-drifted execution holdings",
                    failed_window_execution_policy="unresolved: Stage3 must reconcile holdings before performance claims")
    sinks = []
    with (output_dir/f"{prefix}_run.log").open("w", encoding="utf-8") as log_file:
        def log(message):
            line = datetime.now(timezone.utc).isoformat()+" "+message
            log_file.write(line+"\n")
            log_file.flush()
            print(message, flush=True)

        try:
            prepared, stage1_metadata, signs = load_inputs(input_dir, policy)
            sources = [input_dir/n for n in [*step1.CORE_FILES, "endpoint_calibration.csv", "step1_run_metadata.json"]]
            if "ENDPOINT_BOOT" in requested:
                sources.append(input_dir/"bootstrap_indices.npz")
            # The former Model B weight files are deliberately never loaded.
            metadata["inputs"] = [{"path": str(p), "sha256": step1.sha256_file(p)} for p in sources]
            metadata["stage1_run_id"] = stage1_metadata.get("run_id", "")
            if window_limit:
                prepared = dict(list(prepared.items())[:window_limit])
            tail_check = risk_tail_mapping_check()
            if not tail_check.passed.all():
                raise ValueError("Risk-tail mapping check failed")
            tail_check.to_csv(output_dir/"risk_tail_mapping_check.csv", index=False, encoding="utf-8-sig")
            lock = dict(policy=policy, specs=specs, asset_signs=signs, bootstrap_policy="fixed reference initial positions; samples never chained",
                        degenerate_scope="Algebraic deterministic limits retained, outside strict nondegenerate endpoint theorem",
                        objective_note="Single adopted model: minimum-downside risk minimization under a required net expected return; the risk--mean scalarized objective is removed",
                        mv_note="MV is recomputed as a separate benchmark with the same 0.30% trading fee",
                        constant_objective_term="Risk-minimization objective drops the decision-independent -r_f constant")
            (output_dir/"lp_policy_lock.json").write_text(json.dumps(lock, ensure_ascii=False, indent=2), encoding="utf-8")
            (output_dir/"feasibility_protocol.md").write_text(
                "# Feasibility policy\n\n"
                "Single adopted model: minimum-downside risk minimization under a required net expected return. "
                "The risk--mean scalarized objective of the earlier revision is removed and is not retained.\n\n"
                "Revision LP scales: "+str(policy["revision_relaxation_scales"])+
                "; same across k, RVaR/CVaR, endpoint/interval/heterogeneous experiments.\n\n"
                "Only infeasibility triggers target reduction. No fallback; failures retain NaN weights.\n\n"
                "Each trajectory starts in cash, then references its own last accepted weights, reindexed to the current pool. "
                "Forced exits from a changing stock universe enter the objective and minimum-return constraint.\n\n"
                "Bootstrap samples always use the reference trajectory's initial weights for that window. "
                "No bootstrap sample affects the next sample or next window.\n\n"
                "3n variables; 2n+3 inequalities; 3n lower bounds, n finite upper bounds. "
                "Purchases and sales both incur a 0.30% fee; costs are charged per rebalance.\n", encoding="utf-8")
            asset_sink = ParquetSink(output_dir/f"{prefix}_assets.parquet")
            sinks.append(asset_sink)
            rows, attempts, comparisons = [], [], []
            trajectories, initials, references = {}, {}, {}
            audit_done = False
            for spec in specs:
                if spec["experiment_id"] != "BASELINE_AUDIT" and not audit_done:
                    pd.DataFrame(comparisons).to_csv(output_dir/f"{prefix}_baseline_reproduction.csv", index=False, encoding="utf-8-sig")
                    pd.DataFrame(rows).to_csv(output_dir/f"{prefix}_windows.csv", index=False, encoding="utf-8-sig")
                    pd.DataFrame(attempts).to_csv(output_dir/f"{prefix}_attempts.csv", index=False, encoding="utf-8-sig")
                    solve_summary(pd.DataFrame(rows)).to_csv(output_dir/f"{prefix}_solve_summary.csv", index=False, encoding="utf-8-sig")
                    gate_rows = [r for r in comparisons if r["gate_required"]]
                    gate_rows_present = [r for r in gate_rows if r["reason"] != "missing_historical_weights"]
                    # Old weights are diagnostic only; they may belong to a different model.
                    # Never gate a fresh Model A solution on Model B weight equality.
                    audit_done = True
                previous = None
                asset_frames = []
                for window_id, context in prepared.items():
                    key = (spec["model_id"], window_id)
                    if spec["experiment_id"] == "ENDPOINT_BOOT_REFERENCE":
                        initials[key] = None if previous is None else previous.copy()
                    reference_key = None
                    if spec["experiment_id"] == "K_HETEROG":
                        reference_key = (spec["experiment_id"], spec["risk_measure"]+"_heterog_control", window_id)
                    elif spec["experiment_id"] == "ENDPOINT_ALT":
                        suffix = {0.5:"05", 0.8:"08", 1.1:"11"}[spec["k"]]
                        reference_key = (spec["experiment_id"], spec["risk_measure"]+"_endpoint_baseline_k"+suffix, window_id)
                    elif spec["experiment_id"] == "RVAR_INTERVAL" and spec["risk_measure"] == "RVaR":
                        suffix = {0.5:"05", 0.8:"08", 1.1:"11"}[spec["k"]]
                        reference_key = (spec["experiment_id"], "RVaR_interval_baseline_k"+suffix, window_id)
                    row, assets, trial, previous, weights, ok = solve_records(context, spec, policy, previous, signs,
                                                                            reference=trajectories.get(reference_key))
                    row["endpoint_time_sec"] = 0.0
                    rows.append(row)
                    attempts.extend(trial)
                    asset_frames.append(assets)
                    trajectories[(spec["experiment_id"], spec["model_id"], window_id)] = weights
                    if spec["experiment_id"] == "ENDPOINT_BOOT_REFERENCE":
                        references[key] = weights
                    if spec["experiment_id"] == "BASELINE_AUDIT":
                        comparisons.append(baseline_comparison(window_id, spec, context["stocks"], weights, ok, baseline_dir, policy))
                asset_sink.write(pd.concat(asset_frames, ignore_index=True))
                statuses = pd.Series([r["solver_status"] for r in rows if r["model_id"] == spec["model_id"] and r["experiment_id"] == spec["experiment_id"]]).value_counts().to_dict()
                log(f"{spec['experiment_id']} {spec['model_id']}: {statuses}")
            regular = pd.DataFrame(rows)
            regular.to_csv(output_dir/f"{prefix}_windows.csv", index=False, encoding="utf-8-sig")
            pd.DataFrame(attempts).to_csv(output_dir/f"{prefix}_attempts.csv", index=False, encoding="utf-8-sig")
            pd.DataFrame(comparisons).to_csv(output_dir/f"{prefix}_baseline_reproduction.csv", index=False, encoding="utf-8-sig")
            gate_rows = [r for r in comparisons if r["gate_required"]]
            gate_rows_present = [r for r in gate_rows if r["reason"] != "missing_historical_weights"]
            # An honest gate: every accepted solution must have finite weights and
            # residuals within the fixed tolerance. No Model B comparison is claimed.
            gate_passed, invalid, tolerances = solution_audit(regular, policy)
            metadata["invalid_accepted_solutions"] = regular.loc[invalid,
                ["experiment_id", "model_id", "window_id", "max_constraint_violation"]].to_dict("records")
            metadata["acceptance_tolerances"] = {"LP": policy["constraint_tolerance"], "MV": 1e-5}
            metadata["failed_evaluations"] = int(regular.solver_status.eq("Failed").sum())
            metadata["relaxed_evaluations"] = int(regular.solver_status.eq("Relaxed").sum())
            metadata["all_windows_solved"] = bool(not regular.solver_status.eq("Failed").any())
            if metadata["failed_evaluations"]:
                log(f"WARNING: {metadata['failed_evaluations']} unsolved evaluations remain; "
                    "NaN weights are not executable portfolios. Stage3 must explicitly handle failed windows.")
            if not gate_passed:
                log(f"Numerical acceptance audit failed: {int(invalid.sum())} invalid accepted solutions")
            metadata["baseline_gate_passed"] = gate_passed
            metadata["baseline_gate_note"] = "No Model B weight files are loaded; numerical feasibility is checked using each solver acceptance tolerance"
            metadata["baseline_gate_scope"] = "Model A LP (policy tolerance) and MV (1e-5) constraint acceptance; historical weights are nonbinding diagnostics"
            metadata["all_model_reproduction_passed"] = None  # Not applicable for a fresh Model A run
            metadata["mv_reproduction_mismatches"] = None  # Not applicable
            metadata["mv_reference"] = "MV computed separately as a benchmark; no old-model weights are used"
            # All three in-sample risk criteria evaluated on each of the three
            # solutions, with explicit cross-model allocation comparisons.
            # Actual net-return comparison and statistical significance remain
            # separate stage3/stage4 responsibilities.
            asset_sink.close()
            asset_frame = pd.read_parquet(output_dir/f"{prefix}_assets.parquet")
            cross_risks, weight_pairs = substitution_diagnostics(regular, asset_frame, policy)
            if cross_risks.empty:
                cross_risks = pd.DataFrame(columns=["window_id", "k", "model_id", "experiment_id",
                    "solver_status", "cross_risk_CVaR05", "cross_risk_RVaR", "cross_risk_CVaR20",
                    "model_expected_net_return", "risky_exposure", "cash_weight", "trading_cost",
                    "theoretical_identity_checked", "diagnostic_scope"])
            if weight_pairs.empty:
                weight_pairs = pd.DataFrame(columns=["window_id", "k", "comparison", "left_model_id",
                    "right_model_id", "both_solver_accepted", "weight_l1_distance", "diagnostic_scope"])
            cross_risks.to_csv(output_dir/f"{prefix}_substitution_cross_risks.csv", index=False, encoding="utf-8-sig")
            weight_pairs.to_csv(output_dir/f"{prefix}_substitution_weight_pairs.csv", index=False, encoding="utf-8-sig")
            metadata["substitution_complete_trios"] = int(
                cross_risks.groupby(["window_id", "k"])["theoretical_identity_checked"].apply(
                    lambda flags: len(flags) == 3 and flags.fillna(False).all()).sum()) if not cross_risks.empty else 0
            metadata["substitution_reported_trios"] = int(len(cross_risks) // 3)
            metadata["substitution_scope"] = "in-sample cross-risk and weight comparisons only; stage3/4 required for actual performance"
            summary = solve_summary(regular)
            if "ENDPOINT_BOOT" in requested:
                boot_sinks = {kind: ParquetSink(output_dir/f"{prefix}_bootstrap_{kind}.parquet") for kind in ["windows", "assets", "attempts"]}
                sinks.extend(boot_sinks.values())
                reference_specs = [s for s in specs if s["experiment_id"] == "ENDPOINT_BOOT_REFERENCE"]
                metadata["bootstrap_evaluations"] = run_bootstrap(prepared, reference_specs, policy, signs, initials, references, input_dir, boot_sinks, log)
                for sink in boot_sinks.values():
                    sink.close()
                needed = ["experiment_id", "model_id", "solver_status", "n_assets", "solver_time_sec", "reference_weight_l1", "risky_exposure", "degenerate_endpoints"]
                boot_frame = pd.read_parquet(output_dir/f"{prefix}_bootstrap_windows.parquet", columns=needed)
                summary = pd.concat([summary, solve_summary(boot_frame)], ignore_index=True)
            summary.to_csv(output_dir/f"{prefix}_solve_summary.csv", index=False, encoding="utf-8-sig")
            metadata["regular_evaluations"] = len(rows)
            metadata["status"] = "complete_stage2_inputs" if metadata["baseline_gate_passed"] else "solution_audit_failed"
            if any(step1.sha256_file(i["path"]) != i["sha256"] for i in metadata["inputs"]):
                raise RuntimeError("An input changed during stage2")
            log("Stage2 finished; backtest/performance validation remains in stages3/4")
        except Exception as exc:
            metadata["status"] = "error"
            metadata["error"] = f"{type(exc).__name__}: {exc}"
            log(metadata["error"])
            raise
        finally:
            for sink in sinks:
                sink.close()
            metadata["finished_utc"] = datetime.now(timezone.utc).isoformat()
            metadata["elapsed_seconds"] = time.perf_counter()-started
            metadata["outputs"] = [{"path": str(output_dir/n), "sha256": step1.sha256_file(output_dir/n)}
                                   for n in names if (output_dir/n).is_file() and n not in {f"{prefix}_run_metadata.json", f"{prefix}_run.log"}]
            (output_dir/f"{prefix}_run_metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
            registry = output_dir/"run_registry.csv"
            record = {k: metadata.get(k) for k in ["run_id", "stage", "experiment_id", "started_utc", "finished_utc", "status", "code_sha256", "config_sha256"]}
            record["metadata_json"] = json.dumps(metadata, ensure_ascii=False)
            pd.DataFrame([record]).to_csv(registry, mode="a", header=not registry.exists(), index=False, encoding="utf-8")
    return metadata


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=BASE_DIR/"results1")
    parser.add_argument("--out", type=Path, default=BASE_DIR/"results2")
    parser.add_argument("--baseline", type=Path, default=BASE_DIR/"results2")
    parser.add_argument("--config", type=Path, help="Policy JSON or lp_policy_lock.json")
    parser.add_argument("--experiments", default="core", help="core (default), audit, substitution, standard (all regular), all (includes bootstrap), or comma-separated experiment IDs")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--window-limit", type=int, help="Pilot only; uses a separate step2_pilot prefix")
    parser.add_argument("--bootstrap-replicates", type=int, help="Override sample count; recorded in policy")
    args = parser.parse_args(argv)
    if args.window_limit is not None and not 1 <= args.window_limit <= 40:
        parser.error("window-limit must be inside 1..40")
    policy = load_policy(args.config)
    if args.bootstrap_replicates is not None:
        policy["bootstrap_replicates"] = args.bootstrap_replicates
        # Revalidate all merged settings before generating anything.
        if not 1 <= args.bootstrap_replicates <= 300:
            parser.error("bootstrap-replicates must be inside 1..300")
    metadata = run_stage2(args.input, args.out, args.baseline, policy, args.experiments,
                          args.overwrite, args.window_limit, "step2_pilot" if args.window_limit else "step2")
    print(f"Stage2 status: {metadata['status']}; elapsed {metadata['elapsed_seconds']:.2f}s")
    return 0 if metadata["status"] == "complete_stage2_inputs" else 2



def solve_execution_modelA(mu, rho, rf, target_annual, previous, floor, policy,
                           locks=None, forced_zero=None, sigma=None, scales=None):
    """Self-financing Model A at rebalance; all amounts / pre-trade NAV.

    x are asset amounts after trades, fee=c_buy*buy+c_sell*sell,
    cash=1-sum(x)-fee. Min rho*x-rf*cash+fee (or MV benchmark).
    Net expected daily criterion remains mu*x+rf*cash-fee >= annual/252.
    The one-off fee is NOT silently divided by holding days. This preserves
    the existing conservative fee/return criterion, with exact cash accounting.
    No expected-return term is added to the risk objective (Model B excluded).
    """
    mu,rho,previous = [np.asarray(a,float) for a in (mu,rho,previous)]
    n=len(mu); tol=policy['constraint_tolerance']
    if not n or any(a.shape!=(n,) or not np.isfinite(a).all() for a in (mu,rho,previous)):
        raise ValueError('Nonfinite execution inputs')
    if (previous < -tol).any() or previous.sum()>1+tol or not np.isfinite(rf):
        raise ValueError('Invalid pre-trade holdings/cash')
    locks=np.full(n,np.nan) if locks is None else np.asarray(locks,float)
    forced_zero=np.zeros(n,bool) if forced_zero is None else np.asarray(forced_zero,bool)
    if locks.shape!=(n,) or forced_zero.shape!=(n,): raise ValueError('Invalid execution bounds')
    b,s=policy['cost_buy'],policy['cost_sell']
    fees=np.r_[np.zeros(n),np.full(n,b),np.full(n,s)]
    expo=np.r_[np.ones(n),np.zeros(2*n)]
    net=np.r_[mu-rf,np.zeros(2*n)]-(1+rf)*fees
    obj=np.r_[rho+rf,np.zeros(2*n)]+(1+rf)*fees
    equality=np.c_[np.eye(n),-np.eye(n),np.eye(n)]
    bounds=[(0,policy['weight_upper']) for _ in range(n)]+[(0,None)]*(2*n)
    for j in range(n):
        if np.isfinite(locks[j]):
            if abs(locks[j]-previous[j])>tol: raise ValueError('Lock must equal actual holdings')
            # Price drift can put a locked holding above the usual trade cap.
            bounds[j]=(previous[j],previous[j]); bounds[n+j]=bounds[2*n+j]=(0,0)
        elif forced_zero[j]: bounds[j]=(0,0)
    scales=policy['revision_relaxation_scales'] if scales is None else scales
    attempts=[]
    for level,scale in enumerate(scales):
        A=np.vstack([expo+fees,-expo,-net])
        rhs=np.array([1-policy['cash_lower'],-floor,rf-target_annual/policy['trading_days']*scale])
        started=time.perf_counter()
        if sigma is None:
            result=linprog(obj,A_ub=A,b_ub=rhs,A_eq=equality,b_eq=previous,bounds=bounds,
                           method=policy['solver_method'],options=policy['solver_options'])
            code=int(result.status); message=str(result.message)
            candidate=result.x if result.success else None
        else:
            import cvxpy as cp
            z=cp.Variable(3*n)
            cons=[A@z<=rhs,equality@z==previous]
            for j,(lo,hi) in enumerate(bounds):
                cons.append(z[j]>=lo)
                if hi is not None:cons.append(z[j]<=hi)
            problem=cp.Problem(cp.Minimize(cp.quad_form(z[:n],cp.psd_wrap(sigma))+fees@z),cons)
            try:
                # Let the CVXPY adapter choose OSQP's version-specific polishing
                # option. OSQP releases use different keyword spellings; explicit
                # polishing=True is rejected by some older installations.
                # Accuracy is still audited below using the unchanged tolerance.
                problem.solve(solver=cp.OSQP,eps_abs=1e-9,eps_rel=1e-9,max_iter=200000)
                message=str(problem.status)
                code=0 if problem.status in (cp.OPTIMAL,cp.OPTIMAL_INACCURATE) else (2 if problem.status==cp.INFEASIBLE else 4)
                candidate=z.value if code==0 else None
            except cp.error.SolverError as exc:
                code,message,candidate=4,str(exc),None
        residual=np.nan; accepted=False
        if candidate is not None and np.isfinite(candidate).all():
            # Audit executable trades reconstructed from x; eliminate fictitious round trips.
            x=np.asarray(candidate[:n]).clip(0,None)
            actual=np.r_[x,np.maximum(x-previous,0),np.maximum(previous-x,0)]
            residual=max(0,float(np.max(A@actual-rhs)),float(np.max(abs(equality@actual-previous))))
            for j,(lo,hi) in enumerate(bounds):
                residual=max(residual,lo-actual[j],actual[j]-hi if hi is not None else 0)
            accepted=residual<=tol
        attempts.append(dict(attempt=level,scale=scale,target_annual=target_annual*scale,
                             solver_code=code,accepted=bool(accepted),message=message,
                             max_constraint_violation=residual,solver_time_sec=time.perf_counter()-started))
        if accepted:
            fee=float(fees@actual);cash=float(1-x.sum()-fee)
            info=dict(status='Direct' if level==0 else 'Relaxed',scale=scale,fee=fee,cash=cash,
                      target_annual_effective=target_annual*scale,
                      expected_net_daily=float(mu@x+rf*cash-fee),
                      return_slack=float(mu@x+rf*cash-fee-target_annual/policy['trading_days']*scale),
                      residual=residual,objective=float(x@rho-rf*cash+fee) if sigma is None else float(x@sigma@x+fee),
                      attempts=attempts)
            return x,info
        if code!=2:
            raise RuntimeError(f'Execution solver numerical/error state (not infeasibility): {message}; residual={residual}')
    return None,dict(status='Failed',scale=np.nan,fee=0.,cash=1-previous.sum(),
                     target_annual_effective=np.nan,expected_net_daily=np.nan,
                     return_slack=np.nan,residual=np.nan,objective=np.nan,attempts=attempts)


if __name__ == "__main__":
    sys.exit(main())