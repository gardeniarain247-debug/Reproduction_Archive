"""Offline stage-1 build/audit and calibration; importing this module has no I/O.

Default: python step1.py
    Builds the six canonical CSVs from local cache into --results
    (default: <code>/results1), reading bond/constituent from --data
    (default: <code>/data). Stock caches are read from --cache
    (default: <results>/baostock_cache).

Rebuild explicitly:
    python step1.py --mode build --out <folder>
Audit an existing frozen baseline:
    python step1.py --mode audit --results <frozen> --out <audit_out>

Downloads require --allow-network. A build into a directory that already
contains the six core CSVs is refused, so a frozen baseline cannot be
silently overwritten.
"""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import logging
from pathlib import Path
import platform
import sys
import time

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
CORE_FILES = ("full_return_matrix.csv", "full_turnover_matrix.csv", "risk_free_rate.csv",
              "rolling_windows.csv", "windows_universe.csv", "fuzzy_parameters.csv")
WINDOW_COLUMNS = ["window_id", "est_start", "est_end", "test_start", "test_end"]
ENDPOINT_SCHEMES = {
    "baseline": [0.05, 0.25, 0.75, 0.95],
    "alternative_a": [0.05, 0.20, 0.80, 0.95],
    "alternative_b": [0.10, 0.30, 0.70, 0.90],
}
DEFAULT_SETTINGS = {
    "estimation_days": 250, "holding_days": 60, "step_days": 60,
    "expected_windows": 40, "min_observations": 200,
    "membership_rule": "estimation_start", "quantile_method": "linear",
    "endpoint_schemes": ENDPOINT_SCHEMES, "bootstrap_block_lengths": [5, 10],
    "bootstrap_replicates": 300, "random_seed": 42, "anomaly_abs_return": 0.20,
    "start_date": "2015-01-01", "end_date": "2025-12-31", "adjustflag": "1",
    "trading_days_per_year": 252,
    "bond_file": "bond_1_return_20140101_20260310.csv",
    "constituent_file": "sh50_2015_2025.xlsx",
}
LOGGER = logging.getLogger("step1")


def normalize_stock_code(code):
    value = str(code).strip()
    if value.endswith(".0") and value[:-2].isdigit():
        value = value[:-2]
    if "." in value:
        value = value.rsplit(".", 1)[-1]
    if not value.isdigit() or len(value) > 6:
        raise ValueError(f"Invalid stock code: {code!r}")
    return value.zfill(6)


def to_baostock_code(code):
    return f"sh.{normalize_stock_code(code)}"


def safe_to_csv(df, path, overwrite=False, index=False):
    path = Path(path)
    if path.exists() and not overwrite:
        raise FileExistsError(f"Output already exists: {path}")
    df.to_csv(path, index=index, encoding="utf-8-sig")
    LOGGER.info("Saved: %s", path)


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_probabilities(probabilities):
    values = np.asarray(probabilities, dtype=float)
    if values.shape != (4,) or not np.all(np.isfinite(values)):
        raise ValueError("Exactly four finite endpoint probabilities are required")
    if not (0 < values[0] < values[1] < values[2] < values[3] < 1):
        raise ValueError("Endpoint probabilities must be ordered inside (0, 1)")
    return values


def validate_settings(settings):
    names = ["estimation_days", "holding_days", "step_days", "expected_windows",
             "min_observations", "bootstrap_replicates", "trading_days_per_year"]
    for name in names:
        if isinstance(settings[name], bool) or not isinstance(settings[name], int) or settings[name] <= 0:
            raise ValueError(f"{name} must be a positive integer")
    if settings["min_observations"] > settings["estimation_days"]:
        raise ValueError("min_observations exceeds estimation_days")
    if settings["step_days"] != settings["holding_days"]:
        raise ValueError("Locked design requires step_days == holding_days")
    if settings["membership_rule"] != "estimation_start" or settings["quantile_method"] != "linear":
        raise ValueError("Baseline membership and quantile rules are locked")
    if "baseline" not in settings["endpoint_schemes"]:
        raise ValueError("endpoint_schemes must include baseline")
    for probabilities in settings["endpoint_schemes"].values():
        validate_probabilities(probabilities)
    lengths = settings["bootstrap_block_lengths"]
    if not lengths or len(set(lengths)) != len(lengths):
        raise ValueError("Bootstrap block lengths must be nonempty and unique")
    if any(isinstance(n, bool) or not isinstance(n, int) or not 1 <= n <= settings["estimation_days"] for n in lengths):
        raise ValueError("Invalid bootstrap block length")
    if not isinstance(settings["random_seed"], int) or settings["random_seed"] < 0:
        raise ValueError("random_seed must be a nonnegative integer")
    if not np.isfinite(settings["anomaly_abs_return"]) or settings["anomaly_abs_return"] <= 0:
        raise ValueError("anomaly_abs_return must be positive")
    if pd.Timestamp(settings["start_date"]) > pd.Timestamp(settings["end_date"]):
        raise ValueError("start_date is after end_date")


def load_settings(path=None):
    settings = json.loads(json.dumps(DEFAULT_SETTINGS))
    if path:
        supplied = json.loads(Path(path).read_text(encoding="utf-8-sig"))
        supplied = supplied.get("settings", supplied)
        unknown = set(supplied) - set(settings)
        if unknown:
            raise ValueError(f"Unknown settings: {sorted(unknown)}")
        settings.update(supplied)
    validate_settings(settings)
    return settings


def read_daily_matrix(path, stock_columns=False):
    frame = pd.read_csv(path, index_col=0, parse_dates=True)
    frame.index = pd.DatetimeIndex(frame.index)
    if frame.index.hasnans or frame.index.has_duplicates or not frame.index.is_monotonic_increasing:
        raise ValueError(f"Dates must be valid, unique and sorted: {path}")
    if stock_columns:
        frame.columns = [normalize_stock_code(c) for c in frame.columns]
    if frame.columns.has_duplicates:
        raise ValueError(f"Duplicate columns: {path}")
    return frame.apply(pd.to_numeric, errors="raise")


def read_constituents(path):
    frame = pd.read_excel(path, dtype={"stock_code": str})
    for name in ["period_start", "period_end"]:
        values = frame[name].astype(str).str.replace(r"\.0$", "", regex=True)
        frame[name] = pd.to_datetime(values, format="%Y%m%d")
    frame["stock_code"] = frame.stock_code.map(normalize_stock_code)
    if (frame.period_start > frame.period_end).any():
        raise ValueError("Constituent start date is after end date")
    return frame


def read_bond(path, trading_days=252):
    frame = pd.read_csv(path, parse_dates=["date"]).set_index("date")
    if frame.index.has_duplicates or frame.index.hasnans:
        raise ValueError("Invalid or duplicate bond dates")
    return (pd.to_numeric(frame["returns"], errors="raise") / 100 / trading_days).sort_index()


def align_risk_free(bond, dates, policy="forward_only"):
    """Trace each value to its source date; legacy future backfill stays visible."""
    dates = pd.DatetimeIndex(dates)
    valid = bond.dropna()
    dated = pd.DataFrame({"rf_daily": valid, "source_date": valid.index}, index=valid.index)
    if policy == "legacy":
        aligned = dated.reindex(dates).ffill().bfill()
    elif policy == "forward_only":
        aligned = dated.reindex(dated.index.union(dates)).sort_index().ffill().reindex(dates)
    else:
        raise ValueError(f"Unknown risk-free alignment policy: {policy}")
    aligned["used_future_value"] = aligned.source_date > aligned.index.to_series()
    return aligned


def build_windows(dates, est=250, test=60, step=None):
    dates = pd.DatetimeIndex(dates)
    if dates.has_duplicates or dates.hasnans or not dates.is_monotonic_increasing:
        raise ValueError("Rolling dates must be valid, unique and sorted")
    step = test if step is None else step
    if min(est, test, step) <= 0:
        raise ValueError("Rolling lengths must be positive")
    rows = []
    for start in range(0, len(dates) - est - test + 1, step):
        rows.append([len(rows), dates[start], dates[start + est - 1],
                     dates[start + est], dates[start + est + test - 1]])
    return pd.DataFrame(rows, columns=WINDOW_COLUMNS)


def get_stocks(date, constituents):
    """Keep the original inclusive boundaries and constituent source order."""
    mask = (constituents.period_start <= date) & (constituents.period_end >= date)
    return constituents.loc[mask, "stock_code"].drop_duplicates().tolist()


def fuzzy_params(series, probabilities=None, method="linear"):
    probabilities = validate_probabilities(ENDPOINT_SCHEMES["baseline"] if probabilities is None else probabilities)
    values = pd.to_numeric(pd.Series(series), errors="raise").dropna().to_numpy(dtype=float)
    if not np.all(np.isfinite(values)):
        raise ValueError("Infinite returns in fuzzy calibration")
    endpoints = np.quantile(values, probabilities, method=method) if len(values) else np.full(4, np.nan)
    return dict(zip(list("abcd"), endpoints), mean=float(values.mean()) if len(values) else np.nan, obs=len(values))


def endpoint_status(parameters):
    a, b, c, d = (parameters[name] for name in "abcd")
    if not np.all(np.isfinite([a, b, c, d])):
        return "missing"
    return "valid" if a < b <= c < d else "degenerate"


def calibrate_windows(returns, windows, constituents, settings):
    """Only estimation data determine selection; all endpoint schemes share it."""
    universe_rows, parameter_rows, asset_rows, master_rows = [], [], [], []
    for win in windows.itertuples(index=False):
        candidates = get_stocks(win.est_start, constituents)
        hold_candidates = get_stocks(win.test_start, constituents)
        training = returns.loc[win.est_start:win.est_end]
        holding = returns.loc[win.test_start:win.test_end]
        selected = []
        for code in candidates:
            available = code in returns.columns
            values = training[code] if available else pd.Series(dtype=float)
            obs = int(values.notna().sum())
            included = available and obs >= settings["min_observations"]
            reason = "included" if included else ("missing_return_column" if not available else "insufficient_estimation_observations")
            valid_dates = values.dropna().index
            asset_rows.append({
                "window_id": win.window_id, "stock_code": code, "included": included, "reason": reason,
                "estimation_observations": obs, "estimation_missing_days": len(training) - obs,
                "first_observation": valid_dates.min() if len(valid_dates) else pd.NaT,
                "last_observation": valid_dates.max() if len(valid_dates) else pd.NaT,
                "membership_reference_date": win.est_start, "also_member_at_hold_start": code in hold_candidates,
            })
            if not included:
                continue
            selected.append(code)
            base = {name: getattr(win, name) for name in WINDOW_COLUMNS}
            universe_rows.append(dict(base, stock_code=code, universe_rank=len(selected)))
            for scheme, probabilities in settings["endpoint_schemes"].items():
                parameters = fuzzy_params(values, probabilities, settings["quantile_method"])
                parameter_rows.append(dict(parameters, window_id=win.window_id, stock_code=code,
                                           endpoint_scheme=scheme, endpoint_status=endpoint_status(parameters)))
        master_rows.append({
            **{name: getattr(win, name) for name in WINDOW_COLUMNS},
            "estimation_days": len(training), "holding_days": len(holding),
            "candidate_assets": len(candidates), "eligible_assets": len(selected),
            "excluded_assets": len(candidates) - len(selected), "members_at_hold_start": len(hold_candidates),
            "membership_changes_to_hold_start": len(set(candidates) ^ set(hold_candidates)),
            "holding_missing_observations": int(holding[selected].isna().sum().sum()),
            "estimation_precedes_holding": win.est_end < win.test_start,
        })
    universe_columns = WINDOW_COLUMNS + ["stock_code", "universe_rank"]
    parameter_columns = list("abcd") + ["mean", "obs", "window_id", "stock_code", "endpoint_scheme", "endpoint_status"]
    return (pd.DataFrame(universe_rows, columns=universe_columns),
            pd.DataFrame(parameter_rows, columns=parameter_columns), pd.DataFrame(asset_rows), pd.DataFrame(master_rows))


def moving_block_indices(n_observations, block_length, replicates, rng):
    """Non-circular moving blocks truncated to the original window length."""
    if not 1 <= block_length <= n_observations or replicates <= 0:
        raise ValueError("Invalid moving-block dimensions")
    blocks = (n_observations + block_length - 1) // block_length
    starts = rng.integers(0, n_observations - block_length + 1, size=(replicates, blocks))
    return (starts[..., None] + np.arange(block_length)).reshape(replicates, -1)[:, :n_observations]


def bootstrap_window_indices(windows, returns, settings):
    """Save training dates plus indices shared across all assets in a replicate."""
    arrays = {}
    for win in windows.itertuples(index=False):
        dates = returns.loc[win.est_start:win.est_end].index
        arrays[f"dates_win{win.window_id}"] = dates.to_numpy(dtype="datetime64[ns]")
        for length in settings["bootstrap_block_lengths"]:
            rng = np.random.default_rng(np.random.SeedSequence([settings["random_seed"], win.window_id, length]))
            indices = moving_block_indices(len(dates), length, settings["bootstrap_replicates"], rng)
            arrays[f"indices_win{win.window_id}_block{length}"] = indices.astype(np.int32)
    return arrays


def download_one_stock(code, cache_dir, start_date="2015-01-01", end_date="2025-12-31",
                       adjustflag="1", allow_network=False, max_retry=3):
    """Read cache; caller logs in before explicitly enabled downloads."""
    path = Path(cache_dir) / f"{normalize_stock_code(code)}.csv"
    if path.exists():
        return pd.read_csv(path, parse_dates=["date"])
    if not allow_network:
        raise FileNotFoundError(f"Missing cache (network disabled): {path}")
    import baostock as bs
    for attempt in range(max_retry):
        result = bs.query_history_k_data_plus(to_baostock_code(code), "date,close,turn",
                                             start_date=start_date, end_date=end_date,
                                             frequency="d", adjustflag=adjustflag)
        if result.error_code == "0":
            frame = result.get_data()
            if not frame.empty:
                frame["date"] = pd.to_datetime(frame["date"])
                path.parent.mkdir(parents=True, exist_ok=True)
                frame.to_csv(path, index=False, encoding="utf-8-sig")
                return frame
        LOGGER.warning("Download %s attempt %s: %s", code, attempt + 1, result.error_msg)
        time.sleep(0.5)
    raise RuntimeError(f"Could not download {code}")


def matrices_from_cache(constituents, cache_dir, settings, allow_network=False):
    prices, turnover = [], []
    if allow_network:
        import baostock as bs
        login = bs.login()
        if login.error_code != "0":
            raise RuntimeError(f"Baostock login failed: {login.error_msg}")
    try:
        for code in sorted(constituents.stock_code.unique()):
            frame = download_one_stock(code, cache_dir, settings["start_date"], settings["end_date"],
                                       settings["adjustflag"], allow_network=allow_network)
            frame = frame.set_index("date").sort_index()
            if frame.index.has_duplicates:
                raise ValueError(f"Duplicate cache dates: {code}")
            frame = frame.loc[settings["start_date"]:settings["end_date"]]
            prices.append(pd.to_numeric(frame["close"], errors="raise").rename(code))
            turnover.append(pd.to_numeric(frame["turn"], errors="raise").rename(code))
    finally:
        if allow_network:
            bs.logout()
    price_frame = pd.concat(prices, axis=1).sort_index().dropna(how="all")
    turnover_frame = pd.concat(turnover, axis=1).sort_index().reindex(price_frame.index)
    return price_frame.pct_change(fill_method=None).dropna(how="all"), turnover_frame


def read_result_table(path, code_column=False):
    frame = pd.read_csv(path, dtype={"stock_code": str} if code_column else None)
    frame = frame.loc[:, ~frame.columns.str.startswith("Unnamed:")]
    if code_column:
        frame["stock_code"] = frame.stock_code.map(normalize_stock_code)
    return frame


def audit_inputs(returns, turnover, windows, universe, parameters, master, bond,
                 existing_dir, constituents, cache_dir, settings, mode):
    """Separate integrity failures, review items and unverified provenance."""
    rows = []

    def record(check, status, actual, expected="", detail="", scope="dataset"):
        rows.append(dict(check_id=check, scope=scope, status=status, actual=actual, expected=expected, detail=detail))

    def match_numeric(check, rebuilt, stored):
        if not (rebuilt.index.equals(stored.index) and rebuilt.columns.equals(stored.columns)):
            record(check, "failed", "different date/column axes", "identical axes")
            return
        a, b = rebuilt.to_numpy(dtype=float), stored.to_numpy(dtype=float)
        same = np.allclose(a, b, rtol=0, atol=1e-12, equal_nan=True)
        difference = np.abs(a - b)
        maximum = float(np.nanmax(difference)) if np.isfinite(difference).any() else np.nan
        record(check, "pass" if same else "failed", maximum, "abs difference <= 1e-12; equal missing mask")

    record("completed_windows", "pass" if len(windows) == settings["expected_windows"] else "failed",
           len(windows), settings["expected_windows"])
    timing_ok = bool(master.estimation_precedes_holding.all() and
                     master.estimation_days.eq(settings["estimation_days"]).all() and
                     master.holding_days.eq(settings["holding_days"]).all())
    record("window_timing", "pass" if timing_ok else "failed", timing_ok, True)
    record("infinite_returns", "pass" if not np.isinf(returns.to_numpy()).any() else "failed",
           int(np.isinf(returns.to_numpy()).sum()), 0)
    record("missing_return_observations", "info", int(returns.isna().sum().sum()),
           detail="Retained; no holding data used to filter eligibility")
    for date, column in zip(*np.where(np.abs(returns.to_numpy()) > settings["anomaly_abs_return"])):
        record("large_absolute_return", "review", float(returns.iloc[date, column]), settings["anomaly_abs_return"],
               detail=f"date={returns.index[date].date()}; no automatic removal", scope=str(returns.columns[column]))
    degenerate = parameters.loc[parameters.endpoint_status != "valid"]
    record("endpoint_order", "pass" if degenerate.empty else "review", len(degenerate), 0,
           detail="No jitter or exclusion; solver policy must handle flagged cases")
    record("training_missing_value_policy", "info", "dropna separately per asset",
           detail="Bootstrap retains synchronized date vectors and their missing values")
    record("membership_reference", "info", "estimation_start; inclusive boundaries",
           detail="Historical baseline, not a rebalance-date universe")
    changed = int(master.membership_changes_to_hold_start.gt(0).sum())
    record("membership_changes_before_holding", "review" if changed else "pass", changed,
           detail="Windows with different holding-start constituent sets")
    missing_hold = int(master.holding_missing_observations.sum())
    record("holding_missing_eligible_returns", "review" if missing_hold else "pass", missing_hold, 0)
    legacy, causal = (align_risk_free(bond, returns.index, policy) for policy in ["legacy", "forward_only"])
    record("risk_free_exact_date_missing", "info", int(bond.reindex(returns.index).isna().sum()),
           detail="Original alignment forward-filled exact-date gaps")
    future = int(legacy.used_future_value.sum())
    record("risk_free_future_backfill", "pass" if not future else "failed", future, 0)
    missing_causal = int(causal.rf_daily.isna().sum())
    record("risk_free_causal_missing", "pass" if not missing_causal else "failed", missing_causal, 0)
    match_numeric("risk_free_legacy_vs_causal", legacy[["rf_daily"]], causal[["rf_daily"]])
    if mode == "audit":
        stored_rf = read_daily_matrix(Path(existing_dir) / "risk_free_rate.csv")
        if stored_rf.shape[1] != 1:
            raise ValueError("Expected one risk-free column")
        stored_rf.columns = ["rf_daily"]
        match_numeric("risk_free_existing_reproduction", legacy[["rf_daily"]], stored_rf)
        stored_windows = read_result_table(Path(existing_dir) / "rolling_windows.csv")
        for name in WINDOW_COLUMNS[1:]:
            stored_windows[name] = pd.to_datetime(stored_windows[name])
        same = windows[WINDOW_COLUMNS].equals(stored_windows[WINDOW_COLUMNS])
        record("rolling_windows_existing_reproduction", "pass" if same else "failed", same, True)
        stored_universe = read_result_table(Path(existing_dir) / "windows_universe.csv", True)
        for name in WINDOW_COLUMNS[1:]:
            stored_universe[name] = pd.to_datetime(stored_universe[name])
        columns = ["window_id", "stock_code", "universe_rank", *WINDOW_COLUMNS[1:]]
        same = universe[columns].reset_index(drop=True).equals(stored_universe[columns].reset_index(drop=True))
        record("universe_existing_reproduction", "pass" if same else "failed", same, True)
        base = parameters.loc[parameters.endpoint_scheme == "baseline"].set_index(["window_id", "stock_code"])
        stored = read_result_table(Path(existing_dir) / "fuzzy_parameters.csv", True).set_index(["window_id", "stock_code"])
        columns = list("abcd") + ["mean", "obs"]
        match_numeric("fuzzy_existing_reproduction", base[columns], stored[columns])
        try:
            rebuilt_returns, rebuilt_turnover = matrices_from_cache(constituents, cache_dir, settings)
        except FileNotFoundError as exc:
            record("cache_reproduction", "pending", "cache incomplete", detail=str(exc))
        else:
            if set(rebuilt_returns.columns) == set(returns.columns):
                rebuilt_returns = rebuilt_returns[returns.columns]
            if set(rebuilt_turnover.columns) == set(turnover.columns):
                rebuilt_turnover = rebuilt_turnover[turnover.columns]
            match_numeric("return_matrix_cache_reproduction", rebuilt_returns, returns)
            match_numeric("turnover_matrix_cache_reproduction", rebuilt_turnover, turnover)
    pending = {
        "cached_price_adjustment_provenance": "Original code requests adjustflag=1; cache has no request metadata",
        "suspension_and_listing_provenance": "Cache date/close/turn cannot identify suspension or listing reasons",
        "dividend_and_constituent_effective_date_provenance": "Requires source documentation, not inferred from tables",
        "paper_g2_version": "Current supplied PDF is hashed; its g2 label is unverified",
    }
    for check, explanation in pending.items():
        record(check, "pending", "not verified", detail=explanation)
    return pd.DataFrame(rows), legacy, causal


def implementation_parameters(settings, mode, data_dir, existing_dir, cache_dir):
    values = dict(settings)
    values.update({
        "mode": mode, "data_dir": str(data_dir), "existing_results_dir": str(existing_dir), "cache_dir": str(cache_dir),
        "bootstrap_method": "non-circular moving-block; shared date vectors",
        "missing_values": "retain synchronized date vectors; drop missing per asset in quantiles",
        "eligibility_for_alternatives": "same baseline stock pool; no endpoint-based exclusion",
        "degenerate_endpoints": "flag only; no perturbation, exclusion or solver decision in stage1",
        "risk_free_legacy_policy": "annual percent / 100 / trading_days_per_year; exact reindex then ffill/bfill",
        "risk_free_build_policy": "last available prior observation; leading missing prohibited",
        "price_return_policy": "pct_change(fill_method=None)", "abnormal_returns": "flag only; no winsorization",
    })
    return pd.DataFrame([{"parameter": k, "value": json.dumps(v) if isinstance(v, (dict, list)) else v}
                         for k, v in values.items()])


def output_names(mode, bootstrap=True):
    names = ["windows_master_40.csv", "input_data_audit.csv", "universe_asset_audit.csv",
             "endpoint_calibration.csv", "risk_free_alignment_audit.csv", "implementation_parameters.csv",
             "step1_run_metadata.json", "step1_run.log"]
    if bootstrap:
        names.append("bootstrap_indices.npz")
    return names + (list(CORE_FILES) if mode == "build" else [])


def input_manifest(data_dir, existing_dir, cache_dir, settings):
    sources = [data_dir / settings["bond_file"], data_dir / settings["constituent_file"]]
    sources += [existing_dir / name for name in CORE_FILES if (existing_dir / name).is_file()]
    sources += sorted(cache_dir.glob("*.csv"))
    for path in [BASE_DIR.parent / "manuscript/manuscript.pdf", existing_dir / "step1_baseline_snapshot.zip"]:
        if path.is_file():
            sources.append(path)
    return [{"path": str(p), "sha256": sha256_file(p)} for p in sources]


def register_run(out_dir, metadata):
    """Append an immutable record, including hashes even if diagnostics are replaced."""
    registry = out_dir / "run_registry.csv"
    fields = ["run_id", "stage", "experiment_id", "started_utc", "finished_utc", "status",
              "code_sha256", "config_sha256", "settings_json", "inputs_json", "outputs_json"]
    row = {name: metadata.get(name, "") for name in fields}
    row.update(stage=1, settings_json=json.dumps(metadata["settings"], sort_keys=True),
               inputs_json=json.dumps(metadata.get("inputs", [])),
               outputs_json=json.dumps(metadata.get("outputs", [])))
    if registry.exists():
        with registry.open(encoding="utf-8-sig", newline="") as stream:
            if next(csv.reader(stream), None) != fields:
                raise ValueError(f"Unexpected registry schema: {registry}")
    with registry.open("a", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        if stream.tell() == 0:
            writer.writeheader()
        writer.writerow(row)


def run_stage1(data_dir, existing_dir, out_dir, settings, mode="build", cache_dir=None,
               allow_network=False, overwrite=False, generate_bootstrap=True, experiment_id="BASELINE_AUDIT"):
    validate_settings(settings)
    data_dir, existing_dir, out_dir = [Path(p).resolve() for p in (data_dir, existing_dir, out_dir)]
    cache_dir = Path(cache_dir).resolve() if cache_dir else existing_dir / "baostock_cache"
    if mode not in {"audit", "build"}:
        raise ValueError("mode must be audit or build")
    if allow_network and mode != "build":
        raise ValueError("Audit mode never uses the network")
    if mode == "build" and out_dir == existing_dir:
        # Only protect an *already frozen* baseline; allow the first build into --results.
        frozen_core = [out_dir / name for name in CORE_FILES if (out_dir / name).exists()]
        frozen_markers = [out_dir/name for name in ["windows_master_40.csv", "step1_run_metadata.json"] if (out_dir/name).exists()]
        if frozen_core or (overwrite and frozen_markers):
            raise ValueError("Build requires a separate output directory to preserve the frozen baseline")
    names = output_names(mode, generate_bootstrap)
    conflicts = [out_dir / name for name in names if (out_dir / name).exists()]
    if conflicts and not overwrite:
        raise FileExistsError(f"Outputs already exist (explicit --overwrite required): {conflicts}")
    out_dir.mkdir(parents=True, exist_ok=True)
    handler = logging.FileHandler(out_dir / "step1_run.log", mode="w", encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    LOGGER.addHandler(handler)
    LOGGER.setLevel(logging.INFO)
    started = time.perf_counter()
    metadata = {
        "experiment_id": experiment_id, "started_utc": datetime.now(timezone.utc).isoformat(), "mode": mode,
        "settings": settings, "code_sha256": sha256_file(__file__), "python": platform.python_version(),
        "numpy": np.__version__, "pandas": pd.__version__, "data_dir": str(data_dir),
        "existing_results_dir": str(existing_dir), "out_dir": str(out_dir),
        "bootstrap_generated": generate_bootstrap, "network_allowed": allow_network,
        "paper_version": "Model A reviewer-completion workflow",
    }
    metadata["run_id"] = "step1_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    metadata["config_sha256"] = hashlib.sha256(json.dumps(settings, sort_keys=True).encode()).hexdigest()
    try:
        LOGGER.info("Stage1 started: %s (%s)", experiment_id, mode)
        metadata["inputs"] = input_manifest(data_dir, existing_dir, cache_dir, settings)
        constituents = read_constituents(data_dir / settings["constituent_file"])
        bond = read_bond(data_dir / settings["bond_file"], settings["trading_days_per_year"])
        if mode == "audit":
            returns = read_daily_matrix(existing_dir / "full_return_matrix.csv", True)
            turnover = read_daily_matrix(existing_dir / "full_turnover_matrix.csv", True)
        else:
            returns, turnover = matrices_from_cache(constituents, cache_dir, settings, allow_network)
        windows = build_windows(returns.index, settings["estimation_days"], settings["holding_days"], settings["step_days"])
        if windows.empty:
            raise ValueError("No complete rolling windows")
        universe, parameters, assets, master = calibrate_windows(returns, windows, constituents, settings)
        audit, legacy_rf, causal_rf = audit_inputs(returns, turnover, windows, universe, parameters, master,
                                                  bond, existing_dir, constituents, cache_dir, settings, mode)
        alignment = pd.DataFrame({
            "legacy_rf": legacy_rf.rf_daily, "legacy_source_date": legacy_rf.source_date,
            "legacy_used_future_value": legacy_rf.used_future_value,
            "causal_rf": causal_rf.rf_daily, "causal_source_date": causal_rf.source_date,
        })
        frames = {
            "windows_master_40.csv": master, "input_data_audit.csv": audit, "universe_asset_audit.csv": assets,
            "endpoint_calibration.csv": parameters, "risk_free_alignment_audit.csv": alignment,
            "implementation_parameters.csv": implementation_parameters(settings, mode, data_dir, existing_dir, cache_dir),
        }
        for name, frame in frames.items():
            safe_to_csv(frame, out_dir / name, overwrite=overwrite, index=name == "risk_free_alignment_audit.csv")
        if generate_bootstrap:
            arrays = bootstrap_window_indices(windows, returns, settings)
            np.savez_compressed(out_dir / "bootstrap_indices.npz", **arrays)
            LOGGER.info("Saved synchronized bootstrap inputs: %s arrays", len(arrays))
        failed = audit.loc[audit.status == "failed"]
        if mode == "build" and failed.empty:
            core_frames = {
                "full_return_matrix.csv": returns, "full_turnover_matrix.csv": turnover,
                "risk_free_rate.csv": causal_rf[["rf_daily"]].rename(columns={"rf_daily": "rf"}),
                "rolling_windows.csv": windows, "windows_universe.csv": universe,
                "fuzzy_parameters.csv": parameters.loc[parameters.endpoint_scheme == "baseline",
                                                       ["window_id", "stock_code", *list("abcd"), "mean", "obs"]],
            }
            for name, frame in core_frames.items():
                safe_to_csv(frame, out_dir / name, overwrite=overwrite, index=name in CORE_FILES[:3])
        metadata["status"] = "failed_checks" if not failed.empty else "audit_pass_with_review_items"
        metadata["audit_counts"] = audit.status.value_counts().to_dict()
        metadata["completed_windows"] = len(windows)
        metadata["eligible_asset_counts"] = {"min": int(master.eligible_assets.min()), "max": int(master.eligible_assets.max())}
        if any(sha256_file(item["path"]) != item["sha256"] for item in metadata["inputs"]):
            raise RuntimeError("An input changed during stage1; results are not frozen")
        if allow_network:
            metadata["inputs_after_download"] = input_manifest(data_dir, existing_dir, cache_dir, settings)
        metadata["outputs"] = [{"path": str(out_dir / n), "sha256": sha256_file(out_dir / n)}
                               for n in names if n not in {"step1_run_metadata.json", "step1_run.log"} and (out_dir / n).is_file()]
        LOGGER.info("Completed: %s", metadata["audit_counts"])
    except Exception as exc:
        metadata["status"] = "error"
        metadata["error"] = f"{type(exc).__name__}: {exc}"
        LOGGER.exception("Stage1 failed")
        raise
    finally:
        metadata["finished_utc"] = datetime.now(timezone.utc).isoformat()
        metadata["elapsed_seconds"] = time.perf_counter() - started
        (out_dir / "step1_run_metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
        LOGGER.removeHandler(handler)
        handler.close()
        register_run(out_dir, metadata)
    return metadata


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["audit", "build"], default="build")
    parser.add_argument("--data", type=Path, default=BASE_DIR / "data")
    parser.add_argument("--results", type=Path, default=BASE_DIR / "results1",
                        help="Results/cache directory; also the default build output")
    parser.add_argument("--out", type=Path, help="Audit/build output; defaults to --results")
    parser.add_argument("--cache", type=Path)
    parser.add_argument("--config", type=Path, help="JSON settings or previous run metadata JSON")
    parser.add_argument("--experiment-id", default="BASELINE_BUILD")
    parser.add_argument("--allow-network", action="store_true")
    parser.add_argument("--overwrite", action="store_true", help="Replace only named stage1 output artifacts")
    parser.add_argument("--skip-bootstrap", action="store_true", help="Audit without bootstrap inputs; recorded in metadata")
    args = parser.parse_args(argv)
    settings = load_settings(args.config)
    metadata = run_stage1(args.data, args.results, args.out or args.results, settings, args.mode,
                          args.cache, args.allow_network, args.overwrite, not args.skip_bootstrap, args.experiment_id)
    print(f"Stage1: {metadata['status']}; {metadata['completed_windows']} windows; {metadata['audit_counts']}")
    return 2 if metadata["status"] == "failed_checks" else 0


if __name__ == "__main__":
    sys.exit(main())