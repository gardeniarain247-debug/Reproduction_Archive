"""Candidate publication-size figures for the accepted model (visual changes only).

The figure set follows the manuscript-level empirical narrative:
1) risk-measure substitution,
2) shape-parameter response,
3) allocation mechanism,
4) market performance,
with reviewer-requested supplements kept in separate appendix figures.

Outputs use concise manuscript-facing filenames rather than long code-style names.
"""
import argparse
from pathlib import Path
from datetime import datetime, timezone
import json
import re
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import step1
from step3 import checked_manifest, prepare_output, finish_metadata

BASE_DIR = Path(__file__).resolve().parent


def _safe_read_csv(path):
    path = Path(path)
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def _pair_k(model_id):
    """Resolve shape from the scenario suffix; avoid CVaR05/endpoint numeric collisions."""
    if pd.isna(model_id):
        return np.nan
    mid = str(model_id)
    if 'neutral_k1' in mid:
        return 1.0
    suffix = re.search(r'(?:_k|_fixed_k|_kpure)(05|08|11)(?:_|$)', mid)
    if suffix:
        return {'05': 0.5, '08': 0.8, '11': 1.1}[suffix.group(1)]
    if re.search(r'(?:^|_)opt(?:_|$)', mid):
        return 0.5
    if re.search(r'(?:^|_)neu(?:_|$)', mid):
        return 0.8  # legacy program ID; paper calls this Mid.
    if re.search(r'(?:^|_)pes(?:_|$)', mid):
        return 1.1
    return np.nan


def _scenario_label(v):
    return {0.5: 'Opt.', 0.8: 'Mid.', 1.0: 'Neu.', 1.1: 'Cons.'}.get(float(v), f'k={v:g}') if np.isfinite(v) else 'Unknown'


def _feasibility_wide(frame):
    """Pivot official long-format Step4 output; never discard experiment identifiers."""
    required = {'experiment_id', 'model_id', 'solver_status', 'windows'}
    if not required.issubset(frame.columns):
        raise ValueError(f'Step4 feasibility requires columns {sorted(required)}')
    raw = frame.copy()
    raw['windows'] = pd.to_numeric(raw['windows'], errors='coerce')
    if raw['windows'].isna().any() or (raw['windows'] < 0).any() or (raw['windows'] % 1 != 0).any():
        raise ValueError('Invalid feasibility counts')
    raw = raw.loc[~raw['model_id'].astype(str).str.endswith('_EMEW')].copy()
    if not raw['solver_status'].isin(['Direct', 'Relaxed', 'Failed']).all():
        raise ValueError('Unknown solver status in feasibility results')
    z = raw.pivot_table(index=['experiment_id', 'model_id'], columns='solver_status', values='windows', aggfunc='sum', fill_value=0)
    z = z.reindex(columns=['Direct','Relaxed','Failed'], fill_value=0)
    z = z.loc[~z.index.get_level_values('model_id').str.endswith('_EMEW')].copy()
    z['total'] = z.sum(axis=1)
    if (z.total != 40).any():
        examples=z[z.total != 40].head().to_dict('index')
        raise ValueError(f'Expected 40 windows per strategy: {examples}')
    return z


def _draw_status(frame, selection, title, stem, save):
    rows=[]
    for exp,mid,label in selection:
        key=(exp,mid)
        if key not in frame.index:
            raise ValueError(f'Missing source strategy for {stem}: {key}')
        rows.append((label,frame.loc[key].reindex(['Direct','Relaxed','Failed']).astype(float).to_numpy()))
    if not rows:
        return
    fig,ax=plt.subplots(figsize=(6.3, max(3.3, .41*len(rows)+1.35)))
    names=[x[0] for x in rows];vals=np.stack([x[1] for x in rows]); y=np.arange(len(names))
    left=np.zeros(len(names))
    for i,(status,color) in enumerate([('Direct','#287D6B'),('Relaxed','#E9B44C'),('Failed','#C44E52')]):
        ax.barh(y,vals[:,i],left=left,color=color,label=status,height=.72)
        left+=vals[:,i]
    ax.set_yticks(y,names,fontsize=10.5);ax.invert_yaxis()
    ax.set_xlim(0,40);ax.set_xticks([0,10,20,30,40]);ax.set_xlabel('Rebalancing windows (out of 40)')
    ax.set_title(title,fontsize=11.5,pad=25);ax.legend(loc='upper center',bbox_to_anchor=(.5,1.10),ncol=3,frameon=False,fontsize=10)
    ax.grid(axis='x',alpha=.15)
    save(fig,stem,title)

def _make_heatmap(ax, data, row_labels, col_labels, title, fmt='{:.2f}', cmap='RdBu_r', center=0.0):
    arr = np.asarray(data, dtype=float)
    vmax = np.nanmax(np.abs(arr - center)) if np.isfinite(arr).any() else 1.0
    if vmax == 0:
        vmax = 1.0
    im = ax.imshow(arr, cmap=cmap, vmin=center - vmax, vmax=center + vmax, aspect='auto')
    ax.set_xticks(np.arange(len(col_labels)), col_labels)
    ax.set_yticks(np.arange(len(row_labels)), row_labels)
    ax.set_title(title)
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            val = arr[i, j]
            txt = 'NA' if not np.isfinite(val) else fmt.format(val)
            ax.text(j, i, txt, ha='center', va='center', fontsize=10)
    return im


def run(stage3_dir, stage4_dir, out_dir, overwrite=False, allow_pilot=False):
    m3 = checked_manifest(stage3_dir, 'step3_run_metadata.json', 'accepted_modelA_stage3')
    m4 = checked_manifest(stage4_dir, 'step4_run_metadata.json', 'accepted_modelA_stage4')
    if m4.get('pipeline_revision') != 'review_completion_v1':
        raise ValueError('Need updated consolidated inference')
    if m3.get('pipeline_model') != 'ModelA' or m4.get('pipeline_model') != 'ModelA':
        raise ValueError('Only the accepted single portfolio model is supported')
    if m4['source_stage3_metadata_sha256'] != step1.sha256_file(Path(stage3_dir) / 'step3_run_metadata.json'):
        raise ValueError('Stage4 and Stage3 provenance mismatch')
    if m4['scope'] != 'full' and not allow_pilot:
        raise ValueError('Pilot requires --allow-pilot')

    out = prepare_output(out_dir, 'step5_', overwrite)
    d = pd.read_parquet(Path(stage3_dir) / 'step3_daily_returns.parquet')
    d['date'] = pd.to_datetime(d['date'])
    w = _safe_read_csv(Path(stage3_dir) / 'step3_window_backtests.csv')
    perf = _safe_read_csv(Path(stage4_dir) / 'step4_performance.csv')
    pairs = _safe_read_csv(Path(stage4_dir) / 'step4_paired_comparisons.csv')
    struct_summary = _safe_read_csv(Path(stage4_dir) / 'step4_structure_summary.csv')
    struct_by_window = _safe_read_csv(Path(stage4_dir) / 'step4_structure_by_window.csv')
    state_perf = _safe_read_csv(Path(stage4_dir) / 'step4_state_performance.csv')
    hetero = _safe_read_csv(Path(stage4_dir) / 'step4_heterogeneity_mechanisms.csv')
    neutral = _safe_read_csv(Path(stage4_dir) / 'step4_neutral_reference.csv')
    common_samples = _safe_read_csv(Path(stage4_dir) / 'step4_substitution_common_samples.csv')
    state_contrasts = _safe_read_csv(Path(stage4_dir) / 'step4_state_contrasts.csv')
    feasibility = _safe_read_csv(Path(stage4_dir) / 'step4_feasibility.csv')

    plt.rcParams.update({
        'font.family': 'DejaVu Sans',
        'font.size': 11,
        'axes.titlesize': 11.5,
        'axes.labelsize': 11,
        'xtick.labelsize': 10.5,
        'ytick.labelsize': 10.5,
        'legend.fontsize': 10.5,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'pdf.fonttype': 42,
        'figure.dpi': 150,
    })

    outputs = []
    figure_index = []

    def save(fig, stem, section_note):
        if stem == 'fig_alloc':
            fig.tight_layout(rect=(0.02, 0.075, 1, 0.985), pad=1.1, h_pad=2.2, w_pad=2.2)
        elif stem == 'fig_replace':
            fig.tight_layout(pad=1.2, h_pad=3.8, w_pad=2.8)
        elif stem == 'fig_paths':
            fig.tight_layout(rect=(0, 0, 1, 0.95), pad=1.2, h_pad=1.45, w_pad=1.45)
        else:
            fig.tight_layout(pad=1.2, h_pad=1.8, w_pad=1.6)
        for ext in ['png', 'pdf']:
            p = out / f'{stem}.{ext}'
            fig.savefig(p, bbox_inches='tight', dpi=180)
            outputs.append(p)
        figure_index.append({'file_stem': stem, 'section_role': section_note})
        plt.close(fig)

    # ----- Figure: paths -----
    fig, axes = plt.subplots(3, 2, figsize=(6.3, 7.15), sharex='col')
    scen_rows = [(0.5, 'opt', 'CVaR_opt', 'RVaR_opt', 'CVaR02_k05'),
                 (0.8, 'neu', 'CVaR_neu', 'RVaR_neu', 'CVaR02_k08'),
                 (1.1, 'pes', 'CVaR_pes', 'RVaR_pes', 'CVaR02_k11')]
    colors = {'CVaR': '#355C7D', 'RVaR': '#D95F02', 'CVaR20': '#1B9E77'}
    import matplotlib.dates as mdates
    for i, (k, label, cvar_id, rvar_id, cvar20_id) in enumerate(scen_rows):
        row_series = [
            ('BASELINE_AUDIT', cvar_id, colors['CVaR'], 'CVaR .05'),
            ('BASELINE_AUDIT', rvar_id, colors['RVaR'], 'RVaR'),
            ('CVAR02_ROBUSTNESS', cvar20_id, colors['CVaR20'], 'CVaR .20'),
        ]
        for exp_id, model_id, color, disp in row_series:
            g = d[(d['experiment_id'] == exp_id) & (d['model_id'] == model_id)].sort_values('date')
            if g.empty:
                continue
            axes[i, 0].plot(g['date'], g['nav'], lw=1.35, color=color, label=disp)
            nav = np.r_[1.0, g['nav'].to_numpy(dtype=float)]
            dd = (1 - nav / np.maximum.accumulate(nav))[1:] * 100
            axes[i, 1].plot(g['date'], dd, lw=1.2, color=color, label=disp)
        axes[i, 0].set_title(f'{_scenario_label(k)}: cumulative NAV')
        axes[i, 1].set_title(f'{_scenario_label(k)}: drawdown (%)')
        if i == 0:
            handles, labels = axes[i, 0].get_legend_handles_labels()
            fig.legend(handles, labels, loc='upper center', ncol=3,
                       bbox_to_anchor=(0.5, 0.985), frameon=False, fontsize=9.5)
    for ax in axes[-1, :]:
        ax.xaxis.set_major_locator(mdates.YearLocator(base=2))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        ax.tick_params(axis='x', rotation=0)
    save(fig, 'fig_paths', 'Main text: cumulative performance and drawdowns')

    # ----- Figure: substitution -----
    g = pairs[(pairs['purpose'] == 'rvar_cvar') &
              (pairs['scope'] == 'full_policy') &
              (pairs['left_experiment'] == 'BASELINE_AUDIT')].copy()
    if not g.empty:
        g['k'] = g['left_model'].map(_pair_k)
        g = g.sort_values('k')
        k_labels = [_scenario_label(v) for v in g['k']]

        # exposure gaps and effective-n gaps for mechanism scatter
        ss = struct_summary.copy()
        exp_gap, eff_gap = [], []
        for _, row in g.iterrows():
            left = ss[ss['model_id'] == row['left_model']]
            right = ss[ss['model_id'] == row['right_model']]
            if left.empty or right.empty:
                exp_gap.append(np.nan)
                eff_gap.append(np.nan)
            else:
                exp_gap.append(float(left['mean_risky_amount'].iloc[0] - right['mean_risky_amount'].iloc[0]))
                eff_gap.append(float(left['mean_effective_n'].iloc[0] - right['mean_effective_n'].iloc[0]))
        fig, axes = plt.subplots(2, 2, figsize=(6.3, 7.6))
        x = np.arange(len(g))
        width = 0.35
        axes[0, 0].bar(x - width / 2, 100 * g['delta_annual_return'], width=width, label='Return diff.', color='#D95F02')
        axes[0, 0].bar(x + width / 2, -100 * g['delta_max_drawdown'], width=width, label='MDD improv.', color='#355C7D')
        axes[0, 0].axhline(0, color='gray', lw=1)
        axes[0, 0].set_xticks(x, k_labels)
        axes[0, 0].set_ylabel('Percentage points')
        axes[0, 0].set_title('Return and drawdown gaps', fontsize=10.5, pad=30)
        axes[0, 0].legend(loc='upper left', bbox_to_anchor=(0, 1.01), ncol=2, frameon=False, fontsize=8.0, borderaxespad=0)

        # bootstrap intervals
        y = np.arange(len(g))
        lo = 100 * g['annual_return_ci_low'].to_numpy(dtype=float)
        hi = 100 * g['annual_return_ci_high'].to_numpy(dtype=float)
        pt = 100 * g['delta_annual_return'].to_numpy(dtype=float)
        for yy, ll, hh, pp in zip(y, lo, hi, pt):
            axes[0, 1].plot([ll, hh], [yy, yy], color='#355C7D', lw=2)
            axes[0, 1].plot(pp, yy, 'o', color='#D95F02')
        axes[0, 1].axvline(0, color='gray', lw=1)
        axes[0, 1].set_yticks(y, k_labels)
        axes[0, 1].set_xlabel('Annual return difference (pp)')
        axes[0, 1].set_title('Bootstrap 95% intervals', fontsize=10.5, pad=10)

        axes[1, 0].bar(x - width / 2, g['delta_sharpe'], width=width, label='Sharpe diff.', color='#D95F02')
        axes[1, 0].bar(x + width / 2, -100 * g['delta_ES95'], width=width, label='ES95 improv.', color='#355C7D')
        axes[1, 0].axhline(0, color='gray', lw=1)
        axes[1, 0].set_xticks(x, k_labels)
        axes[1, 0].set_title('Sharpe and ES95 gaps', fontsize=10.5, pad=30)
        axes[1, 0].legend(loc='upper left', bbox_to_anchor=(0, 1.01), ncol=2, frameon=False, fontsize=8.0, borderaxespad=0)

        axes[1, 1].axhline(0, color='gray', lw=1)
        axes[1, 1].axvline(0, color='gray', lw=1)
        axes[1, 1].scatter(100 * np.array(exp_gap), 100 * g['delta_annual_return'], s=110, color='#287D6B')
        for xx, yy, lab, dg in zip(100 * np.array(exp_gap), 100 * g['delta_annual_return'], k_labels, eff_gap):
            note = f'{lab}\nΔN={dg:.2f}' if np.isfinite(dg) else lab
            offsets = {'Opt.': (-8, -24), 'Mid.': (7, -27), 'Cons.': (-35, 12)}
            axes[1, 1].annotate(note, (xx, yy), xytext=offsets.get(lab, (4, 4)),
                                textcoords='offset points', fontsize=8.5)
        axes[1, 1].set_xlabel('Mean risky-exposure gap (pp)')
        axes[1, 1].set_ylabel('Annual return difference (pp)')
        axes[1, 1].set_title('Allocation versus return', fontsize=10.5, pad=10)
        save(fig, 'fig_replace', 'Main text: substitution effect and uncertainty')

    # ----- Figure: shape response (fixed rule) -----
    gf = perf[perf['experiment_id'] == 'FIXED_RULE_CONTROL'].copy()
    if not gf.empty:
        gf['k'] = gf['model_id'].str.extract(r'k(\d+)')[0].astype(float) / 10
        gf['risk'] = np.where(gf['model_id'].str.contains('RVaR'), 'RVaR', 'CVaR')
        gf = gf[~gf['model_id'].str.endswith('_EMEW')].sort_values(['risk', 'k'])
        fig, axes = plt.subplots(2, 2, figsize=(6.3, 5.8))
        metrics = [
            ('annual_return', 'Annual return (%)', 100),
            ('sharpe', 'Sharpe ratio', 1),
            ('mean_risky_exposure', 'Mean risky exposure', 1),
            ('max_drawdown', 'Max drawdown (%)', 100),
        ]
        for ax, (col, title, scale) in zip(axes.flat, metrics):
            for risk, color in [('CVaR', '#355C7D'), ('RVaR', '#D95F02')]:
                z = gf[gf['risk'] == risk].sort_values('k')
                ax.plot(z['k'], scale * z[col], 'o-', lw=1.5, ms=5, label=risk, color=color)
            ax.set_title(title)
            ax.set_xlabel('Common shape parameter k')
            if title == 'Sharpe ratio':
                ax.legend(frameon=False)
        save(fig, 'fig_shape', 'Main text: shape-parameter response under fixed rules')

    # ----- Figure: allocation mechanism -----
    sbw = struct_by_window.copy()
    if not sbw.empty:
        main_ids = ['CVaR_opt', 'RVaR_opt', 'CVaR_neu', 'RVaR_neu', 'CVaR_pes', 'RVaR_pes']
        order = main_ids
        label_map = {
            'CVaR_opt': 'C-Opt', 'RVaR_opt': 'R-Opt',
            'CVaR_neu': 'C-Mid', 'RVaR_neu': 'R-Mid',
            'CVaR_pes': 'C-Cons', 'RVaR_pes': 'R-Cons',
        }
        sbw = sbw[sbw['experiment_id'] == 'BASELINE_AUDIT']
        sbw = sbw[sbw['model_id'].isin(main_ids)].copy()
        sbw['label'] = pd.Categorical(sbw['model_id'].map(label_map),
                                      categories=[label_map[x] for x in order], ordered=True)
        pmain = perf[(perf['experiment_id'] == 'BASELINE_AUDIT') & (perf['model_id'].isin(main_ids))].copy()
        pmain['label'] = pd.Categorical(pmain['model_id'].map(label_map), categories=[label_map[x] for x in order], ordered=True)
        ssum = struct_summary[(struct_summary['experiment_id'] == 'BASELINE_AUDIT') & (struct_summary['model_id'].isin(main_ids))].copy()
        ssum['label'] = pd.Categorical(ssum['model_id'].map(label_map), categories=[label_map[x] for x in order], ordered=True)

        fig, axes = plt.subplots(2, 2, figsize=(6.3, 7.1))
        short_tick_labels = ['C-\nOpt', 'R-\nOpt', 'C-\nMid', 'R-\nMid', 'C-\nCons', 'R-\nCons']
        # boxplot risky exposure
        box_data = [sbw.loc[sbw['label'] == label_map[mid], 'risky_amount'].dropna().to_numpy() for mid in order]
        axes[0, 0].boxplot(box_data, labels=[label_map[mid] for mid in order], showfliers=False)
        axes[0, 0].set_xticklabels(short_tick_labels, fontsize=9, rotation=0)
        axes[0, 0].set_title('Risky exposure', fontsize=10.5, pad=10)

        box_data = [sbw.loc[sbw['label'] == label_map[mid], 'effective_n'].dropna().to_numpy() for mid in order]
        axes[0, 1].boxplot(box_data, labels=[label_map[mid] for mid in order], showfliers=False)
        axes[0, 1].set_xticklabels(short_tick_labels, fontsize=9, rotation=0)
        axes[0, 1].set_title('Effective asset count', fontsize=10.5, pad=10)

        axes[1, 0].bar(pmain.sort_values('label')['label'].astype(str), 100 * pmain.sort_values('label')['mean_turnover'], color='#287D6B')
        axes[1, 0].set_xticks(np.arange(len(short_tick_labels)), labels=short_tick_labels, fontsize=9, rotation=0)
        axes[1, 0].set_title('Turnover per rebalance (%)', fontsize=10.5, pad=10)
        # Step3: 0.5*sum(abs(posttrade_amount - initial_weight)), with
        # both sides normalized by pre-trade portfolio NAV; Step4 averages
        # this window-level statistic without annualization.

        # Group labels for near-overlapping CVaR/RVaR points; keep all observations.
        from matplotlib.transforms import Bbox
        ax = axes[1, 1]
        point_data = []
        for _, record in ssum.iterrows():
            point_data.append((str(record['model_id']), float(record['mean_hhi']),
                               float(record['mean_max_normalized_weight'])))
        for mid, xx, yy in point_data:
            ax.scatter(xx, yy, s=58,
                       color='#355C7D' if mid.startswith('CVaR') else '#D95F02', zorder=3)
        # Place annotations at fixed, separated offsets for nearly coincident pairs.
        # The offsets affect annotation geometry only, never the underlying data.
        annotation_offsets = {
            'CVaR_opt': (6, -12), 'RVaR_opt': (6, 8),
            'CVaR_neu': (6, -12), 'RVaR_neu': (6, 8),
            'CVaR_pes': (-45, -12), 'RVaR_pes': (-45, 8),
        }
        for mid, xx, yy in point_data:
            ax.annotate(label_map[mid], (xx, yy),
                        xytext=annotation_offsets[mid], textcoords='offset points',
                        fontsize=8.5, annotation_clip=False)
        if point_data:
            xs = [p[1] for p in point_data]; ys = [p[2] for p in point_data]
            ax.set_xlim(min(xs)-0.045, max(xs)+0.115)
            ax.set_ylim(min(ys)-0.065, max(ys)+0.065)
        ax.set_xlabel('Mean risky-sleeve HHI', fontsize=9.5)
        ax.set_ylabel('Largest risky weight', fontsize=9.5, labelpad=1)
        ax.set_title('Concentration', fontsize=10.5, pad=10)
        fig.text(.5, .017, 'C = CVaR .05; R = RVaR; Opt/Mid/Cons = shape settings.',
                 ha='center', fontsize=9)
        save(fig, 'fig_alloc', 'Main text: allocation mechanism and concentration')

    # ----- Figure: states -----
    sp = state_perf.copy()
    if not sp.empty:
        sp['k'] = sp['left_model'].map(_pair_k)
        sp = sp.sort_values(['state', 'k'])
        state_order = ['Up', 'Down', 'HighVol', 'LowVol', 'Pre2021', 'Post2021']
        k_order = [0.5, 0.8, 1.1]
        fig, axes = plt.subplots(1, 3, figsize=(6.3, 3.6), sharey=True)
        panels = [
            ('delta_annual_return', 'Return gap (pp)', 100),
            ('delta_sharpe', 'Sharpe gap', 1),
            ('delta_max_drawdown', 'MDD improvement (pp)', -100),
        ]
        for ax, (col, title, scale) in zip(axes, panels):
            mat = []
            for st in state_order:
                row = []
                for kk in k_order:
                    sub = sp[(sp['state'] == st) & (np.isclose(sp['k'], kk))]
                    if sub.empty:
                        row.append(np.nan)
                    else:
                        row.append(scale * float(sub.iloc[0][col]))
                mat.append(row)
            im = _make_heatmap(ax, mat, state_order, [_scenario_label(k) for k in k_order], title, fmt='{:.2f}')
        # Each panel has its own numeric scale. All cells carry values, so an
        # overlapping colorbar adds no information and obstructs the MDD panel.
        # Explicit titles and numerical annotations preserve interpretation.
        save(fig, 'fig_states', 'Main text: state-conditional diagnostics')

    # ----- Appendix figure: neutral k=1 -----
    gn = neutral[(neutral['purpose'] == 'rvar_cvar') & (neutral['left_experiment'] == 'K1_NEUTRAL')].copy()
    if not gn.empty:
        scopes = ['full_policy', 'both_direct_only']
        scope_labels = {'full_policy': 'Full policy', 'both_direct_only': 'Direct-only'}
        fig, axes = plt.subplots(1, 3, figsize=(6.3, 3.4))
        metrics = [
            ('delta_annual_return', 'Annual return (pp)', 100),
            ('delta_sharpe', 'Sharpe diff', 1),
            ('delta_ES95', 'ES95 improvement (pp)', -100),
        ]
        for ax, (col, title, scale) in zip(axes, metrics):
            vals = [scale * float(gn.loc[gn['scope'] == s, col].iloc[0]) if not gn.loc[gn['scope'] == s].empty else np.nan for s in scopes]
            ax.bar([scope_labels[s] for s in scopes], vals, color=['#D95F02', '#355C7D'])
            ax.axhline(0, color='gray', lw=1)
            ax.set_title(title)
            ax.tick_params(axis='x', rotation=20)
        save(fig, 'app_neutral', 'Appendix: separate k=1 reference')

    # ----- Appendix figure: sensitivity -----
    extra = perf[perf['experiment_id'].isin(['ENDPOINT_ALT', 'RVAR_INTERVAL', 'CVAR02_ROBUSTNESS'])].copy()
    if not extra.empty:
        fig, axes = plt.subplots(2, 2, figsize=(6.3, 6.0))
        # endpoint alternatives for RVaR
        ge = perf[perf['experiment_id'] == 'ENDPOINT_ALT'].copy()
        if not ge.empty:
            ge = ge[(ge['model_id'].str.contains('RVaR_endpoint')) & (~ge['model_id'].str.endswith('_EMEW'))]
            ge['k'] = ge['model_id'].str.extract(r'k(\d+)')[0].astype(float) / 10
            ge['scheme'] = ge['model_id'].str.extract(r'endpoint_(baseline|alternative_a|alternative_b)')[0]
            for scheme, mk in [('baseline', 'o-'), ('alternative_a', 's--'), ('alternative_b', 'd:')]:
                z = ge[ge['scheme'] == scheme].sort_values('k')
                axes[0, 0].plot(z['k'], 100 * z['annual_return'], mk, label=scheme)
                axes[0, 1].plot(z['k'], z['sharpe'], mk, label=scheme)
            axes[0, 0].set_title('Endpoint alternatives: annual return (%)')
            axes[0, 1].set_title('Endpoint alternatives: Sharpe ratio')
            axes[0, 0].legend(loc='upper left', bbox_to_anchor=(0, 1.01), ncol=2, frameon=False, fontsize=8.0, borderaxespad=0)
        # interval alternatives for RVaR
        gi = perf[perf['experiment_id'] == 'RVAR_INTERVAL'].copy()
        if not gi.empty:
            gi = gi[(gi['model_id'].str.contains('RVaR_interval')) & (~gi['model_id'].str.endswith('_EMEW'))]
            gi['k'] = gi['model_id'].str.extract(r'k(\d+)')[0].astype(float) / 10
            gi['scheme'] = gi['model_id'].str.extract(r'interval_(baseline|a|b)_')[0]
            for scheme, mk in [('baseline', 'o-'), ('a', 's--'), ('b', 'd:')]:
                z = gi[gi['scheme'] == scheme].sort_values('k')
                axes[1, 0].plot(z['k'], 100 * z['annual_return'], mk, label=scheme)
                axes[1, 1].plot(z['k'], z['sharpe'], mk, label=scheme)
            axes[1, 0].set_title('RVaR interval alternatives: annual return (%)')
            axes[1, 1].set_title('RVaR interval alternatives: Sharpe ratio')
            axes[1, 0].legend(loc='upper left', bbox_to_anchor=(0, 1.01), ncol=2, frameon=False, fontsize=8.0, borderaxespad=0)
        for ax in axes.flat:
            ax.set_xlabel('k')
        save(fig, 'app_sensitivity', 'Appendix: endpoint and interval sensitivity')

    # ----- Appendix figure: heterogeneity -----
    gh = hetero.copy()
    if not gh.empty:
        gh['scenario'] = gh['left_model'].str.extract(r'heterog_(control|partial_10|partial_20|all_10|all_20)')[0]
        order = ['partial_10', 'partial_20', 'all_10', 'all_20']
        names = ['Part .1','Part .2','All .1','All .2']
        fig, axes = plt.subplots(1, 2, figsize=(6.3, 3.1))
        box_data = [gh.loc[gh['scenario'] == s, 'weight_l1'].dropna().to_numpy() for s in order]
        axes[0].boxplot(box_data, labels=names, showfliers=False)
        axes[0].tick_params(axis='x', rotation=0)
        axes[0].set_title('Heterogeneous k: allocation distance')
        means = gh.groupby('scenario')['mean_abs_k_return_difference'].mean().reindex(order)
        axes[1].bar(names, means.values, color='#287D6B')
        axes[1].tick_params(axis='x', rotation=0)
        axes[1].set_title('Average return-side shape gap')
        save(fig, 'app_heterogeneity', 'Appendix: heterogeneous shape scenarios')

    # ----- Appendix feasibility: exact experiment + model identifiers -----
    if not feasibility.empty:
        fw = _feasibility_wide(feasibility)
        core = []
        for code,lab in [('opt','Opt.'),('neu','Mid.'),('pes','Cons.')]:
            core += [('BASELINE_AUDIT','CVaR_'+code,'CVaR 5% / '+lab),
                     ('BASELINE_AUDIT','RVaR_'+code,'RVaR / '+lab)]
        for suf,lab in [('05','Opt.'),('08','Mid.'),('11','Cons.')]:
            core += [('CVAR02_ROBUSTNESS','CVaR02_k'+suf,'CVaR 20% / '+lab)]
        for suf,lab in [('05','Opt.'),('08','Mid.'),('11','Cons.')]:
            core += [('FIXED_RULE_CONTROL','CVaR_fixed_k'+suf,'CVaR fixed / '+lab),
                     ('FIXED_RULE_CONTROL','RVaR_fixed_k'+suf,'RVaR fixed / '+lab)]
        core += [('K1_NEUTRAL','CVaR_neutral_k1','CVaR / Neu.'),
                 ('K1_NEUTRAL','RVaR_neutral_k1','RVaR / Neu.'),
                 ('BASELINE_AUDIT','MV','MV reference')]
        _draw_status(fw,core,'Main, fixed-rule and neutral optimization status','app_feasibility_core',save)

        endpoint=[]
        for risk in ('CVaR','RVaR'):
            for scheme,name in [('baseline','Base'),('alternative_a','A'),('alternative_b','B')]:
                for suf,lab in [('05','Opt.'),('08','Mid.'),('11','Cons.')]:
                    endpoint.append(('ENDPOINT_ALT',f'{risk}_endpoint_{scheme}_k{suf}',f'{risk} {name} / {lab}'))
        _draw_status(fw,endpoint,'Endpoint-calibration feasibility','app_feasibility_endpoint',save)

        intervals=[]
        for scheme,name in [('baseline','Baseline [.05,.20]'),('a','A [.05,.15]'),('b','B [.10,.20]')]:
            for suf,lab in [('05','Opt.'),('08','Mid.'),('11','Cons.')]:
                intervals.append(('RVAR_INTERVAL',f'RVaR_interval_{scheme}_k{suf}',f'{name} / {lab}'))
        _draw_status(fw,intervals,'RVaR interval feasibility','app_feasibility_interval',save)

        heter=[]
        for risk in ('CVaR','RVaR'):
            for scheme,name in [('control','Common'),('partial_10','Part .1'),('partial_20','Part .2'),('all_10','All .1'),('all_20','All .2')]:
                heter.append(('K_HETEROG',f'{risk}_heterog_{scheme}',f'{risk} / {name}'))
        _draw_status(fw,heter,'Heterogeneous-shape feasibility','app_feasibility_heterogeneity',save)

    # Legacy compatibility: only publish specific manuscript-facing PDF names.
    # 'app_feasibility' old single crowded figure is deliberately retired.

    # figure index for the manuscript drafting stage
    (out / 'figure_index.json').write_text(json.dumps(figure_index, indent=2), encoding='utf-8')
    outputs.append(out / 'figure_index.json')

    meta = dict(
        stage=5,
        status='accepted_modelA_stage5',
        pipeline_model='ModelA',
        scope=m4['scope'],
        started_utc=datetime.now(timezone.utc).isoformat(),
        code_sha256=step1.sha256_file(__file__),
        source_stage3_metadata_sha256=step1.sha256_file(Path(stage3_dir) / 'step3_run_metadata.json'),
        source_stage4_metadata_sha256=step1.sha256_file(Path(stage4_dir) / 'step4_run_metadata.json'),
        figure_scope='single-model manuscript narrative: substitution, shape response, allocation mechanism, and market performance',
    )
    return finish_metadata(out, 'step5_run_metadata.json', meta, outputs)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage3', type=Path, default=BASE_DIR / 'results3')
    p.add_argument('--stage4', type=Path, default=BASE_DIR / 'results4')
    p.add_argument('--out', type=Path, default=BASE_DIR / 'results5')
    p.add_argument('--overwrite', action='store_true')
    p.add_argument('--allow-pilot', action='store_true')
    a = p.parse_args(argv)
    print(run(a.stage3, a.stage4, a.out, a.overwrite, a.allow_pilot)['status'])


if __name__ == '__main__':
    run(
        stage3_dir=Path(r'D:\latex\IJFS\results_execution\results3'),
        stage4_dir=Path(r'D:\latex\IJFS\results_execution\results4'),
        out_dir=Path(r'D:\latex\IJFS\results5_layout_trial'),
        overwrite=False
    )