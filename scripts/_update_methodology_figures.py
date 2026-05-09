"""One-shot helper: relocate Chapter 3 methodology figure-generation cells.

- Removes the S5 (Figure 3.1) and S6 (Figure 3.5) cells from evalaution.ipynb.
- Adds three new data-driven cells to Phase1_Preprocessing.ipynb that regenerate
  figures 3_1, 3_2, and 3_5 from the in-memory route_df / df / get_feature_names.
"""
from __future__ import annotations

import json
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EVAL_NB = ROOT / "notebooks" / "evalaution.ipynb"
PHASE1_NB = ROOT / "notebooks" / "Phase1_Preprocessing.ipynb"


def md_cell(text: str) -> dict:
    return {
        "cell_type": "markdown",
        "id": uuid.uuid4().hex[:12],
        "metadata": {},
        "source": text.splitlines(keepends=True),
    }


def code_cell(text: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "id": uuid.uuid4().hex[:12],
        "metadata": {},
        "outputs": [],
        "source": text.splitlines(keepends=True),
    }


# -------------------------------------------------------------- Figure 3.1
FIG_3_1_MD = """\
---
## Methodology Figure 3.1: End-to-End Research Framework

Saves `../figures/3_1.png` and `.pdf`. All counts are computed from the
preprocessed in-memory DataFrames so the figure stays consistent with the
processed data.
"""

FIG_3_1_CODE = """\
# Figure 3.1: End-to-End Research Framework Overview (data-driven)
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from utils.feature_engineering import get_feature_names

FIG_DIR_METH = Path('..') / 'figures'
FIG_DIR_METH.mkdir(exist_ok=True)

# Counts derived from the live data and feature config -------------------
n_raw      = int(n_before_dedup)        # raw segment-level rows (cell 5)
n_clean    = int(len(df))               # rows after preprocessing
n_trips    = int(len(route_df))         # trips after preprocessing
n_removed  = n_raw - n_clean
pct_removed = 100.0 * n_removed / n_raw if n_raw else 0.0
n_route_feat = len(get_feature_names('route'))
n_seg_feat   = len(get_feature_names('segment'))
n_routes   = int(route_df['route_short_name'].nunique())
# Calendar span Jul 29 - Sep 21 2024 = 55 days, including the two anomalous
# dates (Sep 3-4) that are excluded downstream. We report the full span here
# so the figure is consistent with the thesis abstract / Section 3.4.
from utils.temporal_splits import WEEK_BOUNDARIES as _WB
_first = min(pd.Timestamp(s) for s, _ in _WB.values())
_last  = max(pd.Timestamp(e) for _, e in _WB.values())
n_days = (_last - _first).days + 1

print(
    f'Counts -> raw={n_raw:,}  clean={n_clean:,}  '
    f'trips={n_trips:,}  removed={n_removed:,} ({pct_removed:.2f}%)  '
    f'route_feats={n_route_feat}  seg_feats={n_seg_feat}'
)

plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 11,
    'axes.titlesize': 13,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.15,
})

STAGE_COLORS = {
    'data':       '#4e79a7',
    'preprocess': '#59a14f',
    'features':   '#76b7b2',
    'model':      '#f28e2b',
    'conformal':  '#e15759',
    'segment':    '#b07aa1',
}
EXPERIMENT_COLORS = {'exp1': '#1f77b4', 'exp2': '#ff7f0e', 'exp3': '#d62728'}

fig, ax = plt.subplots(figsize=(14, 9))
ax.set_xlim(0, 14)
ax.set_ylim(0, 9)
ax.axis('off')

box_width  = 1.8
box_height = 1.2
y_main     = 7.0
x_positions = [0.5, 2.8, 5.1, 7.4, 9.7, 12.0]

stages = [
    ('1. Data\\nAcquisition &\\nExploration',       'data'),
    ('2. Preprocessing\\n& Cleaning',               'preprocess'),
    ('3. Feature\\nEngineering',                    'features'),
    ('4. Baseline\\nPoint Prediction\\n(XGBoost)',   'model'),
    ('5. Conformal\\nPrediction\\n(Uncertainty)',    'conformal'),
    ('6. Segment-Level\\nUncertainty\\nDecomposition', 'segment'),
]

for i, (label, color_key) in enumerate(stages):
    x = x_positions[i]
    ax.add_patch(FancyBboxPatch(
        (x, y_main), box_width, box_height,
        boxstyle='round,pad=0.1',
        facecolor=STAGE_COLORS[color_key], edgecolor='white',
        linewidth=2, alpha=0.9,
    ))
    ax.text(x + box_width/2, y_main + box_height/2, label,
            ha='center', va='center', fontsize=9, fontweight='bold',
            color='white', linespacing=1.3)

for i in range(len(x_positions) - 1):
    ax.annotate('', xy=(x_positions[i+1], y_main + box_height/2),
                xytext=(x_positions[i] + box_width, y_main + box_height/2),
                arrowprops=dict(arrowstyle='->', color='#333333',
                                lw=2, mutation_scale=15))

ax.text(x_positions[0] + box_width/2, y_main + box_height + 0.45,
        'Raw Bus GPS\\nTrajectories (GTFS)',
        ha='center', va='center', fontsize=8.5, fontstyle='italic', color='#555555',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='#f0f0f0',
                  edgecolor='#cccccc', linewidth=1))
ax.annotate('', xy=(x_positions[0] + box_width/2, y_main + box_height),
            xytext=(x_positions[0] + box_width/2, y_main + box_height + 0.2),
            arrowprops=dict(arrowstyle='->', color='#888888', lw=1.5))

ax.text(x_positions[5] + box_width/2, y_main + box_height + 0.45,
        'Uncertainty-Aware\\nETA Predictions',
        ha='center', va='center', fontsize=8.5, fontstyle='italic', color='#555555',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='#f0f0f0',
                  edgecolor='#cccccc', linewidth=1))
ax.annotate('', xy=(x_positions[5] + box_width/2, y_main + box_height + 0.2),
            xytext=(x_positions[5] + box_width/2, y_main + box_height),
            arrowprops=dict(arrowstyle='->', color='#888888', lw=1.5))

y_detail      = 5.2
detail_height = 0.9
details = [
    (x_positions[0], f'{n_raw:,} segment\\nrecords, {n_routes} routes,\\n{n_days} days'),
    (x_positions[1], f'{n_clean:,} records\\n{n_trips:,} trips\\n({pct_removed:.2f}% removed)'),
    (x_positions[2], f'{n_route_feat} route-level\\n{n_seg_feat} segment-level\\nfeatures'),
    (x_positions[3], 'Point predictions\\n+ residuals\\n(nonconformity scores)'),
    (x_positions[4], 'Prediction intervals\\nwith coverage\\nguarantees'),
    (x_positions[5], 'Per-segment\\nuncertainty\\nattribution'),
]

for x, text in details:
    ax.add_patch(FancyBboxPatch(
        (x, y_detail), box_width, detail_height,
        boxstyle='round,pad=0.08',
        facecolor='#fafafa', edgecolor='#bbbbbb', linewidth=1, alpha=0.95,
    ))
    ax.text(x + box_width/2, y_detail + detail_height/2, text,
            ha='center', va='center', fontsize=7.5, color='#444444', linespacing=1.25)
    ax.annotate('', xy=(x + box_width/2, y_detail + detail_height),
                xytext=(x + box_width/2, y_main),
                arrowprops=dict(arrowstyle='->', color='#aaaaaa',
                                lw=1, linestyle='--'))

y_exp_title = 3.8
ax.text(7.0, y_exp_title, 'Experimental Mapping',
        ha='center', va='center', fontsize=12, fontweight='bold', color='#333333')
ax.plot([1, 13], [3.55, 3.55], color='#dddddd', linewidth=1.5)

y_exp      = 2.2
exp_width  = 3.5
exp_height = 1.2
experiments = [
    {'label': 'Experiment 1 (RQ1)',
     'title': 'Static Conformal Prediction\\nUnder Distribution Shift',
     'color': EXPERIMENT_COLORS['exp1'], 'connects_to': 4},
    {'label': 'Experiment 2 (RQ2)',
     'title': 'Online Adaptive\\nConformal Prediction',
     'color': EXPERIMENT_COLORS['exp2'], 'connects_to': 4},
    {'label': 'Experiment 3 (RQ3)',
     'title': 'Segment-Level Uncertainty\\nDecomposition & Attribution',
     'color': EXPERIMENT_COLORS['exp3'], 'connects_to': 5},
]
exp_x_positions = [1.2, 5.2, 9.2]

for i, exp in enumerate(experiments):
    x = exp_x_positions[i]
    ax.add_patch(FancyBboxPatch(
        (x, y_exp), exp_width, exp_height,
        boxstyle='round,pad=0.12',
        facecolor=exp['color'], edgecolor='white', linewidth=2, alpha=0.85,
    ))
    ax.text(x + exp_width/2, y_exp + exp_height - 0.25, exp['label'],
            ha='center', va='center', fontsize=9, fontweight='bold', color='white')
    ax.text(x + exp_width/2, y_exp + 0.35, exp['title'],
            ha='center', va='center', fontsize=8, color='white', linespacing=1.2)
    stage_x = x_positions[exp['connects_to']] + box_width / 2
    ax.annotate('', xy=(stage_x, y_detail),
                xytext=(x + exp_width/2, y_exp + exp_height),
                arrowprops=dict(arrowstyle='->', color=exp['color'],
                                lw=1.8, linestyle='--',
                                connectionstyle='arc3,rad=0.0'))

y_key = 0.8
ax.text(1.2, y_key, 'Key:', ha='left', va='center',
        fontsize=9, fontweight='bold', color='#555555')
for label, color, kx in [
    ('Pipeline Stage', '#4e79a7', 2.5),
    ('Stage Output',   '#fafafa', 5.5),
    ('Experiment',     '#ff7f0e', 8.5),
]:
    ec = '#bbbbbb' if color == '#fafafa' else 'white'
    ax.add_patch(FancyBboxPatch((kx, y_key - 0.2), 1.0, 0.4,
                                boxstyle='round,pad=0.05',
                                facecolor=color, edgecolor=ec,
                                linewidth=1, alpha=0.9))
    ax.text(kx + 1.2, y_key, label, ha='left', va='center',
            fontsize=8, color='#555555')

out = FIG_DIR_METH / '3_1.png'
fig.savefig(out, dpi=300, bbox_inches='tight', facecolor='white')
fig.savefig(FIG_DIR_METH / '3_1.pdf', bbox_inches='tight', facecolor='white')
plt.show()
print('Saved:', out)
"""


# -------------------------------------------------------------- Figure 3.2
FIG_3_2_MD = """\
---
## Methodology Figure 3.2: Temporal Split Timeline

Saves `../figures/3_2.png` and `.pdf`. Trip counts per period are computed from
`route_df` via `get_temporal_split_by_period`, so they always match the
processed dataset and Table 3.x in the thesis.
"""

FIG_3_2_CODE = """\
# Figure 3.2: Temporal split timeline (data-driven)
import io
from contextlib import redirect_stdout
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from utils.temporal_splits import (
    get_temporal_split_by_period, WEEK_BOUNDARIES, ANOMALOUS_DATES,
)

FIG_DIR_METH = Path('..') / 'figures'
FIG_DIR_METH.mkdir(exist_ok=True)

# Trip counts per temporal period (suppress utility's own prints) ---------
with redirect_stdout(io.StringIO()):
    route_splits = get_temporal_split_by_period(route_df, exclude_anomalous=True)

period_trips = {p: len(route_splits[p]) for p in route_splits}
period_days  = {p: route_splits[p]['date'].nunique() for p in route_splits}

print('Trips per period:', period_trips)
print('Days per period :', period_days)

# Layout ------------------------------------------------------------------
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 11,
    'figure.dpi': 300,
    'savefig.dpi': 300,
})

WEEK_TO_PERIOD = {
    'W1': 'train',       'W2': 'train',       'W3': 'train',
    'W4': 'calibration',
    'W5': 'test_near',
    'W6': 'test_mid',
    'W7': 'test_far',    'W8': 'test_far',
}
PERIOD_GROUPS = {
    'Training':    ['W1', 'W2', 'W3'],
    'Calibration': ['W4'],
    'Test-Near':   ['W5'],
    'Test-Mid':    ['W6'],
    'Test-Far':    ['W7', 'W8'],
}
GROUP_LABELS = {
    'Training':    ('train',       '#4e79a7'),
    'Calibration': ('calibration', '#59a14f'),
    'Test-Near':   ('test_near',   '#f28e2b'),
    'Test-Mid':    ('test_mid',    '#e15759'),
    'Test-Far':    ('test_far',    '#b07aa1'),
}
EXCLUDED_COLOR = '#bbbbbb'

fig, ax = plt.subplots(figsize=(15, 7.6))
ax.set_xlim(0, 16)
ax.set_ylim(-1.8, 6.2)
ax.axis('off')

week_order = ['W1', 'W2', 'W3', 'W4', 'W5', 'W6', 'W7', 'W8']
n_weeks = len(week_order)
x0, x1 = 0.6, 15.4
total_w = x1 - x0
week_w  = total_w / n_weeks
y_bar   = 3.30
bar_h   = 1.1

# Group header bars (trip totals + day counts) ----------------------------
y_header = 5.05
header_h = 0.45
y_label  = 5.6

for group, weeks in PERIOD_GROUPS.items():
    period_key, color = GROUP_LABELS[group]
    gx0 = x0 + week_order.index(weeks[0]) * week_w
    gx1 = x0 + (week_order.index(weeks[-1]) + 1) * week_w
    ax.text((gx0 + gx1) / 2, y_label, group,
            ha='center', va='center', fontsize=11.5, fontweight='bold',
            color=color)
    days = period_days[period_key]
    trips = period_trips[period_key]
    note = '*' if group == 'Test-Mid' else ''
    ax.text((gx0 + gx1) / 2, y_header + header_h / 2,
            f'{trips:,} trips ({days} days{note})',
            ha='center', va='center', fontsize=9, color='#333333')
    ax.plot([gx0 + 0.05, gx1 - 0.05], [y_header - 0.02, y_header - 0.02],
            color=color, linewidth=2.0, alpha=0.85)

# Week boxes --------------------------------------------------------------
for i, w in enumerate(week_order):
    period_key, color = GROUP_LABELS[
        next(g for g, ws in PERIOD_GROUPS.items() if w in ws)
    ]
    gx = x0 + i * week_w
    ax.add_patch(FancyBboxPatch(
        (gx + 0.05, y_bar), week_w - 0.10, bar_h,
        boxstyle='round,pad=0.04',
        facecolor=color, edgecolor='white', linewidth=1.5, alpha=0.92,
    ))
    ax.text(gx + week_w / 2, y_bar + bar_h / 2, w,
            ha='center', va='center', fontsize=12, fontweight='bold',
            color='white')

    start, end = WEEK_BOUNDARIES[w]
    start_ts = pd.Timestamp(start)
    end_ts   = pd.Timestamp(end)
    ax.text(gx + week_w / 2, y_bar - 0.35,
            f'{start_ts.strftime(\"%b %d\")}-{end_ts.strftime(\"%b %d\")}',
            ha='center', va='center', fontsize=8, color='#666666')

# Excluded Sep 3-4 marker inside W6 ---------------------------------------
w6_idx = week_order.index('W6')
w6_x   = x0 + w6_idx * week_w
strip_w = (week_w - 0.10) * 2 / 7  # roughly two days out of seven
ax.add_patch(FancyBboxPatch(
    (w6_x + 0.07, y_bar + 0.04), strip_w, bar_h - 0.08,
    boxstyle='round,pad=0.02',
    facecolor=EXCLUDED_COLOR, edgecolor='white', linewidth=1, alpha=0.95,
))
ax.text(w6_x + 0.07 + strip_w / 2, y_bar + bar_h / 2, 'XX',
        ha='center', va='center', fontsize=11, fontweight='bold', color='#555555')

# (Sep 3-4 footnote rendered below the legend, see end of cell.)

# Temporal-distance arrows below the timeline -----------------------------
def _arrow(ax, x_from, x_to, y, text, color):
    ax.annotate('', xy=(x_to, y), xytext=(x_from, y),
                arrowprops=dict(arrowstyle='<->', color=color,
                                lw=1.5, mutation_scale=12))
    ax.text((x_from + x_to) / 2, y + 0.10, text,
            ha='center', va='bottom', fontsize=8.5, color=color)

w5_x_mid = x0 + (week_order.index('W5') + 0.5) * week_w
w5_x_end = x0 + (week_order.index('W5') + 1)   * week_w
w6_x_mid = x0 + (week_order.index('W6') + 0.5) * week_w
w6_x_end = x0 + (week_order.index('W6') + 1)   * week_w
w7_x_mid = x0 + (week_order.index('W7') + 0.5) * week_w
w8_x_end = x0 + (week_order.index('W8') + 1)   * week_w
w5_x_start = x0 + week_order.index('W5') * week_w

ax.text(0.6, 1.85, 'Temporal distance\\nfrom calibration ->',
        ha='left', va='center', fontsize=8.5, fontstyle='italic', color='#666666')
_arrow(ax, w5_x_start, w5_x_end, 2.10, '1-7 days',  '#f28e2b')
_arrow(ax, w5_x_start, w6_x_end, 1.65, '8-14 days', '#e15759')
_arrow(ax, w5_x_start, w8_x_end, 1.20, '15-27 days','#b07aa1')

# Distribution shift bar --------------------------------------------------
shift_y = 0.55
ax.imshow(
    [[0.0, 0.5, 1.0]], extent=(w5_x_start, w8_x_end, shift_y - 0.10, shift_y + 0.10),
    aspect='auto', cmap='RdPu', alpha=0.7, zorder=1,
)
ax.text((w5_x_start + w8_x_end) / 2, shift_y - 0.32,
        'Increasing Distribution Shift ->',
        ha='center', va='top', fontsize=9, fontweight='bold', color='#7a1f6b')

# Bottom legend -----------------------------------------------------------
legend_y = -1.05
legend_x = 0.6
gap = 2.6
items = [
    ('Training (W1-W3)',    '#4e79a7'),
    ('Calibration (W4)',    '#59a14f'),
    ('Test-Near (W5)',      '#f28e2b'),
    ('Test-Mid (W6)',       '#e15759'),
    ('Test-Far (W7-W8)',    '#b07aa1'),
    ('Excluded',            EXCLUDED_COLOR),
]
for i, (label, color) in enumerate(items):
    cx = legend_x + i * gap
    ax.add_patch(FancyBboxPatch((cx, legend_y), 0.6, 0.25,
                                boxstyle='round,pad=0.02',
                                facecolor=color, edgecolor='white', linewidth=1))
    ax.text(cx + 0.7, legend_y + 0.125, label, ha='left', va='center',
            fontsize=8, color='#444444')

ax.text(0.6, legend_y - 0.30,
        '* Test-Mid (W6) covers 5 days; Sep 3-4 excluded due to data collection failure.',
        ha='left', va='top', fontsize=7.5, fontstyle='italic', color='#777777')

out = FIG_DIR_METH / '3_2.png'
fig.savefig(out, dpi=300, bbox_inches='tight', facecolor='white')
fig.savefig(FIG_DIR_METH / '3_2.pdf', bbox_inches='tight', facecolor='white')
plt.show()
print('Saved:', out)
"""


# -------------------------------------------------------------- Figure 3.5
FIG_3_5_MD = """\
---
## Methodology Figure 3.5: Split Conformal Prediction Process

Saves `../figures/3_5.png` and `.pdf`. The training and calibration trip
counts in the diagram are read from `route_df` via the temporal split, so they
match the rest of the thesis. The lower panels use a small synthetic example
purely for illustration of the procedure, not real residuals.
"""

FIG_3_5_CODE = """\
# Figure 3.5: Split conformal prediction process (data-driven counts)
import io
from contextlib import redirect_stdout
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from utils.temporal_splits import get_temporal_split_by_period

FIG_DIR_METH = Path('..') / 'figures'
FIG_DIR_METH.mkdir(exist_ok=True)

with redirect_stdout(io.StringIO()):
    route_splits = get_temporal_split_by_period(route_df, exclude_anomalous=True)

n_train = len(route_splits['train'])
n_cal   = len(route_splits['calibration'])
print(f'Train trips (W1-W3): {n_train:,}   Calibration scores (W4): {n_cal:,}')

plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 11,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.15,
})

fig = plt.figure(figsize=(15, 8.5))
gs  = fig.add_gridspec(2, 1, height_ratios=[1, 1.2], hspace=0.35)

# -- Top: 4-step pipeline -------------------------------------------------
ax_top = fig.add_subplot(gs[0])
ax_top.set_xlim(0, 15)
ax_top.set_ylim(0, 3.5)
ax_top.axis('off')

steps = [
    {'label':  'Step 1\\nTrain Model',
     'detail': f'XGBoost trained\\non W1-W3\\n({n_train:,} trips)',
     'color':  '#4e79a7', 'x': 0.3},
    {'label':  'Step 2\\nCompute Residuals',
     'detail': f'Predict on W4\\n$R_i = |y_i - \\\\hat{{y}}_i|$\\n({n_cal:,} scores)',
     'color':  '#76b7b2', 'x': 3.3},
    {'label':  'Step 3\\nFind Quantile',
     'detail': 'Sort residuals\\n$\\\\hat{q}$ = 90th percentile\\n(threshold)',
     'color':  '#f28e2b', 'x': 6.3},
    {'label':  'Step 4\\nForm Intervals',
     'detail': 'For each test sample:\\n$[\\\\hat{y} - \\\\hat{q},\\\\; \\\\hat{y} + \\\\hat{q}]$\\n(constant width)',
     'color':  '#e15759', 'x': 9.3},
]

bw, bh = 2.5, 2.8
for i, s in enumerate(steps):
    x = s['x']
    ax_top.add_patch(FancyBboxPatch(
        (x, 0.3), bw, bh, boxstyle='round,pad=0.12',
        facecolor=s['color'], edgecolor='white', linewidth=2, alpha=0.9,
    ))
    ax_top.text(x + bw/2, 0.3 + bh - 0.45, s['label'],
                ha='center', va='center', fontsize=10, fontweight='bold',
                color='white', linespacing=1.3)
    ax_top.text(x + bw/2, 0.3 + bh/2 - 0.35, s['detail'],
                ha='center', va='center', fontsize=9, color='white',
                linespacing=1.3)
    if i < len(steps) - 1:
        ax_top.annotate('',
                        xy=(steps[i+1]['x'] - 0.1, 0.3 + bh/2),
                        xytext=(x + bw + 0.1, 0.3 + bh/2),
                        arrowprops=dict(arrowstyle='->', color='#444444',
                                        lw=2.5, mutation_scale=18))

out_x = 12.3
ax_top.add_patch(FancyBboxPatch(
    (out_x, 0.7), 2.3, 2.0, boxstyle='round,pad=0.12',
    facecolor='#59a14f', edgecolor='white', linewidth=2, alpha=0.9,
))
ax_top.text(out_x + 1.15, 1.7 + 0.45, 'Output',
            ha='center', va='center', fontsize=10,
            fontweight='bold', color='white')
ax_top.text(out_x + 1.15, 1.7 - 0.25,
            'Prediction\\nIntervals\\nwith 90%\\ncoverage\\nguarantee',
            ha='center', va='center', fontsize=8.5, color='white', linespacing=1.2)
ax_top.annotate('',
                xy=(out_x - 0.1, 1.7),
                xytext=(steps[-1]['x'] + bw + 0.1, 1.7),
                arrowprops=dict(arrowstyle='->', color='#444444',
                                lw=2.5, mutation_scale=18))

# -- Bottom: synthetic illustration ---------------------------------------
ax_bot = fig.add_subplot(gs[1])
ax_bot.axis('off')

np.random.seed(42)
n_demo = 80
residuals = np.abs(np.random.normal(0, 400, n_demo)) + np.random.exponential(200, n_demo)
residuals = np.sort(residuals)
q_90 = np.percentile(residuals, 90)

n_test = 25
x_test = np.arange(n_test)
y_pred = np.random.uniform(3500, 6500, n_test)
y_true = y_pred + np.random.normal(0, 500, n_test)
lower  = y_pred - q_90
upper  = y_pred + q_90
covered = (y_true >= lower) & (y_true <= upper)

ax_hist = fig.add_axes([0.06, 0.05, 0.28, 0.42])
ax_hist.hist(residuals, bins=20, color='#76b7b2', edgecolor='white',
             linewidth=0.8, alpha=0.85)
ax_hist.axvline(q_90, color='#e15759', linewidth=2.5, linestyle='--',
                label=f'$\\\\hat{{q}}$ = {q_90:.0f}s (90th pct.)')
ylim = ax_hist.get_ylim()
ax_hist.fill_betweenx([0, max(ylim[1], 1)], q_90, residuals.max() + 50,
                      alpha=0.1, color='#e15759')
ax_hist.set_xlabel('Nonconformity Score $R_i = |y_i - \\\\hat{y}_i|$ (seconds)', fontsize=9)
ax_hist.set_ylabel('Frequency', fontsize=9)
ax_hist.set_title('Calibration Residuals (W4) -- illustrative',
                  fontsize=10, fontweight='bold')
ax_hist.legend(fontsize=8, loc='upper right')
ax_hist.spines['top'].set_visible(False)
ax_hist.spines['right'].set_visible(False)
ax_hist.tick_params(labelsize=8)

ax_pred = fig.add_axes([0.42, 0.05, 0.55, 0.42])
for i in range(n_test):
    color = '#59a14f' if covered[i] else '#e15759'
    ax_pred.plot([x_test[i], x_test[i]], [lower[i], upper[i]],
                 color=color, linewidth=2, alpha=0.5)
ax_pred.scatter(x_test, y_pred, color='#4e79a7', s=40, zorder=5,
                label='Point prediction $\\\\hat{y}$', marker='s')
ax_pred.scatter(x_test[covered], y_true[covered], color='#59a14f', s=45, zorder=6,
                label='True value (covered)', marker='o',
                edgecolors='white', linewidth=0.5)
ax_pred.scatter(x_test[~covered], y_true[~covered], color='#e15759', s=55, zorder=6,
                label='True value (not covered)', marker='X',
                edgecolors='white', linewidth=0.5)
mid_idx = 12
ax_pred.annotate('', xy=(n_test + 0.8, upper[mid_idx]),
                 xytext=(n_test + 0.8, lower[mid_idx]),
                 arrowprops=dict(arrowstyle='<->', color='#f28e2b', lw=2))
ax_pred.text(n_test + 1.3, y_pred[mid_idx],
             f'Width = 2$\\\\hat{{q}}$\\n= {2*q_90:.0f}s\\n({2*q_90/60:.0f} min)',
             ha='left', va='center', fontsize=8, color='#f28e2b', fontweight='bold')
picp = covered.sum() / n_test
ax_pred.set_xlabel('Test Sample Index', fontsize=9)
ax_pred.set_ylabel('Travel Time (seconds)', fontsize=9)
ax_pred.set_title(f'Prediction Intervals on Test Data  (PICP = {picp:.0%})',
                  fontsize=10, fontweight='bold')
ax_pred.legend(fontsize=8, loc='upper left', framealpha=0.9)
ax_pred.spines['top'].set_visible(False)
ax_pred.spines['right'].set_visible(False)
ax_pred.tick_params(labelsize=8)

out = FIG_DIR_METH / '3_5.png'
fig.savefig(out, dpi=300, bbox_inches='tight', facecolor='white')
fig.savefig(FIG_DIR_METH / '3_5.pdf', bbox_inches='tight', facecolor='white')
plt.show()
print('Saved:', out)
"""


# -------------------------------------------------------------- main edits
def remove_eval_methodology_cells() -> None:
    nb = json.loads(EVAL_NB.read_text())
    cells = nb["cells"]
    keep = []
    drop = 0
    for i, c in enumerate(cells):
        src = "".join(c.get("source", []))
        if (
            "S5 -- Figure 3.1" in src
            or "S6 -- Figure 3.5" in src
            or "Figure 3.1: End-to-End Research Framework" in src
            or "Figure 3.5: Split Conformal Prediction Process" in src
        ):
            drop += 1
            continue
        keep.append(c)
    nb["cells"] = keep
    EVAL_NB.write_text(json.dumps(nb, indent=1) + "\n")
    print(f"evalaution.ipynb: removed {drop} cell(s); kept {len(keep)}")


def insert_phase1_methodology_cells() -> None:
    nb = json.loads(PHASE1_NB.read_text())
    cells = nb["cells"]

    # If we re-ran this script, drop any previously inserted cells first
    cells = [
        c for c in cells
        if not any(
            tag in "".join(c.get("source", []))
            for tag in (
                "Methodology Figure 3.1: End-to-End Research Framework",
                "Methodology Figure 3.2: Temporal Split Timeline",
                "Methodology Figure 3.5: Split Conformal Prediction Process",
                "Figure 3.1: End-to-End Research Framework Overview (data-driven)",
                "Figure 3.2: Temporal split timeline (data-driven)",
                "Figure 3.5: Split conformal prediction process (data-driven counts)",
            )
        )
    ]

    # Insert before the "## 9. Save Processed Data" markdown cell
    insert_at = next(
        (
            i for i, c in enumerate(cells)
            if c["cell_type"] == "markdown"
            and "9. Save Processed Data" in "".join(c["source"])
        ),
        None,
    )
    if insert_at is None:
        raise RuntimeError("Could not locate '## 9. Save Processed Data' anchor in Phase1 notebook")

    new_cells = [
        md_cell(FIG_3_1_MD), code_cell(FIG_3_1_CODE),
        md_cell(FIG_3_2_MD), code_cell(FIG_3_2_CODE),
        md_cell(FIG_3_5_MD), code_cell(FIG_3_5_CODE),
    ]
    cells = cells[:insert_at] + new_cells + cells[insert_at:]
    nb["cells"] = cells
    PHASE1_NB.write_text(json.dumps(nb, indent=1) + "\n")
    print(f"Phase1_Preprocessing.ipynb: inserted {len(new_cells)} cells before index {insert_at}")


if __name__ == "__main__":
    remove_eval_methodology_cells()
    insert_phase1_methodology_cells()
