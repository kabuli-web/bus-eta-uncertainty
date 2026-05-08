Here's the synthesis. The rerun has bigger consequences than just numbers changing — the methodology shift to pre-trip segment features invalidates several paragraphs of methodology and re-shapes the Experiment 3 narrative, while also (helpfully) closing the supervisor's biggest remaining risk.

Solid plan to bring V5 → V6
1. Snapshot of what actually moved
Methodology change (high impact, not yet reflected in thesis):

Segment model now uses 19 pre-trip-only features (temporal + spatial + 7-day historical lookback). All cumulative-progress and prev-segment-lag features are gone (still defined in utils/feature_engineering.py but not called from Phase 2).
Preprocessing steps still active: duplicate removal → anomalous-date filter → trip-completeness filter (≥30 segments). IQR outlier removal and "segment 1 / direction 1" filter are commented out.
Result deltas (sample, all need transcribing):

Metric	V5 (inline)	Current rerun
Final segment records	747,798	782,719
Final trip records	19,614	19,685
Calibration trips (W4)	2,740	2,745
Combined test trips	9,276	9,321
Route MAE Cal (s)	812.68	785.50
Route MAE Test-Far (s)	844.14	886.95
Segment MAE Cal (s)	33.73	49.07 ← jumped because pre-trip is weaker
Static CP Test-Near PICP	0.8907	0.8709 ← drift now bigger
MPIW Static CP	3,951	3,962
Re-Cal Per-Route MPIW	1,958	2,323
Direct→Sum residual std	1,133 → 626	1,145 → 796 ← diversification gain shrank
Top attribution segment	Seg 1 Dir 1 (1,599 s)	Seg 1 Dir 0 (1,827 s) ← direction flipped
Best CP config (Winkler)	Slide-14d × Norm. DE (4,690)	Online Exp × Mondrian+DE (5,551)
The narrative shifts: pre-trip-only is honest but the segment model is weaker, so Experiment 3's advantage over direct route CP is smaller. That's a feature of the new framing, not a bug — but the discussion needs to acknowledge it.

2. Action plan, ordered by risk
Tier A — Supervisor-flagged risks (must fix; these are what put V5 borderline)
#	Risk	Where	Action
A1	"Three steps but lists four" + stale sample count	thesisV5.tex:378-388	Rewrite Section 3.2.2 to: (a) say four steps explicitly, (b) describe what was kept and what was removed and why (IQR removal + seg1/dir1 filter), (c) update final counts to 782,719 segments / 19,685 trips.
A2	Exp 3 "in-trip" framing (unfair vs pre-departure baseline)	L403, L405, L409, L657	Delete the "cumulative trip progress" and "preceding segment lags" paragraphs entirely. Rewrite Exp 3 to state plainly: both the route model and the segment model use pre-trip-only features; this is now a fair pre-departure comparison. Replace the "in-trip prediction setting" sentence at L657 with "pre-departure prediction with finer spatial granularity."
A3	Autorank score ranges don't match Table 4.9	autorank ¶ at L1022; Winkler matrix at L985-995	Re-derive the four NCM-family Winkler ranges from the new 4×4 table (Absolute: 6,411–6,449; Mondrian: 6,164–6,200; Norm. DE: 5,601–5,707; Mondrian+DE: 5,551–5,827) and update the paragraph. Also update the "best Winkler 4,690" claim → 5,551, Online Exp × Mondrian+DE.
A4	\cite{lofstrom2024calibrated} used for Mondrian CP	L223	Replace with the actual Mondrian CP source (Vovk et al., 2003 — Mondrian Confidence Machines) for the theory citation. Keep lofstrom2024calibrated only at L467 where the library is cited.
A5	Mixed UK/US spelling	global	Standardise on British (already dominant: 15 vs 9). Targeted sed-pass: analyze→analyse, behavior→behaviour, normalize→normalise, normalization→normalisation, modeling→modelling. Verify after with grep.
Tier B — Numbers and tables (mechanical but extensive)
Everything in this tier is value substitution. The plan is to add to_latex export cells to the evaluation notebook so the round-trip is reproducible (the way Exp 3 attribution already is), then transcribe.

B1. Hand-update inline tables that already have a regenerated .tex source on disk:

tab:temporal_split (L282) ← outputs/tables/T1_3_temporal_split_statistics.tex
tab:hyperparams_repro (L731) ← Phase3 best config (now n_est=500, max_depth=6, lr=0.01, λ=5)
tab:route_performance (L785) ← T3_2_route_xgboost_performance.tex
tab:segment_performance (L810) ← T3_3_segment_xgboost_performance.tex
tab:exp3_attribution (L1073) ← T_exp3_attribution.tex (after re-running the cell I rewrote earlier)
B2. Add export cells to notebooks/evalaution.ipynb for the 8 tables that currently only exist as printed output, then transcribe:

Exp 1: tab:exp1_overall, tab:exp1_period, tab:exp1_multiconf
Exp 2: tab:exp2_online_vs_static, tab:exp2_period, tab:exp2_picp_matrix, tab:exp2_winkler_matrix
Exp 3: tab:exp3_route_recal
B3. Sample-count find-and-replace for every prose mention: 2,740 → 2,745, 9,276 → 9,321, 747,798 → 782,719, 19,614 → 19,685 (locations: L380, L388, L566, L622, every table caption from L841 to L1075).

Tier C — Narrative updates from new numbers
C1. RQ1 / Static-CP story. Test-Near coverage now drops to 0.8709 (calibration error 0.029, was 0.014). The "1.8 pp drop" sentence at L893 is now ~3 pp at 90% — the drift effect is stronger, which actually strengthens the claim that marginal coverage hides the issue. Rewrite that paragraph to lean into it.

C2. RQ2 / NCM ranking flipped. Best Winkler is now Online Expanding × Mondrian+DE (5,551) rather than Slide-14d × Norm. DE (4,690). Update the "best overall" sentence at L1014 and the Discussion paragraph at L1107-L1118.

C3. RQ3 / residual diversification weakened. New std reduction is 1,145 → 796 (30%, was 44%); MPIW improvement is 3,962 → 2,323 = 41% (was 54%). Rewrite L1056-L1062 with the new percentages and add one sentence acknowledging the trade-off: "The narrower diversification gain is the price paid for using only pre-trip features at the segment level — a fair comparison with the route baseline." This is the bridge that turns the methodology change from a bug-fix into a contribution.

C4. Top-5 attribution interpretation (user's request). Append a paragraph after L1088 explaining why Segment 1 dominates. Suggested framing: "Segment 1 carries 23% of the per-trip uncertainty budget despite being only one of ~30 segments per trip. This is a data-collection artefact rather than a real traffic phenomenon: segment-1 records cover the dwell time at the origin terminal plus the time the vehicle spends signing on, where actual departure can lag the scheduled departure by several minutes. The historical-statistic features (hist_seg_mean, hist_seg_std) reflect this larger raw variance, and the difficulty estimator inherits it." — confirm the artefact story matches reality before locking it in.

Tier D — Bibliography and minor
D1. Add Vovk Mondrian CP entry to bibliography; fix L223 citation (see A4).
D2. Re-render the residual-diversification figure (figures/exp3_residual_diversification.pdf) and the autorank CD diagram if the underlying numbers fed them — confirm with the user which figures were re-saved.
D3. Search for "in-trip" and "cumulative" globally one more time after Tier A2 to make sure no stragglers remain.

3. Suggested execution order
A1, A2, A4, A5 in one editing pass (these are textual; no numbers needed).
B1 (5 inline tables — already have source files).
Run all evaluation-notebook cells once (regenerates T_exp3_attribution.tex with the seconds-scale fix from earlier this session).
B2 (add the 8 to_latex export cells, re-run, transcribe).
B3 sample-count sweep.
A3 + C1, C2, C3, C4 (narrative — can only be done after numbers are in).
D1, D2, D3 final pass.
4. Things I want to verify with you before editing
The segment-1 artefact story in C4 — is the "origin-terminal dwell + sign-on lag" mechanism actually the right interpretation, or is it something else (e.g. how the segments are indexed in the source feed)? I don't want to ship a confident hypothesis that turns out wrong.
What to do with the segment-level hyperparameters — they're not in any output file. Were they tuned the same way as the route model, or hard-coded? tab:hyperparams_repro currently shows segment values that may also be stale.
Figures — were the figures (figures/*.pdf) regenerated as part of the rerun, or do some still come from the previous run? If yes, list which ones so I include them in Tier D2.
Want me to start with Tier A (the supervisor-flagged textual fixes), or with Tier B1 (the five inline tables we already have data for)?