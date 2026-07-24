"""qr bench — multi-model classification benchmark on a feature CSV.

Usage::

    qr bench -i features.csv --outcome event --models LR,RF,XGB,LGBM,FLAML --cv 10
    qr bench -i features.csv --outcome event --all-models --hpo optuna
    qr bench -i features.csv --outcome event --external-test ext.csv
    qr bench -i features.csv --outcome event --models TPOT --external-test ext.csv \\
        --tpot-repeats 5 --calibrate-threshold

Runs nested cross-validation over the selected models and reports OOF ROC-AUC,
average precision, and Brier score. If --external-test is provided, each
CV-selected model is refitted on the full training set and evaluated on the
test set.

Models available: LR, SVM, ExtraTrees, RF, GBM, MLP, XGB, LGBM, FLAML, TPOT
(TPOT is opt-in only — request it explicitly via --models, --all-models never
includes it, because it is materially more expensive and, per below, not
reproducible from a single run).

TPOT caveat: tpot's genetic-programming search is NOT reproducible run-to-run
even with a fixed seed (dask gather order is wall-clock dependent — see
qradiomics.classification.registry._tpot for the full explanation). Its
--cv fold-mean/std therefore mixes real fold variance with genetic-search
noise, and any single external-test refit is a single noisy draw. Use
--tpot-repeats (default 5) to refit TPOT N independent times on
--external-test and report ext_*_mean/ext_*_std instead of a single number.

--calibrate-threshold (default off, backward compatible): computes
out-of-fold predicted probabilities on the training data, picks the
Youden's-J threshold instead of the hardcoded 0.5, and reports the chosen
threshold plus accuracy/sensitivity/specificity on --external-test at that
threshold. Cost compounds with --tpot-repeats: each TPOT repeat also runs a
full inner OOF cross-validation (up to 10 more TPOT fits) purely to select
its own threshold, so --models TPOT --calibrate-threshold with the default
--tpot-repeats 5 can mean ~55 TPOT genetic searches. Opt into both together
only when you can afford it.

See also: ``qr ml benchmark`` — the canonical, recommended entry point that
unifies this command with ``qr ml classify`` (adds the correlation
pre-filter step) on the same underlying engine (``_run_benchmark`` below).
``qr bench`` is kept as a permanent backward-compatible alias; it is not
deprecated.
"""
from __future__ import annotations

import warnings
from pathlib import Path

import click
import numpy as np
import pandas as pd


def _run_benchmark(
    df, outcome, *,
    label,
    output_dir,
    csv_prefix,
    models,
    all_models,
    cv,
    inner_cv,
    seed,
    n_jobs,
    hpo,
    optuna_trials,
    ext_path,
    tpot_repeats,
    calibrate_threshold,
    exclude=(),
    extra_exclude=(),
    fillna_median=False,
    restrict_train_to_shared=False,
    corr_threshold=None,
    features_echo_label="features",
    best_label="Best model",
    print_artefacts_line=True,
    title_prefix="qr bench",
):
    """Shared benchmark engine: CV benchmark + optional correlation
    pre-filter + optional external-test refit/scoring + HTML report.

    Used by ``qr bench``, ``qr ml classify``, and ``qr ml benchmark`` (the
    canonical entry point) so all three stay on one implementation. Most
    keyword arguments mirror CLI flags shared by at least two of the three
    commands; a few (``extra_exclude``, ``fillna_median``,
    ``restrict_train_to_shared``, ``features_echo_label``, ``best_label``,
    ``print_artefacts_line``, ``title_prefix``) are not CLI flags — they
    exist purely so each thin command can reproduce its own pre-existing
    console/report output byte-for-byte instead of silently adopting a
    sibling command's behavior.

    Returns (summary: pd.DataFrame, ext_rows: list[dict], result: BenchmarkResult).
    """
    from qradiomics.classification import cross_val_benchmark
    from qradiomics.classification.registry import DEFAULT_MODELS, MODEL_REGISTRY

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    if outcome not in df.columns:
        raise click.UsageError(f"Outcome column '{outcome}' not found. Available: {list(df.columns)}")

    exclude_cols = set(exclude) | set(extra_exclude) | {outcome}
    feature_cols = [c for c in df.columns if c not in exclude_cols and pd.api.types.is_numeric_dtype(df[c])]
    X = df[feature_cols]
    if fillna_median:
        X = X.fillna(X.median())
    y = df[outcome].astype(int)

    click.echo(
        f"[{label}] n={len(df)}  {features_echo_label}={len(feature_cols)}  "
        f"events={y.sum()} ({y.mean()*100:.1f}%)"
    )

    if corr_threshold is not None and corr_threshold < 1.0:
        corr = X.corr().abs()
        upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
        drop_cols = [c for c in upper.columns if any(upper[c] > corr_threshold)]
        X = X.drop(columns=drop_cols)
        click.echo(f"[{label}] after corr filter (r>{corr_threshold}): {len(X.columns)} features ({len(drop_cols)} dropped)")

    # Resolve model list
    if all_models:
        model_list = list(DEFAULT_MODELS)
    else:
        model_list = [m.strip() for m in models.split(",") if m.strip()]

    unknown = [m for m in model_list if m not in MODEL_REGISTRY]
    if unknown:
        raise click.UsageError(f"Unknown models: {unknown}. Available: {list(MODEL_REGISTRY)}")

    click.echo(f"[{label}] models: {', '.join(model_list)}")
    click.echo(f"[{label}] {cv}×{inner_cv} nested CV  hpo={hpo}  seed={seed}")

    result = cross_val_benchmark(
        X, y,
        models=model_list,
        cv=cv, inner_cv=inner_cv,
        random_state=seed,
        n_jobs=n_jobs,
        hpo=hpo,
        optuna_trials=optuna_trials,
        verbose=True,
    )

    summary = result.summary()
    summary.to_csv(out / f"{csv_prefix}_cv_results.csv", index=False)

    click.echo("\n=== CV Results (sorted by OOF AUC) ===")
    click.echo(summary.to_string(index=False))

    best = result.best()
    click.echo(f"\n★ {best_label}: {best.model}  OOF AUC={best.oof_auc:.3f}  AP={best.oof_ap:.3f}  Brier={best.oof_brier:.3f}")

    # Optional external evaluation
    ext_results = []
    if ext_path:
        ext_df = pd.read_csv(ext_path)
        y_ext = ext_df[outcome].astype(int).to_numpy()
        y_tr = y.to_numpy(dtype=int)

        if restrict_train_to_shared:
            shared_feats = [c for c in X.columns if c in ext_df.columns]
            X_ext = ext_df[shared_feats]
            if fillna_median:
                X_ext = X_ext.fillna(X_ext.median())
            X_ext_arr = X_ext.to_numpy(dtype=float)
            X_tr = X[shared_feats].to_numpy(dtype=float)
        else:
            X_ext = ext_df[[c for c in feature_cols if c in ext_df.columns]]
            X_ext_arr = X_ext.to_numpy(dtype=float)
            X_tr = X.to_numpy(dtype=float)

        click.echo(f"\n[{label}] external test n={len(ext_df)}  events={y_ext.sum()}")

        cv_oof_lookup = {mr.model: mr.oof_auc for mr in result.model_results}
        ext_results = build_external_test_rows(
            model_list, cv_oof_lookup, X_tr, y_tr, X_ext_arr, y_ext,
            seed=seed, inner_cv=inner_cv, n_jobs=n_jobs,
            tpot_repeats=tpot_repeats, calibrate_threshold=calibrate_threshold,
            verbose=True,
        )

        ext_df_out = pd.DataFrame(ext_results).sort_values("ext_auc", ascending=False)
        ext_df_out.to_csv(out / f"{csv_prefix}_external_results.csv", index=False)
        click.echo("\n=== External Test Results ===")
        display_cols = ["model", "cv_oof_auc", "ext_auc", "ext_auc_std", "ext_repeats",
                        "ext_ap", "ext_brier"]
        if calibrate_threshold:
            # Some/all rows may have skipped calibration (see the
            # minority-class guard in _fit_and_score_external), in which
            # case these columns are simply absent from ext_df_out —
            # only display the ones that actually exist.
            display_cols += [c for c in ("threshold", "accuracy", "sensitivity", "specificity")
                             if c in ext_df_out.columns]
        click.echo(ext_df_out[display_cols].to_string(index=False))

    # Write HTML report
    _write_html_report(summary, ext_results, out / f"{csv_prefix}_report.html",
                       n_train=len(y), n_features=len(X.columns),
                       outcome=outcome, models=model_list, cv=cv, seed=seed,
                       tpot_repeats=tpot_repeats, calibrate_threshold=calibrate_threshold,
                       title_prefix=title_prefix)
    if print_artefacts_line:
        click.echo(f"\n[{label}] Artefacts → {out}")

    return summary, ext_results, result


def _shared_benchmark_options(f):
    """``@click.option`` decorators identical across ``qr bench``,
    ``qr ml classify``, and ``qr ml benchmark`` — all three are thin CLI
    wrappers around the shared ``_run_benchmark`` engine above.

    Applied to all three commands so there is exactly one source of truth
    for each shared flag's default/help text (previously each command
    copy-pasted its own ``@click.option`` block, which had already drifted:
    ``--n-jobs``'s help text differed between ``qr bench``/``qr ml
    benchmark`` and ``qr ml classify``).

    Command-specific options that only one or two of the three expose stay
    as individual ``@click.option`` decorators stacked on top of this one:
    ``--corr-threshold`` (``classify``/``benchmark`` only), ``--optuna-
    trials`` (``bench``/``benchmark`` only — ``classify`` never exposed it
    and must keep relying on ``cross_val_benchmark``'s own default),
    and ``--output-dir`` (present on all three, but with a different
    per-command default).
    """
    options = [
        click.option("--input", "-i", "input_path", required=True, type=click.Path(exists=True),
                     help="Feature CSV (rows=samples, columns=features + outcome)."),
        click.option("--outcome", "-o", required=True, help="Binary outcome column name."),
        click.option("--models", "-m", default="LR,SVM,ExtraTrees,RF,GBM,MLP,XGB,LGBM",
                     show_default=True,
                     help="Comma-separated model names. Use --all-models to include FLAML. "
                          "TPOT is opt-in only and must be named explicitly (--models TPOT)."),
        click.option("--all-models", "all_models", is_flag=True,
                     help="Run all registered models including FLAML AutoML (never includes "
                          "TPOT — request it explicitly via --models TPOT)."),
        click.option("--cv", default=10, show_default=True, help="Outer CV folds."),
        click.option("--inner-cv", default=5, show_default=True, help="Inner HPO folds."),
        click.option("--seed", default=42, show_default=True, help="Random seed."),
        click.option("--n-jobs", default=-1, show_default=True,
                     help="Parallelism for GridSearchCV. Also bounds the outer "
                          "--tpot-repeats refit loop when parallelized (TPOT's own "
                          "internal parallelism is pinned to 1 per repeat to avoid "
                          "oversubscription — see module docstring)."),
        click.option("--hpo", type=click.Choice(["grid", "optuna"]), default="grid",
                     show_default=True, help="Hyperparameter optimisation method."),
        click.option("--external-test", "ext_path", default=None, type=click.Path(exists=True),
                     help="Optional external test CSV (same columns as --input). Refit best "
                          "model and evaluate."),
        click.option("--tpot-repeats", default=5, show_default=True, type=click.IntRange(min=1),
                     help="TPOT only: independent refits (seed+i) on --external-test, "
                          "reported as ext_*_mean/ext_*_std. TPOT's genetic search is "
                          "not reproducible from a single run — see module docstring. "
                          "Must be >= 1."),
        click.option("--calibrate-threshold", "calibrate_threshold", is_flag=True,
                     help="Pick a Youden's-J decision threshold from out-of-fold "
                          "training probabilities (instead of hardcoded 0.5) for "
                          "accuracy/sensitivity/specificity on --external-test."),
        click.option("--exclude", multiple=True,
                     help="Column names to exclude from features (in addition to --outcome)."),
    ]
    for opt in reversed(options):
        f = opt(f)
    return f


@click.command("bench")
@_shared_benchmark_options
@click.option("--optuna-trials", default=30, show_default=True,
              help="Optuna trial count (only when --hpo optuna).")
@click.option("--output-dir", "-d", default="results/bench", show_default=True,
              type=click.Path(), help="Output directory for CSV + HTML report.")
def bench(
    input_path, outcome, models, all_models, cv, inner_cv,
    seed, n_jobs, hpo, optuna_trials, ext_path, tpot_repeats,
    calibrate_threshold, output_dir, exclude,
):
    """Multi-model classification benchmark: CV + optional external evaluation.

    See also: ``qr ml benchmark`` — the recommended canonical form of this
    same benchmark engine (adds the correlation pre-filter from
    ``qr ml classify``). This command remains a permanent backward-compatible
    alias, not deprecated.
    """
    df = pd.read_csv(input_path)
    _run_benchmark(
        df, outcome,
        label="bench",
        output_dir=output_dir,
        csv_prefix="bench",
        models=models,
        all_models=all_models,
        cv=cv, inner_cv=inner_cv,
        seed=seed, n_jobs=n_jobs,
        hpo=hpo, optuna_trials=optuna_trials,
        ext_path=ext_path,
        tpot_repeats=tpot_repeats,
        calibrate_threshold=calibrate_threshold,
        exclude=exclude,
    )


def _fit_and_score_external(pipe, grid, X_tr, y_tr, X_ext, y_ext, *,
                            inner_cv, n_jobs, seed, calibrate_threshold):
    """Fit one pipeline (GridSearchCV if it has a param grid, else a plain
    fit — AutoML backends like FLAML/TPOT manage their own tuning and are
    registered with an empty grid) on the full training set and score it
    against an external/held-out test set.

    If ``calibrate_threshold`` is set, also computes out-of-fold predicted
    probabilities on the training data (a fresh clone per fold — see
    ``qradiomics.analytics.classification_calibration.oof_proba``) to pick a
    Youden's-J decision threshold, then reports accuracy/sensitivity/
    specificity on the external set at that threshold instead of the
    default 0.5.

    Returns (metrics: dict, best_params: dict).
    """
    from sklearn.metrics import (
        accuracy_score,
        average_precision_score,
        brier_score_loss,
        confusion_matrix,
        roc_auc_score,
    )
    from sklearn.model_selection import GridSearchCV, StratifiedKFold

    if grid:
        # Unlike the no-grid/AutoML branch below, do NOT suppress warnings
        # here: these are real sklearn models with actual hyperparameter
        # grids, and convergence warnings etc. are meaningful signal about
        # whether the refit actually converged. Let them surface normally.
        inner = StratifiedKFold(n_splits=inner_cv, shuffle=True, random_state=seed + 1)
        gs = GridSearchCV(pipe, grid, cv=inner, scoring="roc_auc", n_jobs=n_jobs, refit=True)
        gs.fit(X_tr, y_tr)
        estimator, best_params = gs.best_estimator_, gs.best_params_
    else:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            pipe.fit(X_tr, y_tr)
        estimator, best_params = pipe, {}

    y_score = estimator.predict_proba(X_ext)[:, 1]
    metrics = {
        "auc": float(roc_auc_score(y_ext, y_score)),
        "ap": float(average_precision_score(y_ext, y_score)),
        "brier": float(brier_score_loss(y_ext, y_score)),
    }

    if calibrate_threshold:
        minority_count = int(np.bincount(y_tr).min())
        if minority_count < 2:
            # A single-member minority class cannot be stratified into >=2
            # folds without guaranteeing an empty-class training partition
            # in at least one fold (StratifiedKFold would raise). Forcing a
            # minimum of 2 splits here used to paper over this and crash
            # downstream in oof_proba/youden_threshold instead. Skip
            # threshold calibration entirely and fall back to the default
            # 0.5 boundary (i.e. leave threshold/accuracy/sensitivity/
            # specificity unset) rather than crash.
            warnings.warn(
                "Skipping --calibrate-threshold: training minority class has "
                f"only {minority_count} member(s), insufficient for "
                "out-of-fold stratified threshold calibration. Falling back "
                "to the default 0.5 decision boundary.",
                stacklevel=2,
            )
        else:
            from qradiomics.analytics.classification_calibration import (
                oof_proba,
                youden_threshold,
            )

            n_splits = min(inner_cv, 10, minority_count)
            # Clone `estimator` (not the raw `pipe` template) so the OOF
            # search used to pick the threshold matches the hyperparameters
            # actually used for the final external-test scoring model —
            # sklearn's clone() resets fitted state but preserves
            # get_params(), so this is safe even though `estimator` (e.g.
            # gs.best_estimator_) is already fitted on the full training
            # set.
            oof = oof_proba(estimator, X_tr, y_tr, n_splits=n_splits, random_state=seed)
            thr = youden_threshold(y_tr, oof)
            y_pred = (y_score >= thr).astype(int)
            tn, fp, fn, tp = confusion_matrix(y_ext, y_pred, labels=[0, 1]).ravel()
            metrics["threshold"] = float(thr)
            metrics["accuracy"] = float(accuracy_score(y_ext, y_pred))
            metrics["sensitivity"] = float(tp / (tp + fn)) if (tp + fn) else float("nan")
            metrics["specificity"] = float(tn / (tn + fp)) if (tn + fp) else float("nan")

    return metrics, best_params


def _repeat_fit_and_score(name, i, seed, X_tr, y_tr, X_ext, y_ext, *,
                          inner_cv, n_jobs, calibrate_threshold, pin_inner_n_jobs):
    """Single independent refit+score for one ``--tpot-repeats`` repeat.

    Kept module-level (not a closure) so it is picklable for joblib's
    process-based backend when ``build_external_test_rows`` parallelizes
    the repeat loop across repeats (see there for why).
    """
    from qradiomics.classification.registry import build_model

    pipe, grid = build_model(name, random_state=seed + i)
    if pin_inner_n_jobs is not None:
        # Avoid oversubscription: when the outer repeat loop itself is
        # parallelized, pin this model's own internal parallelism (e.g.
        # TPOT's genetic-search n_jobs) to a single worker and let the
        # outer joblib.Parallel own the parallelism instead.
        try:
            pipe.set_params(clf__n_jobs=pin_inner_n_jobs)
        except ValueError:
            pass
    return _fit_and_score_external(
        pipe, grid, X_tr, y_tr, X_ext, y_ext,
        inner_cv=inner_cv, n_jobs=n_jobs, seed=seed + i,
        calibrate_threshold=calibrate_threshold,
    )


def build_external_test_rows(model_names, cv_oof_lookup, X_tr, y_tr, X_ext, y_ext, *,
                             seed, inner_cv, n_jobs, tpot_repeats=5,
                             calibrate_threshold=False, verbose=False):
    """Refit each named model on the full training set and evaluate on an
    external/held-out test set. Shared by ``qr bench``, ``qr ml classify``,
    and ``qr ml benchmark`` (formerly duplicated inline in the first two).

    TPOT special-case: TPOT's genetic search is not reproducible run-to-run
    (see ``qradiomics.classification.registry._tpot``), so it is refit
    ``tpot_repeats`` independent times (random_state = seed + i) and the
    external metrics are reported as mean +/- std (``ext_auc``/
    ``ext_auc_std``/... with ``ext_repeats`` = tpot_repeats) rather than a
    single noisy number. This is intentionally a *different* statistic from
    the k-fold ``cv_auc_mean``/``cv_auc_std`` columns in the main CV summary
    (those mix real fold variance with TPOT's search noise; these mix
    genetic-search noise across independent full-data refits) — the two
    must never be conflated under the same column name.

    Each repeat is fully independent (different random_state, independent
    fit+score), so when there is more than one (TPOT only) they are
    dispatched via ``joblib.Parallel`` instead of a sequential Python loop.
    To avoid oversubscribing cores, each repeat's own internal parallelism
    (e.g. TPOT's genetic-search ``n_jobs``) is pinned to 1 while the outer
    ``joblib.Parallel(n_jobs=n_jobs)`` owns the parallelism. Single-fit
    models (``n_repeats == 1``) skip the parallel-dispatch machinery
    entirely — there is nothing to parallelize.

    All other models are fit once (``ext_repeats`` = 1, ``*_std`` = 0.0), so
    the returned rows share one uniform schema regardless of model.

    Returns a list of dict rows (one per model).
    """
    rows = []
    for name in model_names:
        n_repeats = tpot_repeats if name == "TPOT" else 1

        if n_repeats > 1:
            from joblib import Parallel, delayed

            results = Parallel(n_jobs=n_jobs)(
                delayed(_repeat_fit_and_score)(
                    name, i, seed, X_tr, y_tr, X_ext, y_ext,
                    inner_cv=inner_cv, n_jobs=1,
                    calibrate_threshold=calibrate_threshold,
                    pin_inner_n_jobs=1,
                )
                for i in range(n_repeats)
            )
        else:
            results = [
                _repeat_fit_and_score(
                    name, 0, seed, X_tr, y_tr, X_ext, y_ext,
                    inner_cv=inner_cv, n_jobs=n_jobs,
                    calibrate_threshold=calibrate_threshold,
                    pin_inner_n_jobs=None,
                )
            ]

        run_metrics = [m for m, _ in results]
        best_params: dict = results[0][1]

        if verbose and n_repeats > 1:
            for i, (m, _bp) in enumerate(results):
                click.echo(f"    [{name}] repeat {i + 1}/{n_repeats}: "
                           f"ext_auc={m['auc']:.3f}")

        row = {
            "model": name,
            "cv_oof_auc": round(cv_oof_lookup.get(name, float("nan")), 4),
            "ext_repeats": n_repeats,
        }
        for key in ("auc", "ap", "brier"):
            vals = [m[key] for m in run_metrics]
            row[f"ext_{key}"] = round(float(np.mean(vals)), 4)
            row[f"ext_{key}_std"] = round(float(np.std(vals)), 4) if n_repeats > 1 else 0.0
        if calibrate_threshold:
            # A given repeat may have skipped calibration (see
            # _fit_and_score_external's minority-class guard), in which
            # case its metrics dict simply lacks these keys. Aggregate over
            # whichever repeats actually calibrated; if none did, leave the
            # columns out of this row entirely rather than crash.
            for key in ("threshold", "accuracy", "sensitivity", "specificity"):
                vals = [m[key] for m in run_metrics if key in m]
                if not vals:
                    continue
                row[key] = round(float(np.mean(vals)), 4)
                row[f"{key}_std"] = round(float(np.std(vals)), 4) if len(vals) > 1 else 0.0
        row["best_params"] = str(best_params)
        rows.append(row)
    return rows


def _write_html_report(summary, ext_rows, outpath, *, title_prefix="qr bench", **meta):
    cv_rows = ""
    has_tpot_cv = False
    for i, (_, r) in enumerate(summary.iterrows()):
        if r["model"] == "TPOT":
            has_tpot_cv = True
        cls = ' style="font-weight:bold;background:#e8fae8"' if i == 0 else ""
        cv_rows += (
            f"<tr{cls}><td>{r['model']}</td><td>{r['oof_auc']}</td>"
            f"<td>{r['oof_ap']}</td><td>{r['oof_brier']}</td>"
            f"<td>{r['cv_auc_mean']}±{r['cv_auc_std']}</td></tr>\n"
        )

    has_repeats = any(r.get("ext_repeats", 1) > 1 for r in ext_rows)
    has_threshold = any("threshold" in r for r in ext_rows)

    ext_html = ""
    if ext_rows:
        thresh_header = (
            "<th>Threshold</th><th>Accuracy</th><th>Sensitivity</th><th>Specificity</th>"
            if has_threshold else ""
        )
        for i, r in enumerate(sorted(ext_rows, key=lambda x: -x["ext_auc"])):
            cls = ' style="font-weight:bold;background:#e8fae8"' if i == 0 else ""
            auc_cell = f"{r['ext_auc']}"
            if r.get("ext_repeats", 1) > 1:
                auc_cell += f" ± {r.get('ext_auc_std', 0)} (N={r['ext_repeats']})"
            thresh_cells = ""
            if has_threshold:
                if "threshold" in r:
                    thresh_cells = (
                        f"<td>{r['threshold']}</td><td>{r['accuracy']}</td>"
                        f"<td>{r['sensitivity']}</td><td>{r['specificity']}</td>"
                    )
                else:
                    thresh_cells = "<td>—</td><td>—</td><td>—</td><td>—</td>"
            ext_html += (
                f"<tr{cls}><td>{r['model']}</td><td>{r['cv_oof_auc']}</td>"
                f"<td>{auc_cell}</td><td>{r['ext_ap']}</td><td>{r['ext_brier']}</td>"
                f"{thresh_cells}</tr>\n"
            )
        ext_section = f"""<h2>External Test Results</h2>
<table><tr><th>Model</th><th>CV OOF AUC</th><th>Ext AUC</th><th>Ext AP</th><th>Ext Brier</th>
{thresh_header}</tr>
{ext_html}</table>"""
    else:
        ext_section = ""

    footnotes = []
    if has_tpot_cv or has_repeats:
        footnotes.append(
            "<strong>TPOT variance note:</strong> the CV fold-mean±std column "
            "above (all models) reflects k-fold data-split variance. For TPOT, "
            "that number ALSO includes genetic-search run-to-run noise "
            "(TPOT's search is not reproducible from random_state alone — "
            "see qradiomics.classification.registry._tpot). In the External "
            "Test table, TPOT's \"Ext AUC ± std (N=...)\" reflects "
            "--tpot-repeats independent full refits (mean/std across "
            "repeats) — a different statistic from the CV column, reported "
            "separately by design; never conflate the two."
        )
    if has_threshold:
        footnotes.append(
            "<strong>Threshold calibration:</strong> Accuracy/Sensitivity/"
            "Specificity use a Youden's-J threshold picked from out-of-fold "
            "training probabilities (qradiomics.analytics."
            "classification_calibration.youden_threshold), not the default "
            "0.5 boundary."
        )
    footnote_html = "".join(f"<p>{f}</p>" for f in footnotes)

    best = summary.iloc[0]
    html = f"""<!DOCTYPE html><html><head><meta charset="UTF-8">
<title>{title_prefix} — Multi-model Classification Report</title>
<style>
body{{font-family:Arial,sans-serif;max-width:900px;margin:0 auto;padding:20px;background:#fafafa}}
h1{{color:#1a3a5c;border-bottom:3px solid #1a3a5c}}h2{{color:#1a3a5c;margin-top:24px}}
table{{border-collapse:collapse;width:100%;margin:10px 0;background:#fff;box-shadow:0 1px 3px rgba(0,0,0,.08)}}
th{{background:#1a3a5c;color:#fff;padding:8px 10px;text-align:left}}
td{{padding:6px 10px;border-bottom:1px solid #dde}}tr:hover td{{background:#f0f5ff}}
</style></head><body>
<h1>{title_prefix} — Multi-Model Classification</h1>
<div style="background:#f4f7fb;border-left:4px solid #1a3a5c;padding:10px;margin:12px 0">
n_train={meta.get('n_train')} | features={meta.get('n_features')} | outcome={meta.get('outcome')} |
CV={meta.get('cv')}-fold | seed={meta.get('seed')} | models={', '.join(meta.get('models',[]))}
</div>
<div style="background:#e8fae8;border-left:4px solid green;padding:10px;margin:12px 0">
<strong>★ Best: {best['model']}  OOF AUC={best['oof_auc']}  AP={best['oof_ap']}  Brier={best['oof_brier']}</strong>
</div>

<h2>Cross-Validation Results (OOF)</h2>
<table><tr><th>Model</th><th>OOF AUC</th><th>OOF AP</th><th>OOF Brier</th><th>CV fold-mean±std</th></tr>
{cv_rows}</table>
{ext_section}
<div style="color:#555;font-size:.85em;margin-top:16px;background:#fff8e6;border-left:4px solid #d9a441;padding:10px">
{footnote_html}
</div>
<div style="color:#888;font-size:.8em;margin-top:24px;border-top:1px solid #dde;padding-top:8px">
Generated by {title_prefix} | qradiomics classification module
</div></body></html>"""
    outpath.write_text(html)
