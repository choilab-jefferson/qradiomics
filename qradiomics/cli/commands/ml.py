"""qr ml — train / evaluate / predict outcome-prediction models.

Atomic ML primitives that close the radiomics 4-stage data flow:

    data → image → features → **modeling**

Subcommands:
  * benchmark — canonical multi-model nested CV benchmark (LR, SVM, ET, RF,
                GBM, MLP, XGB, LGBM by default; FLAML via --all-models; TPOT
                opt-in only via --models TPOT — see qr ml benchmark --help
                for the TPOT reproducibility caveat and --tpot-repeats /
                --calibrate-threshold flags). Recommended entry point;
                unifies `qr bench` and `qr ml classify` (both kept as
                permanent backward-compatible aliases, not deprecated).
  * classify  — backward-compatible alias of `qr ml benchmark` (same engine).
  * train     — fit a single model (LR or Cox) + serialize to pkl for deployment
  * predict   — apply a serialized model to a new feature CSV
  * evaluate  — hold-out evaluation of a serialized model

Use `qr ml benchmark` (or the `qr bench` / `qr ml classify` aliases) for
model comparison.
Use `qr ml train` when you need a serialized model artifact.
"""
from __future__ import annotations

import json
import pickle
import warnings
from pathlib import Path

import click
import numpy as np
import pandas as pd

from qradiomics.cli._tracking import (
    log_artifact,
    log_dict,
    log_metrics,
    log_params,
    mlflow_run,
)
from qradiomics.cli.commands.bench import _shared_benchmark_options

# lifelines emits a multi-kilobyte ConvergenceWarning per fold listing every
# low-variance column. The information is real but it floods the CLI output —
# downstream filtering is the user's job; here we silence the worst offender.
warnings.filterwarnings("ignore", category=UserWarning, module="lifelines")
try:
    from lifelines.exceptions import ConvergenceWarning as _LLConvWarn
    warnings.filterwarnings("ignore", category=_LLConvWarn)
except Exception:
    pass


def _select_feature_columns(df: pd.DataFrame, exclude: set) -> list[str]:
    return [c for c in df.columns if c not in exclude and pd.api.types.is_numeric_dtype(df[c])]


def _drop_correlated(X: pd.DataFrame, threshold: float = 0.95) -> pd.DataFrame:
    """Drop features whose |Pearson r| with a kept feature exceeds threshold."""
    if X.shape[1] < 2:
        return X
    corr = X.corr().abs()
    # Iterate columns left-to-right; drop any later column highly correlated
    # with an earlier kept one. O(p²) memory but only run once.
    kept: list[str] = []
    kept_set: set[str] = set()
    for col in corr.columns:
        if kept and (corr.loc[col, kept] >= threshold).any():
            continue
        kept.append(col)
        kept_set.add(col)
    return X[kept]


def _univariate_cox_topk(X: pd.DataFrame, T: pd.Series, E: pd.Series,
                         time_col: str, event_col: str, k: int) -> list[str]:
    """Return the k features with smallest univariate Cox p-value.

    Falls back to absolute |risk ratio| when p-value is unavailable.
    """
    from lifelines import CoxPHFitter
    pvals: list[tuple[str, float]] = []
    for col in X.columns:
        try:
            cph = CoxPHFitter(penalizer=0.01)
            cph.fit(pd.concat([X[[col]], T, E], axis=1),
                    duration_col=time_col, event_col=event_col,
                    show_progress=False)
            p = float(cph.summary["p"].iloc[0])
            pvals.append((col, p))
        except Exception:
            continue
    pvals.sort(key=lambda x: x[1])
    return [c for c, _ in pvals[:k]]


def _select_for_survival(X_tr: pd.DataFrame, T_tr: pd.Series, E_tr: pd.Series,
                         time_col: str, event_col: str,
                         top_features: int, corr_threshold: float) -> list[str]:
    """Variance filter + corr drop + univariate-Cox top-k, computed only on
    the training subset. Returns ordered feature names to keep."""
    keep = [c for c in X_tr.columns if X_tr[c].std() > 1e-6]
    Xk = X_tr[keep]
    if Xk.shape[1] > top_features:
        Xk = _drop_correlated(Xk, corr_threshold)
    if Xk.shape[1] > top_features:
        chosen = _univariate_cox_topk(Xk, T_tr, E_tr, time_col, event_col, top_features)
        Xk = Xk[chosen]
    return list(Xk.columns)


def _make_group_aware_splitter(df: pd.DataFrame, y_for_stratify, folds: int,
                               group_col: str | None, stratify: bool):
    """Return (splitter, split_args, n_effective, groups) for a CV loop.

    Falls back to the original (non-grouped) splitter — same class, same
    random_state=42 shuffle — whenever `group_col` is absent or every group
    has exactly one row, so single-row-per-patient cohorts are unaffected.
    Only switches to Group(Stratified)KFold when a group genuinely repeats
    (multi-row-per-patient manifests), and clamps `folds` to the number of
    unique groups so no fold ends up empty.
    """
    from sklearn.model_selection import GroupKFold, KFold, StratifiedGroupKFold, StratifiedKFold

    n = len(df)
    groups = df[group_col] if group_col and group_col in df.columns else None
    if groups is not None and groups.nunique() < n:
        n_units = int(groups.nunique())
        folds = max(2, min(folds, n_units))
        if stratify:
            splitter = StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=42)
            split_args = (df.index, y_for_stratify, groups)
        else:
            splitter = GroupKFold(n_splits=folds, shuffle=True, random_state=42)
            split_args = (df.index, y_for_stratify, groups)
        return splitter, split_args, n_units, groups
    folds = max(2, min(folds, n // 2))
    if stratify:
        splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=42)
        split_args = (df.index, y_for_stratify)
    else:
        splitter = KFold(n_splits=folds, shuffle=True, random_state=42)
        split_args = (df.index,)
    return splitter, split_args, n, None


def _assert_no_group_leakage(groups, tr, te, fold_i, folds):
    if groups is None:
        return
    tr_groups = set(groups.iloc[tr])
    te_groups = set(groups.iloc[te])
    overlap = tr_groups & te_groups
    if overlap:
        raise RuntimeError(
            f"Group leakage detected in fold {fold_i}/{folds}: "
            f"{len(overlap)} group(s) present in both train and test "
            f"({sorted(overlap)[:5]}{'...' if len(overlap) > 5 else ''})"
        )


def _train_survival(df: pd.DataFrame, time_col: str, event_col: str, folds: int,
                    top_features: int = 50, corr_threshold: float = 0.95,
                    group_col: str | None = "patient_id"):
    from lifelines import CoxPHFitter
    from lifelines.utils import concordance_index

    from qradiomics.analytics import survival_calibration as _surv_cal

    exclude = {"patient_id", "PatientID", time_col, event_col}
    feats = _select_feature_columns(df, exclude)
    print(f"  [survival] {len(df)} rows, {len(feats)} candidate features", flush=True)
    X = df[feats].fillna(df[feats].median())
    T = df[time_col].astype(float)
    E = df[event_col].astype(float)
    print(f"  [survival] events={int(E.sum())}/{len(E)}", flush=True)

    # CV: select features INSIDE each fold to avoid leakage. Also group-aware:
    # when `group_col` repeats (e.g. multiple rows per patient_id), folds are
    # split on unique groups so no group's rows land in both train and test.
    n = len(X)
    splitter, split_args, n_units, groups = _make_group_aware_splitter(
        df, None, folds, group_col, stratify=False)
    folds = splitter.get_n_splits()
    fold_c: list = []
    fold_feature_counts: list[int] = []
    oof_T: list[float] = []
    oof_E: list[float] = []
    oof_risk: list[float] = []
    fold_ibs: list[float] = []
    if n >= 4:
        unit_msg = f", {n_units} unique {group_col}" if groups is not None else ""
        print(f"  [survival] cross-validating ({folds}-fold{unit_msg}, "
              f"feature selection per fold)", flush=True)
        for i, (tr, te) in enumerate(splitter.split(*split_args), 1):
            _assert_no_group_leakage(groups, tr, te, i, folds)
            try:
                X_tr_all = X.iloc[tr]
                T_tr = T.iloc[tr]; E_tr = E.iloc[tr]
                chosen = _select_for_survival(
                    X_tr_all, T_tr, E_tr, time_col, event_col,
                    top_features, corr_threshold,
                )
                fold_feature_counts.append(len(chosen))
                if not chosen:
                    print(f"    fold {i}/{folds}: 0 features survived selection", flush=True)
                    continue
                X_tr = X_tr_all[chosen]
                X_te = X.iloc[te][chosen]
                train = pd.concat([X_tr, T_tr, E_tr], axis=1)
                cph = CoxPHFitter(penalizer=0.1)
                cph.fit(train, duration_col=time_col, event_col=event_col,
                        show_progress=False)
                risk = cph.predict_partial_hazard(X_te)
                c = concordance_index(T.iloc[te], -risk, E.iloc[te])
                fold_c.append(c)
                # Accumulate out-of-fold risk for a pooled c-index CI, and an
                # Integrated Brier Score per fold (survival calibration).
                oof_T.extend(np.asarray(T.iloc[te], dtype=float).tolist())
                oof_E.extend(np.asarray(E.iloc[te], dtype=float).tolist())
                oof_risk.extend(np.asarray(risk, dtype=float).tolist())
                ibs = _surv_cal.fold_integrated_brier(
                    cph, X_te, T_tr, E_tr, T.iloc[te], E.iloc[te])
                if not np.isnan(ibs):
                    fold_ibs.append(ibs)
                print(f"    fold {i}/{folds}: c-index={c:.3f} "
                      f"(train={len(tr)}, test={len(te)}, k={len(chosen)})",
                      flush=True)
            except Exception as e:
                print(f"    fold {i}/{folds}: failed ({type(e).__name__})", flush=True)
                continue
    else:
        print(f"  [survival] n={n} < 4, skipping CV", flush=True)

    # Discrimination uncertainty (C06) + survival calibration (C05).
    c_ci_lo = c_ci_hi = None
    ibs_mean = ibs_std = None
    if fold_c:
        _, c_ci_lo, c_ci_hi = _surv_cal.concordance_ci(oof_T, oof_risk, oof_E)
        if fold_ibs:
            ibs_mean = float(np.mean(fold_ibs))
            ibs_std = float(np.std(fold_ibs))
        ibs_msg = f"  IBS={ibs_mean:.3f}" if ibs_mean is not None else ""
        print(f"  [survival] pooled c-index 95% CI = "
              f"[{c_ci_lo:.3f}, {c_ci_hi:.3f}]{ibs_msg}", flush=True)

    # Final model: select on all data, then fit.
    print(f"  [survival] fitting final model on all {n} samples", flush=True)
    final_keep = _select_for_survival(X, T, E, time_col, event_col,
                                       top_features, corr_threshold)
    print(f"  [survival] final feature set: {len(final_keep)} features", flush=True)
    X_final = X[final_keep]
    final = CoxPHFitter(penalizer=0.1)
    final.fit(pd.concat([X_final, T, E], axis=1), duration_col=time_col,
              event_col=event_col, show_progress=False)
    return final, {
        "task": "survival",
        "n": int(n),
        "estimator": "CoxPHFitter(penalizer=0.1)",
        "folds": folds if fold_c else 0,
        "cv_c_index_mean": float(np.mean(fold_c)) if fold_c else None,
        "cv_c_index_std": float(np.std(fold_c)) if fold_c else None,
        "cv_c_index_ci_lo": c_ci_lo,
        "cv_c_index_ci_hi": c_ci_hi,
        "cv_integrated_brier_mean": ibs_mean,
        "cv_integrated_brier_std": ibs_std,
        "cv_c_index_folds": [float(c) for c in fold_c],
        "cv_features_per_fold": fold_feature_counts,
        "features": final_keep,
        "selection": {"corr_threshold": corr_threshold, "top_features": top_features,
                      "leakage_safe": True},
        "group_col": group_col if groups is not None else None,
        "n_groups": n_units if groups is not None else None,
        "note": ("CV skipped: n<4" if not fold_c else None),
    }


def _univariate_classify_topk(X: pd.DataFrame, y: pd.Series, k: int) -> list[str]:
    """Return k features ranked by univariate logistic Wald |z|.

    Falls back to absolute Pearson r if logistic fit fails.
    """
    from sklearn.linear_model import LogisticRegression
    scores: list[tuple[str, float]] = []
    for col in X.columns:
        try:
            clf = LogisticRegression(max_iter=200, C=1.0)
            clf.fit(X[[col]].values, y.values)
            scores.append((col, abs(float(clf.coef_[0][0]))))
        except Exception:
            try:
                r = float(X[col].corr(y))
                scores.append((col, abs(r)))
            except Exception:
                continue
    scores.sort(key=lambda x: x[1], reverse=True)
    return [c for c, _ in scores[:k]]


def _select_for_classify(X_tr: pd.DataFrame, y_tr: pd.Series,
                         top_features: int, corr_threshold: float) -> list[str]:
    """Variance + corr drop + univariate logistic top-k on the train fold only."""
    keep = [c for c in X_tr.columns if X_tr[c].std() > 1e-6]
    Xk = X_tr[keep]
    if Xk.shape[1] > top_features:
        Xk = _drop_correlated(Xk, corr_threshold)
    if Xk.shape[1] > top_features:
        chosen = _univariate_classify_topk(Xk, y_tr, top_features)
        Xk = Xk[chosen]
    return list(Xk.columns)


def _train_classify(df: pd.DataFrame, outcome: str, folds: int,
                    top_features: int = 50, corr_threshold: float = 0.95,
                    group_col: str | None = "patient_id"):
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import classification_report, roc_auc_score
    from sklearn.preprocessing import StandardScaler

    from qradiomics.analytics import classification_calibration as _clf_cal
    from qradiomics.analytics import decision_curve as _dca

    exclude = {"patient_id", "PatientID", outcome}
    feats = _select_feature_columns(df, exclude)
    print(f"  [classify] {len(df)} rows, {len(feats)} candidate features", flush=True)
    X = df[feats].fillna(df[feats].median())
    y = df[outcome].astype(int)
    classes = sorted(y.unique().tolist())
    n_classes = len(classes)
    is_multiclass = n_classes > 2
    if is_multiclass:
        print(f"  [classify] multi-class: {n_classes} classes {classes}, "
              f"counts={y.value_counts().sort_index().to_dict()}", flush=True)
    else:
        pos = int(y.sum()); neg = len(y) - pos
        print(f"  [classify] pos={pos}, neg={neg}", flush=True)

    # CV: select features INSIDE each fold to avoid leakage. Also group-aware:
    # when `group_col` repeats (e.g. multiple rows per patient_id), folds are
    # split on unique groups so no group's rows land in both train and test.
    n = len(X)
    splitter, split_args, n_units, groups = _make_group_aware_splitter(
        df, y, folds, group_col, stratify=True)
    folds = splitter.get_n_splits()
    fold_auc: list = []
    fold_feature_counts: list[int] = []
    oof_y: list[float] = []
    oof_p: list[float] = []       # binary: P(y=1) per row. multiclass: unused.
    oof_pred: list = []           # multiclass only: argmax predicted label per row.
    if n >= 4 and n_classes > 1:
        unit_msg = f", {n_units} unique {group_col}" if groups is not None else ""
        print(f"  [classify] cross-validating ({folds}-fold stratified{unit_msg}, "
              f"feature selection per fold)", flush=True)
        for i, (tr, te) in enumerate(splitter.split(*split_args), 1):
            _assert_no_group_leakage(groups, tr, te, i, folds)
            try:
                X_tr_all = X.iloc[tr]
                y_tr = y.iloc[tr]
                chosen = _select_for_classify(X_tr_all, y_tr,
                                              top_features, corr_threshold)
                fold_feature_counts.append(len(chosen))
                if not chosen:
                    print(f"    fold {i}/{folds}: 0 features survived selection", flush=True)
                    continue
                X_tr = X_tr_all[chosen]
                X_te = X.iloc[te][chosen]
                scaler = StandardScaler().fit(X_tr)
                clf = LogisticRegression(max_iter=500, C=0.1)
                clf.fit(scaler.transform(X_tr), y_tr)
                proba_full = clf.predict_proba(scaler.transform(X_te))
                y_te = y.iloc[te]
                if is_multiclass:
                    auc = roc_auc_score(y_te, proba_full, multi_class="ovr",
                                        labels=clf.classes_)
                    oof_y.extend(np.asarray(y_te, dtype=float).tolist())
                    oof_pred.extend(clf.classes_[np.argmax(proba_full, axis=1)].tolist())
                else:
                    proba = proba_full[:, 1]
                    auc = roc_auc_score(y_te, proba)
                    oof_y.extend(np.asarray(y_te, dtype=float).tolist())
                    oof_p.extend(np.asarray(proba, dtype=float).tolist())
                fold_auc.append(auc)
                print(f"    fold {i}/{folds}: AUC={auc:.3f} "
                      f"(train={len(tr)}, test={len(te)}, k={len(chosen)})",
                      flush=True)
            except Exception as e:
                print(f"    fold {i}/{folds}: failed ({type(e).__name__})", flush=True)
                continue
    else:
        print(f"  [classify] n={n} or single-class, skipping CV", flush=True)

    # Discrimination uncertainty (C06) + calibration quality (C05) on the
    # pooled out-of-fold predictions. Binary only — brier/ECE/decision-curve
    # are defined for a single P(y=1) column; see classification_calibration
    # module docstring. Multi-class gets a per-class classification_report
    # instead (precision/recall/F1 per class + macro/weighted averages).
    auc_ci_lo = auc_ci_hi = cv_brier = cv_ece = None
    cv_decision_curve = None
    cv_classification_report = None
    if is_multiclass and fold_auc and oof_y:
        cv_classification_report = classification_report(
            oof_y, oof_pred, output_dict=True, zero_division=0)
        print(f"  [classify] pooled OVR AUC (macro) over {len(oof_y)} "
              f"out-of-fold rows: {np.mean(fold_auc):.3f}", flush=True)
    elif fold_auc and oof_y:
        _, auc_ci_lo, auc_ci_hi = _clf_cal.auc_ci(oof_y, oof_p)
        cv_brier, cv_ece = _clf_cal.brier_ece(oof_y, oof_p)
        cv_decision_curve = _dca.decision_curve_summary(oof_y, oof_p)
        _useful = [t for t, d in cv_decision_curve.items() if d["beats_default"]]
        print(f"  [classify] pooled AUC 95% CI = [{auc_ci_lo:.3f}, {auc_ci_hi:.3f}]  "
              f"Brier={cv_brier:.3f} ECE={cv_ece:.3f}  "
              f"net-benefit>default at thresholds {_useful or 'none'}", flush=True)

    # Final model: select on all data, then fit. LogisticRegression handles
    # multi-class natively (multinomial) — no change needed for the fit itself.
    print(f"  [classify] fitting final model on all {n} samples", flush=True)
    final_keep = _select_for_classify(X, y, top_features, corr_threshold)
    print(f"  [classify] final feature set: {len(final_keep)} features", flush=True)
    X_final = X[final_keep]
    scaler = StandardScaler().fit(X_final)
    final = LogisticRegression(max_iter=500, C=0.1).fit(scaler.transform(X_final), y)
    return (
        {"scaler": scaler, "classifier": final, "features": final_keep},
        {
            "task": "classify",
            "n": int(n),
            "n_classes": n_classes,
            "classes": classes,
            "estimator": "LogisticRegression(C=0.1)",
            "folds": folds if fold_auc else 0,
            # cv_auc_mean/std: binary ROC-AUC, or macro one-vs-rest AUC when
            # n_classes > 2 — same field, so existing consumers (e.g.
            # pipelines/*/run.sh summary printers) keep working unchanged.
            "cv_auc_mean": float(np.mean(fold_auc)) if fold_auc else None,
            "cv_auc_std": float(np.std(fold_auc)) if fold_auc else None,
            "cv_auc_ci_lo": auc_ci_lo,
            "cv_auc_ci_hi": auc_ci_hi,
            "cv_brier": cv_brier,
            "cv_ece": cv_ece,
            "cv_decision_curve": cv_decision_curve,
            "cv_classification_report": cv_classification_report,
            "cv_auc_folds": [float(a) for a in fold_auc],
            "cv_features_per_fold": fold_feature_counts,
            "features": final_keep,
            "selection": {"corr_threshold": corr_threshold,
                          "top_features": top_features, "leakage_safe": True},
            "group_col": group_col if groups is not None else None,
            "n_groups": n_units if groups is not None else None,
            "note": ("CV skipped: n<4 or single class" if not fold_auc else None),
        },
    )


@click.group()
def ml():
    """Train, evaluate, and predict with outcome-prediction models."""


@ml.command("train")
@click.option("--input", "-i", "input_path", required=True, type=click.Path(exists=True),
              help="analysis_ready.csv (features + outcome columns)")
@click.option("--task", type=click.Choice(["survival", "classify"]), required=True)
@click.option("--outcome", required=True, help="Outcome column (event or binary label)")
@click.option("--time-col", default="OS_months",
              help="Survival time column (survival task only; default OS_months)")
@click.option("--folds", default=5, help="Cross-validation folds (default 5)")
@click.option("--top-features", default=50, type=int,
              help="Maximum features retained after correlation drop + "
                   "univariate ranking (default 50). Higher values risk "
                   "Cox/logistic convergence problems on small cohorts.")
@click.option("--corr-threshold", default=0.95, type=float,
              help="Drop one of each pair with |Pearson r| above this "
                   "(default 0.95). Set to 1.0 to disable.")
@click.option("--group-col", default="patient_id", show_default=True,
              help="Column identifying rows that must never split across "
                   "train/test folds (e.g. multiple images or lesions per "
                   "patient). CV falls back to plain K-fold, unchanged, "
                   "when every value in this column is unique or the "
                   "column is absent. Set to '' to disable grouping "
                   "explicitly.")
@click.option("--model", "model_path", required=True, type=click.Path(),
              help="Output path for the fitted model (.pkl)")
@click.option("--metrics", "metrics_path", required=True, type=click.Path(),
              help="Output path for the CV metrics JSON")
@click.option("--experiment", default=None,
              help="MLflow experiment name (default: input file's parent dir; "
                   "no-op when MLFLOW_TRACKING_URI is unset)")
@click.option("--run-name", default=None,
              help="MLflow run name (default: derived from input filename)")
def train(input_path, task, outcome, time_col, folds, top_features,
          corr_threshold, group_col, model_path, metrics_path, experiment, run_name):
    """Fit a CV'd survival or classification model on radiomics features."""
    click.echo(f"qr ml train: task={task} outcome={outcome} input={input_path}")
    click.echo(f"  feature reduction: corr<{corr_threshold}, top-{top_features}")
    df = pd.read_csv(input_path)
    click.echo(f"  loaded {len(df)} rows × {len(df.columns)} cols")
    group_col = group_col or None

    input_p = Path(input_path)
    exp_name = experiment or input_p.parent.name or "qr_ml"
    rn = run_name or f"{input_p.stem}-{task}"

    with mlflow_run(experiment=exp_name, run_name=rn) as run_id:
        log_params({
            "task": task,
            "outcome": outcome,
            "time_col": time_col if task == "survival" else "",
            "folds": folds,
            "top_features": top_features,
            "corr_threshold": corr_threshold,
            "group_col": group_col or "",
            "input": input_p.name,
            "n_rows": len(df),
            "n_input_cols": len(df.columns),
        })

        if task == "survival":
            if outcome not in df.columns:
                raise click.UsageError(f"Outcome column '{outcome}' not in input")
            model, metrics = _train_survival(df, time_col, outcome, folds,
                                             top_features, corr_threshold, group_col)
        else:
            if outcome not in df.columns:
                raise click.UsageError(f"Outcome column '{outcome}' not in input")
            model, metrics = _train_classify(df, outcome, folds,
                                             top_features, corr_threshold, group_col)

        if run_id:
            metrics["mlflow_run_id"] = run_id

        Path(model_path).parent.mkdir(parents=True, exist_ok=True)
        with open(model_path, "wb") as f:
            pickle.dump(model, f)
        Path(metrics_path).parent.mkdir(parents=True, exist_ok=True)
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)

        scalar_metrics = {k: v for k, v in metrics.items()
                          if isinstance(v, (int, float)) and not isinstance(v, bool)}
        scalar_metrics["n_selected_features"] = len(metrics.get("features", []))
        log_metrics(scalar_metrics)
        log_dict(metrics, "cv_metrics.json")
        log_artifact(model_path)

    click.echo(f"Model -> {model_path}")
    click.echo(f"Metrics -> {metrics_path}")
    if task == "survival":
        click.echo(f"  CV c-index: {metrics['cv_c_index_mean']:.3f} ± "
                   f"{metrics['cv_c_index_std']:.3f}")
    else:
        click.echo(f"  CV AUC: {metrics['cv_auc_mean']:.3f} ± "
                   f"{metrics['cv_auc_std']:.3f}")
    if metrics.get("mlflow_run_id"):
        click.echo(f"  MLflow run: {metrics['mlflow_run_id']}")


@ml.command("predict")
@click.option("--input", "-i", "input_path", required=True, type=click.Path(exists=True),
              help="features.csv (or analysis_ready.csv)")
@click.option("--model", required=True, type=click.Path(exists=True),
              help="Trained model .pkl")
@click.option("--task", type=click.Choice(["survival", "classify"]), required=True)
@click.option("--output", "-o", required=True, type=click.Path(),
              help="Output predictions CSV (patient_id + risk/proba)")
def predict(input_path, model, task, output):
    """Apply a trained model to a new feature CSV."""
    df = pd.read_csv(input_path)
    with open(model, "rb") as f:
        m = pickle.load(f)

    if task == "survival":
        feats = m.summary.index.tolist()
        X = df[feats].fillna(df[feats].median())
        risk = m.predict_partial_hazard(X)
        out = pd.DataFrame({"patient_id": df["patient_id"], "risk": risk.values})
    else:
        feats = m["features"]
        X = df[feats].fillna(df[feats].median())
        clf = m["classifier"]
        proba_full = clf.predict_proba(m["scaler"].transform(X))
        if len(clf.classes_) > 2:
            out = pd.DataFrame(proba_full, columns=[f"proba_{c}" for c in clf.classes_])
            out.insert(0, "patient_id", df["patient_id"].values)
            out["pred_class"] = clf.classes_[np.argmax(proba_full, axis=1)]
        else:
            out = pd.DataFrame({"patient_id": df["patient_id"], "proba": proba_full[:, 1]})

    Path(output).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output, index=False)
    click.echo(f"Predictions ({len(out)}) -> {output}")


@ml.command("evaluate")
@click.option("--input", "-i", "input_path", required=True, type=click.Path(exists=True))
@click.option("--model", required=True, type=click.Path(exists=True))
@click.option("--task", type=click.Choice(["survival", "classify"]), required=True)
@click.option("--outcome", required=True)
@click.option("--time-col", default="OS_months")
@click.option("--report", required=True, type=click.Path(),
              help="Output evaluation report JSON")
@click.option("--metrics", "train_metrics_path", default=None, type=click.Path(),
              help="Train-time cv_metrics.json to attach holdout metrics to the "
                   "same MLflow run (default: <model>.json sibling)")
@click.option("--experiment", default=None,
              help="MLflow experiment name (used only when no prior train run is found)")
def evaluate(input_path, model, task, outcome, time_col, report,
             train_metrics_path, experiment):
    """Apply a trained model and compute hold-out performance metrics."""
    click.echo(f"qr ml evaluate: task={task} model={model}")
    df = pd.read_csv(input_path)
    click.echo(f"  loaded {len(df)} rows")
    with open(model, "rb") as f:
        m = pickle.load(f)

    train_run_id = None
    if train_metrics_path is None:
        guess = Path(model).with_suffix(".json")
        if guess.exists():
            train_metrics_path = str(guess)
    if train_metrics_path and Path(train_metrics_path).exists():
        try:
            train_run_id = json.loads(Path(train_metrics_path).read_text()).get("mlflow_run_id")
        except (json.JSONDecodeError, OSError):
            train_run_id = None

    input_p = Path(input_path)
    exp_name = experiment or input_p.parent.name or "qr_ml"
    rn = f"{input_p.stem}-{task}-eval"

    with mlflow_run(experiment=exp_name, run_name=rn, run_id=train_run_id) as run_id:
        if task == "survival":
            from lifelines.utils import concordance_index

            feats = m.summary.index.tolist()
            X = df[feats].fillna(df[feats].median())
            risk = m.predict_partial_hazard(X)
            c = concordance_index(df[time_col], -risk, df[outcome])
            result = {"task": "survival", "c_index": float(c), "n": int(len(df))}
            log_metrics({"holdout_c_index": c, "holdout_n": len(df)})
        else:
            from sklearn.metrics import classification_report, roc_auc_score

            feats = m["features"]
            X = df[feats].fillna(df[feats].median())
            clf = m["classifier"]
            proba_full = clf.predict_proba(m["scaler"].transform(X))
            y_true = df[outcome]
            if len(clf.classes_) > 2:
                auc = roc_auc_score(y_true, proba_full, multi_class="ovr",
                                    labels=clf.classes_)
                pred = clf.classes_[np.argmax(proba_full, axis=1)]
                result = {
                    "task": "classify", "n_classes": len(clf.classes_),
                    "auc": float(auc), "n": int(len(df)),
                    "classification_report": classification_report(
                        y_true, pred, output_dict=True, zero_division=0),
                }
                log_metrics({"holdout_auc": auc, "holdout_n": len(df)})
            else:
                proba = proba_full[:, 1]
                auc = roc_auc_score(y_true, proba)
                result = {"task": "classify", "auc": float(auc), "n": int(len(df))}
                log_metrics({"holdout_auc": auc, "holdout_n": len(df)})

        if run_id:
            result["mlflow_run_id"] = run_id
            log_dict(result, "evaluation.json")

        Path(report).parent.mkdir(parents=True, exist_ok=True)
        with open(report, "w") as f:
            json.dump(result, f, indent=2)
        log_artifact(report)

    click.echo(json.dumps(result))


@ml.command("classify")
@_shared_benchmark_options
@click.option("--corr-threshold", default=0.95, type=float, show_default=True,
              help="Drop one of each pair with |Pearson r| above this before training.")
@click.option("--output-dir", "-d", default="results/classify", show_default=True,
              type=click.Path(), help="Output directory for CSV + HTML report.")
def classify(input_path, outcome, models, all_models, cv, inner_cv, seed,
             n_jobs, corr_threshold, hpo, ext_path, tpot_repeats,
             calibrate_threshold, output_dir, exclude):
    """Multi-model classification benchmark with correlation-based feature pre-filter.

    Runs nested CV over 8 classifiers by default (LR, SVM, ExtraTrees, RF, GBM,
    MLP, XGB, LGBM) and reports OOF ROC-AUC. Use --all-models to add FLAML.
    TPOT is opt-in only (--models TPOT) — never pulled in by --all-models,
    both for cost and because its genetic search is not reproducible from a
    single run (its CV fold-mean/std therefore mixes real fold variance with
    search noise); use --tpot-repeats on --external-test for a stable
    mean +/- std instead of trusting one run. --calibrate-threshold applies
    to any model, not just TPOT.

    See also: qr bench (same benchmark without the feature pre-filter step)
    and qr ml benchmark — the recommended canonical entry point that unifies
    this command with qr bench on one engine. This command remains a
    permanent backward-compatible alias, not deprecated.
    """
    from qradiomics.cli.commands.bench import _run_benchmark

    df = pd.read_csv(input_path)
    _run_benchmark(
        df, outcome,
        label="classify",
        output_dir=output_dir,
        csv_prefix="classify",
        models=models,
        all_models=all_models,
        cv=cv, inner_cv=inner_cv,
        seed=seed, n_jobs=n_jobs,
        # `classify` never exposed --optuna-trials; preserve its historical
        # reliance on cross_val_benchmark's own default (20) rather than
        # bench's --optuna-trials default (30).
        hpo=hpo, optuna_trials=20,
        ext_path=ext_path,
        tpot_repeats=tpot_repeats,
        calibrate_threshold=calibrate_threshold,
        exclude=exclude,
        extra_exclude=("patient_id", "PatientID"),
        fillna_median=True,
        restrict_train_to_shared=True,
        corr_threshold=corr_threshold,
        features_echo_label="raw_features",
        best_label="Best",
    )


@ml.command("benchmark")
@_shared_benchmark_options
@click.option("--corr-threshold", default=0.95, type=float, show_default=True,
              help="Drop one of each pair with |Pearson r| above this before training.")
@click.option("--optuna-trials", default=30, show_default=True,
              help="Optuna trial count (only when --hpo optuna).")
@click.option("--output-dir", "-d", default="results/benchmark", show_default=True,
              type=click.Path(), help="Output directory for CSV + HTML report.")
def benchmark(input_path, outcome, models, all_models, cv, inner_cv, seed,
              n_jobs, corr_threshold, hpo, optuna_trials, ext_path,
              tpot_repeats, calibrate_threshold, output_dir, exclude):
    """Canonical multi-model classification benchmark (recommended entry point).

    Unifies `qr bench` and `qr ml classify` on one engine
    (qradiomics.cli.commands.bench._run_benchmark): nested CV over 8
    classifiers by default (LR, SVM, ExtraTrees, RF, GBM, MLP, XGB, LGBM;
    --all-models adds FLAML; TPOT is opt-in only via --models TPOT), an
    optional correlation-based feature pre-filter (--corr-threshold, as in
    `qr ml classify`), optional external-test refit/scoring with TPOT-repeat
    aggregation and Youden's-J threshold calibration (--tpot-repeats /
    --calibrate-threshold, as in both `qr bench` and `qr ml classify`), and
    an HTML report.

    `qr bench` and `qr ml classify` remain permanent backward-compatible
    aliases of this same engine — neither is deprecated.
    """
    from qradiomics.cli.commands.bench import _run_benchmark

    df = pd.read_csv(input_path)
    _run_benchmark(
        df, outcome,
        label="ml benchmark",
        output_dir=output_dir,
        csv_prefix="benchmark",
        models=models,
        all_models=all_models,
        cv=cv, inner_cv=inner_cv,
        seed=seed, n_jobs=n_jobs,
        hpo=hpo, optuna_trials=optuna_trials,
        ext_path=ext_path,
        tpot_repeats=tpot_repeats,
        calibrate_threshold=calibrate_threshold,
        exclude=exclude,
        extra_exclude=("patient_id", "PatientID"),
        fillna_median=True,
        restrict_train_to_shared=True,
        corr_threshold=corr_threshold,
        features_echo_label="raw_features",
        best_label="Best",
        title_prefix="qr ml benchmark",
    )
