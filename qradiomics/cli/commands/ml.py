"""qr ml — train / evaluate / predict outcome-prediction models.

Atomic ML primitives that close the radiomics 4-stage data flow:

    data → image → features → **modeling**

Two tasks supported out of the box:
  * survival — Cox proportional hazards with k-fold cross-validated c-index
  * classify — logistic regression with k-fold cross-validated ROC-AUC

Both serialize the fitted estimator to a `.pkl` and emit a JSON metrics
report. Larger / non-linear models (Random Survival Forest, XGBoost,
SHAP-explained tree models) belong in `qradiomics_lab` (the private
extension) — the public CLI keeps the canonical baseline.
"""
from __future__ import annotations

import json
import pickle
import warnings
from pathlib import Path

import click
import numpy as np
import pandas as pd

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


def _train_survival(df: pd.DataFrame, time_col: str, event_col: str, folds: int,
                    top_features: int = 50, corr_threshold: float = 0.95):
    from lifelines import CoxPHFitter
    from lifelines.utils import concordance_index
    from sklearn.model_selection import KFold

    exclude = {"patient_id", "PatientID", time_col, event_col}
    feats = _select_feature_columns(df, exclude)
    print(f"  [survival] {len(df)} rows, {len(feats)} candidate features", flush=True)
    X = df[feats].fillna(df[feats].median())
    T = df[time_col].astype(float)
    E = df[event_col].astype(float)
    print(f"  [survival] events={int(E.sum())}/{len(E)}", flush=True)

    # CV: select features INSIDE each fold to avoid leakage.
    n = len(X)
    folds = max(2, min(folds, n // 2))
    fold_c: list = []
    fold_feature_counts: list[int] = []
    if n >= 4:
        print(f"  [survival] cross-validating ({folds}-fold, feature selection per fold)",
              flush=True)
        kf = KFold(n_splits=folds, shuffle=True, random_state=42)
        for i, (tr, te) in enumerate(kf.split(X), 1):
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
                print(f"    fold {i}/{folds}: c-index={c:.3f} "
                      f"(train={len(tr)}, test={len(te)}, k={len(chosen)})",
                      flush=True)
            except Exception as e:
                print(f"    fold {i}/{folds}: failed ({type(e).__name__})", flush=True)
                continue
    else:
        print(f"  [survival] n={n} < 4, skipping CV", flush=True)

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
        "cv_c_index_folds": [float(c) for c in fold_c],
        "cv_features_per_fold": fold_feature_counts,
        "features": final_keep,
        "selection": {"corr_threshold": corr_threshold, "top_features": top_features,
                      "leakage_safe": True},
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
                    top_features: int = 50, corr_threshold: float = 0.95):
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedKFold
    from sklearn.preprocessing import StandardScaler

    exclude = {"patient_id", "PatientID", outcome}
    feats = _select_feature_columns(df, exclude)
    print(f"  [classify] {len(df)} rows, {len(feats)} candidate features", flush=True)
    X = df[feats].fillna(df[feats].median())
    y = df[outcome].astype(int)
    pos = int(y.sum()); neg = len(y) - pos
    print(f"  [classify] pos={pos}, neg={neg}", flush=True)

    # CV: select features INSIDE each fold to avoid leakage.
    n = len(X)
    folds = max(2, min(folds, n // 2))
    fold_auc: list = []
    fold_feature_counts: list[int] = []
    if n >= 4 and len(set(y)) > 1:
        print(f"  [classify] cross-validating ({folds}-fold stratified, "
              f"feature selection per fold)", flush=True)
        skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=42)
        for i, (tr, te) in enumerate(skf.split(X, y), 1):
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
                proba = clf.predict_proba(scaler.transform(X_te))[:, 1]
                auc = roc_auc_score(y.iloc[te], proba)
                fold_auc.append(auc)
                print(f"    fold {i}/{folds}: AUC={auc:.3f} "
                      f"(train={len(tr)}, test={len(te)}, k={len(chosen)})",
                      flush=True)
            except Exception as e:
                print(f"    fold {i}/{folds}: failed ({type(e).__name__})", flush=True)
                continue
    else:
        print(f"  [classify] n={n} or single-class, skipping CV", flush=True)

    # Final model: select on all data, then fit.
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
            "estimator": "LogisticRegression(C=0.1)",
            "folds": folds if fold_auc else 0,
            "cv_auc_mean": float(np.mean(fold_auc)) if fold_auc else None,
            "cv_auc_std": float(np.std(fold_auc)) if fold_auc else None,
            "cv_auc_folds": [float(a) for a in fold_auc],
            "cv_features_per_fold": fold_feature_counts,
            "features": final_keep,
            "selection": {"corr_threshold": corr_threshold,
                          "top_features": top_features, "leakage_safe": True},
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
@click.option("--model", "model_path", required=True, type=click.Path(),
              help="Output path for the fitted model (.pkl)")
@click.option("--metrics", "metrics_path", required=True, type=click.Path(),
              help="Output path for the CV metrics JSON")
def train(input_path, task, outcome, time_col, folds, top_features,
          corr_threshold, model_path, metrics_path):
    """Fit a CV'd survival or classification model on radiomics features."""
    click.echo(f"qr ml train: task={task} outcome={outcome} input={input_path}")
    click.echo(f"  feature reduction: corr<{corr_threshold}, top-{top_features}")
    df = pd.read_csv(input_path)
    click.echo(f"  loaded {len(df)} rows × {len(df.columns)} cols")
    if task == "survival":
        if outcome not in df.columns:
            raise click.UsageError(f"Outcome column '{outcome}' not in input")
        model, metrics = _train_survival(df, time_col, outcome, folds,
                                         top_features, corr_threshold)
    else:
        if outcome not in df.columns:
            raise click.UsageError(f"Outcome column '{outcome}' not in input")
        model, metrics = _train_classify(df, outcome, folds,
                                         top_features, corr_threshold)

    Path(model_path).parent.mkdir(parents=True, exist_ok=True)
    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    Path(metrics_path).parent.mkdir(parents=True, exist_ok=True)
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    click.echo(f"Model -> {model_path}")
    click.echo(f"Metrics -> {metrics_path}")
    if task == "survival":
        click.echo(f"  CV c-index: {metrics['cv_c_index_mean']:.3f} ± "
                   f"{metrics['cv_c_index_std']:.3f}")
    else:
        click.echo(f"  CV AUC: {metrics['cv_auc_mean']:.3f} ± "
                   f"{metrics['cv_auc_std']:.3f}")


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
        proba = m["classifier"].predict_proba(m["scaler"].transform(X))[:, 1]
        out = pd.DataFrame({"patient_id": df["patient_id"], "proba": proba})

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
def evaluate(input_path, model, task, outcome, time_col, report):
    """Apply a trained model and compute hold-out performance metrics."""
    click.echo(f"qr ml evaluate: task={task} model={model}")
    df = pd.read_csv(input_path)
    click.echo(f"  loaded {len(df)} rows")
    with open(model, "rb") as f:
        m = pickle.load(f)

    if task == "survival":
        from lifelines.utils import concordance_index

        feats = m.summary.index.tolist()
        X = df[feats].fillna(df[feats].median())
        risk = m.predict_partial_hazard(X)
        c = concordance_index(df[time_col], -risk, df[outcome])
        result = {"task": "survival", "c_index": float(c), "n": int(len(df))}
    else:
        from sklearn.metrics import roc_auc_score

        feats = m["features"]
        X = df[feats].fillna(df[feats].median())
        proba = m["classifier"].predict_proba(m["scaler"].transform(X))[:, 1]
        auc = roc_auc_score(df[outcome], proba)
        result = {"task": "classify", "auc": float(auc), "n": int(len(df))}

    Path(report).parent.mkdir(parents=True, exist_ok=True)
    with open(report, "w") as f:
        json.dump(result, f, indent=2)
    click.echo(json.dumps(result))
