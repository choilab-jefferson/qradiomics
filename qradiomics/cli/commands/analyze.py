"""Analyze command group — survival analysis and statistical reporting."""

import os

import click
import pandas as pd


@click.group()
def analyze():
    """Statistical analysis commands.

    \b
    Examples:
        qr analyze survival --input analysis_ready.csv --outcome OS_months --event OS_event
    """
    pass


@analyze.command()
@click.option(
    "--input",
    "-i",
    "input_path",
    required=True,
    type=click.Path(exists=True),
    help="Path to analysis_ready.csv (patient_id + OS_months + OS_event + features)",
)
@click.option("--outcome", default="OS_months", help="Survival time column (default: OS_months)")
@click.option("--event", default="OS_event", help="Event column, 1=event (default: OS_event)")
@click.option(
    "--output",
    "-o",
    required=True,
    type=click.Path(),
    help="Output CSV path for Cox PH results",
)
@click.option("--top-n", default=20, help="Features to show in console summary (default: 20)")
def survival(input_path, outcome, event, output, top_n):
    """Run univariate Cox Proportional Hazards analysis on radiomics features.

    \b
    Features are ranked by p-value. Output CSV contains: feature, coef, hr,
    hr_ci_low, hr_ci_high, p.

    \b
    Examples:
        qr analyze survival \\
          --input artifacts/uploads/datasets/NSCLC/analysis_ready.csv \\
          --outcome OS_months --event OS_event \\
          --output artifacts/uploads/datasets/NSCLC/cox_results.csv
    """
    try:
        from lifelines import CoxPHFitter
    except ImportError:
        click.echo("lifelines not installed. Run: pip install lifelines")
        raise SystemExit(1)

    df = pd.read_csv(input_path)

    if outcome not in df.columns:
        click.echo(f"Outcome column '{outcome}' not found in {input_path}")
        click.echo(f"   Available columns: {', '.join(list(df.columns)[:10])}...")
        raise SystemExit(1)
    if event not in df.columns:
        click.echo(f"Event column '{event}' not found in {input_path}")
        raise SystemExit(1)

    exclude = {"patient_id", "PatientID", outcome, event}
    feature_cols = [
        c for c in df.columns if c not in exclude and pd.api.types.is_numeric_dtype(df[c])
    ]

    if not feature_cols:
        click.echo("No numeric feature columns found")
        raise SystemExit(1)

    click.echo(f"Running univariate Cox PH: {len(feature_cols)} features, {len(df)} patients...")

    rows = []
    for col in feature_cols:
        sub = df[[col, outcome, event]].dropna()
        if len(sub) < 10 or sub[col].std() == 0:
            continue
        try:
            cph = CoxPHFitter()
            cph.fit(sub, duration_col=outcome, event_col=event, show_progress=False)
            s = cph.summary
            rows.append(
                {
                    "feature": col,
                    "coef": float(s["coef"].iloc[0]),
                    "hr": float(s["exp(coef)"].iloc[0]),
                    "hr_ci_low": float(s["exp(coef) lower 95%"].iloc[0]),
                    "hr_ci_high": float(s["exp(coef) upper 95%"].iloc[0]),
                    "p": float(s["p"].iloc[0]),
                }
            )
        except Exception:
            continue

    if not rows:
        click.echo("No features produced valid Cox PH results")
        raise SystemExit(1)

    results_df = pd.DataFrame(rows).sort_values("p")

    os.makedirs(os.path.dirname(os.path.abspath(output)), exist_ok=True)
    results_df.to_csv(output, index=False)

    sig_count = int((results_df["p"] < 0.05).sum())
    click.echo(f"\n{'─'*62}")
    click.echo(
        f"Cox PH Univariate  |  {len(results_df)} features  |  {sig_count} significant (p<0.05)"
    )
    click.echo(f"{'─'*62}")
    click.echo(f"  {'Feature':<44} {'HR':>6}  {'p':>8}")
    click.echo(f"{'─'*62}")
    for _, row in results_df.head(top_n).iterrows():
        marker = "*" if row["p"] < 0.05 else " "
        click.echo(f"{marker} {row['feature']:<44} {row['hr']:>6.3f}  {row['p']:>8.4f}")
    click.echo(f"{'─'*62}")
    click.echo(f"\nFull results -> {output}")


@analyze.command()
@click.option(
    "--input",
    "-i",
    "input_path",
    required=True,
    type=click.Path(exists=True),
    help="Path to analysis_ready.csv (features + outcome)",
)
@click.option("--outcome", required=True, help="Outcome column (binary label or survival time)")
@click.option(
    "--event",
    default=None,
    help="Event column for survival (1=event). Omit for classification.",
)
@click.option(
    "--method",
    default="all",
    type=click.Choice(["model", "permutation", "shap", "all"]),
    help="Importance method (default: all)",
)
@click.option("--output", "-o", required=True, type=click.Path(), help="Output CSV path")
@click.option("--top-n", default=20, help="Features to show per method (default: 20)")
def importance(input_path, outcome, event, method, output, top_n):
    """Compute feature importance via model, permutation, and/or SHAP.

    \b
    For survival outcomes, provide --event to binarize (above/below median OS).
    For classification, omit --event.
    Methods: model (RF feature_importances_), permutation, shap.

    \b
    Examples:
        qr analyze importance \\
          --input artifacts/uploads/datasets/NSCLC/analysis_ready.csv \\
          --outcome OS_months --event OS_event --method all \\
          --output artifacts/uploads/datasets/NSCLC/feature_importance.csv

        qr analyze importance \\
          --input artifacts/uploads/datasets/ACRIN/analysis_ready.csv \\
          --outcome fdg_uptake_binary --method shap \\
          --output artifacts/uploads/datasets/ACRIN/feature_importance.csv
    """
    try:
        import numpy as np
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.inspection import permutation_importance as sk_permutation_importance
    except ImportError as e:
        click.echo(f"Required package not installed: {e}")
        click.echo("Run: pip install scikit-learn numpy")
        raise SystemExit(1)

    df = pd.read_csv(input_path)

    if outcome not in df.columns:
        click.echo(f"Outcome column '{outcome}' not found in {input_path}")
        click.echo(f"   Available: {', '.join(list(df.columns)[:10])}...")
        raise SystemExit(1)

    exclude = {"patient_id", "PatientID", outcome}
    if event:
        if event not in df.columns:
            click.echo(f"Event column '{event}' not found")
            raise SystemExit(1)
        exclude.add(event)
        median_os = df[outcome].median()
        y = (df[outcome] >= median_os).astype(int)
        mode_label = f"survival (binarized at median {median_os:.1f})"
    else:
        y = df[outcome]
        mode_label = "classification"

    feature_cols = [
        c for c in df.columns if c not in exclude and pd.api.types.is_numeric_dtype(df[c])
    ]
    if not feature_cols:
        click.echo("No numeric feature columns found")
        raise SystemExit(1)

    X = df[feature_cols].fillna(df[feature_cols].median())

    click.echo(
        f"Feature importance ({mode_label}): "
        f"{len(feature_cols)} features, {len(X)} patients, method={method}"
    )

    clf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
    clf.fit(X, y)

    rows = []
    methods_to_run = ["model", "permutation", "shap"] if method == "all" else [method]

    if "model" in methods_to_run:
        for feat, imp in zip(feature_cols, clf.feature_importances_):
            rows.append({"feature": feat, "importance": float(imp), "method": "model"})

    if "permutation" in methods_to_run:
        click.echo("  Computing permutation importance...")
        perm = sk_permutation_importance(clf, X, y, n_repeats=10, random_state=42, n_jobs=-1)
        for feat, imp, std in zip(feature_cols, perm.importances_mean, perm.importances_std):
            rows.append(
                {
                    "feature": feat,
                    "importance": float(imp),
                    "importance_std": float(std),
                    "method": "permutation",
                }
            )

    if "shap" in methods_to_run:
        try:
            import shap

            click.echo("  Computing SHAP values...")
            explainer = shap.TreeExplainer(clf)
            shap_values = explainer(X)
            vals = shap_values.values
            if vals.ndim == 3:
                vals = vals[:, :, 1]
            mean_abs = np.abs(vals).mean(axis=0)
            for feat, imp in zip(feature_cols, mean_abs):
                rows.append({"feature": feat, "importance": float(imp), "method": "shap"})
        except ImportError:
            click.echo("  SHAP not available — skipping (pip install shap)")

    if not rows:
        click.echo("No importance results produced")
        raise SystemExit(1)

    results_df = pd.DataFrame(rows).sort_values(["method", "importance"], ascending=[True, False])

    os.makedirs(os.path.dirname(os.path.abspath(output)), exist_ok=True)
    results_df.to_csv(output, index=False)

    click.echo(f"\n{'─'*62}")
    click.echo(f"Feature Importance  |  {len(feature_cols)} features  |  {len(X)} patients")
    click.echo(f"{'─'*62}")
    for m in methods_to_run:
        sub = results_df[results_df["method"] == m].head(top_n)
        if sub.empty:
            continue
        click.echo(f"\n  [{m.upper()}]")
        click.echo(f"  {'Feature':<44} {'Score':>8}")
        click.echo(f"  {'─'*54}")
        for _, row in sub.iterrows():
            click.echo(f"  {row['feature']:<44} {row['importance']:>8.4f}")
    click.echo(f"\nFull results -> {output}")


@analyze.command()
@click.option(
    "--input",
    "-i",
    "input_path",
    required=True,
    type=click.Path(exists=True),
    help="Path to CSV with patient_id + outcome + radiomics features",
)
@click.option("--outcome", required=True, help="Binary or ordinal outcome column")
@click.option(
    "--output",
    "-o",
    required=True,
    type=click.Path(),
    help="Output CSV path for classification results",
)
@click.option("--top-n", default=20, help="Features to show in console summary (default: 20)")
def classify(input_path, outcome, output, top_n):
    """Run univariate logistic regression classification on radiomics features.

    \b
    Outcome column should be binary (0/1) or binarized ordinal.
    Features are ranked by p-value. Output CSV contains: feature, auc, coef, p.

    \b
    Examples:
        qr analyze classify \\
          --input artifacts/uploads/datasets/ACRIN/analysis_ready.csv \\
          --outcome fdg_uptake_binary \\
          --output artifacts/uploads/datasets/ACRIN/classify_results.csv
    """
    try:
        import numpy as np
        from scipy.special import expit
        from scipy.stats import chi2_contingency
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import roc_auc_score
        from statsmodels.api import Logit
    except ImportError as e:
        click.echo(f"Required package not installed: {e}")
        click.echo("Run: pip install scikit-learn statsmodels scipy")
        raise SystemExit(1)

    df = pd.read_csv(input_path)

    if outcome not in df.columns:
        click.echo(f"Outcome column '{outcome}' not found in {input_path}")
        click.echo(f"   Available columns: {', '.join(list(df.columns)[:10])}...")
        raise SystemExit(1)

    exclude = {"patient_id", "PatientID", outcome}
    feature_cols = [
        c for c in df.columns if c not in exclude and pd.api.types.is_numeric_dtype(df[c])
    ]

    if not feature_cols:
        click.echo("No numeric feature columns found")
        raise SystemExit(1)

    y_all = df[outcome].dropna()
    unique_vals = y_all.unique()
    if len(unique_vals) < 2:
        click.echo(f"Outcome column '{outcome}' has fewer than 2 unique values")
        raise SystemExit(1)

    click.echo(
        f"Running univariate logistic regression: {len(feature_cols)} features, "
        f"{len(df)} patients..."
    )

    rows = []
    for col in feature_cols:
        sub = df[[col, outcome]].dropna()
        if len(sub) < 10 or sub[col].std() == 0:
            continue
        try:
            import statsmodels.api as sm

            X = sm.add_constant(sub[col].values.astype(float))
            y = sub[outcome].values.astype(float)
            model = sm.Logit(y, X)
            fit = model.fit(disp=False)
            coef = float(fit.params[1])
            p = float(fit.pvalues[1])
            y_pred = fit.predict(X)
            auc = float(roc_auc_score(y, y_pred))
            rows.append({"feature": col, "coef": coef, "auc": auc, "p": p})
        except Exception:
            continue

    if not rows:
        click.echo("No features produced valid logistic regression results")
        raise SystemExit(1)

    results_df = pd.DataFrame(rows).sort_values("p")

    os.makedirs(os.path.dirname(os.path.abspath(output)), exist_ok=True)
    results_df.to_csv(output, index=False)

    sig_count = int((results_df["p"] < 0.05).sum())
    click.echo(f"\n{'─'*62}")
    click.echo(
        f"Logistic Regression  |  {len(results_df)} features  |  {sig_count} significant (p<0.05)"
    )
    click.echo(f"{'─'*62}")
    click.echo(f"  {'Feature':<40} {'AUC':>6}  {'p':>8}")
    click.echo(f"{'─'*62}")
    for _, row in results_df.head(top_n).iterrows():
        marker = "*" if row["p"] < 0.05 else " "
        click.echo(f"{marker} {row['feature']:<40} {row['auc']:>6.3f}  {row['p']:>8.4f}")
    click.echo(f"{'─'*62}")
    click.echo(f"\nFull results -> {output}")
