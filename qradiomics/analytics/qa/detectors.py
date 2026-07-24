"""Automatically-computable detectors for the radiomics-QA rubric.

Each detector returns a :class:`~qradiomics.analytics.qa.scorecard.Check`.
Detectors that need data the caller did not supply return an ``NA`` check
with an explanatory message rather than fabricating a numeric result.

Implemented (computable) detectors:

* ``detect_c01_leakage``    — AST static analysis of a pipeline ``.py`` file.
* ``detect_c05_calibration`` / ``detect_c06_discrimination`` — scan a
  results dir / column set for calibration and discrimination-with-CI artifacts.
* ``detect_c07_epv``        — events-per-variable from explicit counts or a table.
* ``detect_c08_c09_stability`` — wrap the ICC filter on a replicate-keyed matrix.
* ``detect_c13_harmonization`` — pre/post-ComBat batch-association collapse.
* ``detect_c17_naming``     — IBSI/pyradiomics feature-name compliance.
* ``detect_c24_seed_split`` — fixed seeds + serialized split indices (repo grep).

Heuristic / evidence-only detectors return NA-with-evidence when the
underlying data is not present (C02 external cohort, C10/C11 perturbation).
"""
from __future__ import annotations

import ast
import re
from pathlib import Path
from typing import Iterable, Optional, Sequence

import pandas as pd

from qradiomics.analytics.qa.scorecard import Check, Verdict

__all__ = [
    "detect_c01_leakage",
    "detect_c05_calibration",
    "detect_c06_discrimination",
    "detect_c07_epv",
    "detect_c08_c09_stability",
    "detect_c13_harmonization",
    "detect_c17_naming",
    "detect_c18_documentation",
    "detect_c24_seed_split",
    "detect_c02_external",
    "detect_c10_c11_perturbation",
]


# --------------------------------------------------------------------------- #
# C01 — Feature-selection leakage (flagship, AST-based)
# --------------------------------------------------------------------------- #

# Estimator-like names that leak if fit on the full matrix before a split.
_LEAKY_FITTERS = {
    "StandardScaler", "MinMaxScaler", "RobustScaler", "Normalizer",
    "PowerTransformer", "QuantileTransformer", "PCA", "FeatureAgglomeration",
    "SelectKBest", "SelectPercentile", "RFE", "RFECV", "SelectFromModel",
    "VarianceThreshold", "Lasso", "LassoCV", "SimpleImputer", "KNNImputer",
}
# Substrings that mark a harmonization / stability operation (also leaky on
# full data): ComBat harmonization and ICC/stability filtering.
_LEAKY_NAME_SUBSTR = ("combat", "harmoniz", "icc_filter", "icc", "stability")
# Tokens that mark the train/test split or a CV split.
_SPLIT_TOKENS = (
    "train_test_split", "KFold", "StratifiedKFold", "GroupKFold",
    "RepeatedKFold", "RepeatedStratifiedKFold", "ShuffleSplit",
    "StratifiedShuffleSplit", "TimeSeriesSplit",
)
_CV_RUNNERS = (
    "cross_val_score", "cross_val_predict", "cross_validate", "GridSearchCV",
    "RandomizedSearchCV",
)
# Supervised feature-reduction / selection / harmonization helpers. Fitting any
# of these on the *whole* training matrix (outside the CV folds) and then
# evaluating that matrix by cross-validation is the canonical radiomics
# selection-leakage pattern: the reduction has already seen every fold.
_REDUCTION_SUBSTR = (
    "reduction", "reduce", "select", "agglomerat", "mutual_info", "cmi",
    "lasso", "rfe", "kbest", "combat", "harmoniz", "boruta", "relief",
)
# Argument names that signal a call is outcome-aware (supervised).
_OUTCOME_NAMES = {
    "y", "y_train", "ytrain", "label", "labels", "outcome", "outcomes",
    "target", "targets", "event", "events", "status",
}


def _call_name(node: ast.Call) -> str:
    """Best-effort dotted name of a call target (e.g. ``scaler.fit_transform``)."""
    func = node.func
    if isinstance(func, ast.Attribute):
        parts = [func.attr]
        cur = func.value
        while isinstance(cur, ast.Attribute):
            parts.append(cur.attr)
            cur = cur.value
        if isinstance(cur, ast.Name):
            parts.append(cur.id)
        return ".".join(reversed(parts))
    if isinstance(func, ast.Name):
        return func.id
    return ""


def _is_reduction_call(name: str) -> bool:
    """True if a call name looks like a feature-reduction/selection helper."""
    return any(s in name.lower() for s in _REDUCTION_SUBSTR)


def _call_has_outcome_arg(node: ast.Call) -> bool:
    """True if any positional/keyword argument is an outcome-like variable."""
    args = list(node.args) + [kw.value for kw in node.keywords]
    for a in args:
        if isinstance(a, ast.Name) and a.id.lower() in _OUTCOME_NAMES:
            return True
        if isinstance(a, ast.Attribute) and a.attr.lower() in _OUTCOME_NAMES:
            return True
    return False


def _in_loop(target: ast.AST, tree: ast.AST) -> bool:
    """True if ``target`` is lexically inside any for/while body (i.e. it may
    be re-fit per CV fold rather than once on the whole training set)."""
    for node in ast.walk(tree):
        if isinstance(node, (ast.For, ast.AsyncFor, ast.While)):
            for sub in ast.walk(node):
                if sub is target:
                    return True
    return False


def _has_cv(trees: "list[ast.AST]") -> bool:
    """True if any scanned file uses a CV split / cross-validation runner."""
    for tree in trees:
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                leaf = _call_name(node).rsplit(".", 1)[-1]
                if leaf in _SPLIT_TOKENS or leaf in _CV_RUNNERS:
                    return True
    return False


def detect_c01_leakage(pipeline_path: str | Path) -> Check:
    """Static-analysis leakage detector.

    Flags ``.fit`` / ``.fit_transform`` calls that (a) fit a known leaky
    transformer/selector/scaler, or invoke ComBat/ICC filtering, and (b)
    occur at module scope or *before* the first split/CV call, and (c) are
    not lexically inside a Pipeline construction.

    Returns FAIL (with offending file:line evidence) if any such call is
    found before the split; PARTIAL if leaky fits exist but only after the
    split is established (ambiguous — re-fit-per-fold not provable
    statically); PASS if no module-scope leaky fit is found.
    """
    path = Path(pipeline_path)
    files = _iter_py_files(path)
    if not files:
        return Check("C01", Verdict.NA,
                     message=f"No Python source found at {path}")

    findings: list[str] = []
    sel_findings: list[str] = []
    all_trees: list[ast.AST] = []
    safe_files = 0
    leaky_after_split = False

    for f in files:
        src = f.read_text(errors="replace")
        try:
            tree = ast.parse(src)
        except SyntaxError as e:
            findings.append(f"{f.name}: could not parse ({e})")
            continue
        all_trees.append(tree)

        split_line = _first_split_line(tree)
        in_pipeline_lines = _pipeline_arg_lines(tree)
        file_findings = []

        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            name = _call_name(node)
            attr = name.rsplit(".", 1)[-1] if name else ""

            # Outcome-aware feature reduction/selection/harmonization fit once
            # on the whole training matrix (not inside a CV-fold loop, not in a
            # Pipeline). This is the canonical radiomics selection-leakage case
            # — e.g. `fit_reduction(X_train, y)` then CV on the reduced matrix.
            if (_is_reduction_call(name) or attr in ("fit", "fit_transform")) \
                    and _call_has_outcome_arg(node) \
                    and node.lineno not in in_pipeline_lines \
                    and not _in_loop(node, tree):
                sel_findings.append(
                    f"{f.name}:{node.lineno}: `{name}(..., <outcome>)` fit once "
                    f"on the full training set, outside any CV-fold loop"
                )

            if attr not in ("fit", "fit_transform"):
                # also catch direct calls like combat_harmonize(...) / icc_filter(...)
                if not _is_leaky_func_call(name):
                    continue
                # a bare leaky-function call before split is also leakage
                if _at_module_scope(node, tree) and (
                    split_line is None or node.lineno < split_line
                ):
                    file_findings.append(
                        f"{f.name}:{node.lineno}: `{name}(...)` on full data "
                        f"before split"
                    )
                continue

            # .fit / .fit_transform on a leaky receiver
            receiver = name.rsplit(".", 1)[0] if "." in name else ""
            if not _receiver_is_leaky(receiver, src):
                continue
            if node.lineno in in_pipeline_lines:
                continue  # inside a Pipeline(...) construction — safe
            if _at_module_scope(node, tree) and (
                split_line is None or node.lineno < split_line
            ):
                file_findings.append(
                    f"{f.name}:{node.lineno}: `{name}` on full feature matrix "
                    f"before the data split"
                )
            elif split_line is not None and node.lineno < split_line:
                file_findings.append(
                    f"{f.name}:{node.lineno}: `{name}` fit before split (line "
                    f"{split_line})"
                )
            else:
                leaky_after_split = True

        if file_findings:
            findings.extend(file_findings)
        else:
            safe_files += 1

    # Dedup selection findings against the stronger pre-split findings.
    sel_findings = [s for s in sel_findings if s not in findings]
    cv_present = _has_cv(all_trees)

    if findings:
        return Check(
            "C01", Verdict.FAIL,
            message=f"{len(findings)} leakage signal(s): "
                    f"transformer/selector/harmonization fit on full data "
                    f"before the split.",
            evidence=findings,
        )
    if sel_findings and cv_present:
        return Check(
            "C01", Verdict.FAIL,
            message=f"{len(sel_findings)} outcome-aware feature-reduction fit(s) "
                    f"on the whole training set while cross-validation is used "
                    f"elsewhere — the reduction has seen every fold (selection "
                    f"leakage). Re-fit selection inside each CV fold.",
            evidence=sel_findings,
        )
    if sel_findings:
        return Check(
            "C01", Verdict.PARTIAL,
            message="Outcome-aware feature reduction/selection is fit once on "
                    "the full training set (no CV detected here). Safe only if "
                    "the test cohort is a genuinely untouched external holdout; "
                    "leaky if this matrix is later cross-validated.",
            evidence=sel_findings,
        )
    if leaky_after_split:
        return Check(
            "C01", Verdict.PARTIAL,
            message="Leaky transformers are fit only after the split, but "
                    "per-fold re-fitting could not be proven statically. "
                    "Prefer wrapping them in an sklearn Pipeline passed to CV.",
        )
    return Check(
        "C01", Verdict.PASS,
        message="No full-data selection/scaling/harmonization fit detected "
                "before the split.",
    )


def _is_leaky_func_call(name: str) -> bool:
    low = name.lower()
    return any(s in low for s in _LEAKY_NAME_SUBSTR if s not in ("icc",)) or \
        low.endswith("icc_filter")


def _receiver_is_leaky(receiver: str, src: str) -> bool:
    """Decide whether the object that ``.fit`` is called on is leaky.

    Heuristic: the receiver variable was assigned from a known leaky class,
    or its name itself signals scaling/selection/harmonization.
    """
    if not receiver:
        return False
    low = receiver.lower()
    if any(s in low for s in ("scaler", "selector", "pca", "combat",
                              "harmoniz", "imputer", "agglom", "lasso",
                              "rfe", "kbest")):
        return True
    # was it assigned `receiver = SomeLeakyClass(...)` anywhere?
    for cls in _LEAKY_FITTERS:
        if re.search(rf"\b{re.escape(receiver)}\s*=\s*{cls}\b", src):
            return True
    return False


def _first_split_line(tree: ast.AST) -> Optional[int]:
    lines = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            name = _call_name(node)
            leaf = name.rsplit(".", 1)[-1] if name else ""
            if leaf in _SPLIT_TOKENS or leaf in _CV_RUNNERS:
                lines.append(node.lineno)
    return min(lines) if lines else None


def _pipeline_arg_lines(tree: ast.AST) -> set[int]:
    """Line numbers of nodes lexically inside a ``Pipeline(...)`` /
    ``make_pipeline(...)`` call — treated as safe (re-fit per fold)."""
    safe: set[int] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            name = _call_name(node)
            leaf = name.rsplit(".", 1)[-1] if name else ""
            if leaf in ("Pipeline", "make_pipeline", "ColumnTransformer",
                        "make_column_transformer"):
                for sub in ast.walk(node):
                    ln = getattr(sub, "lineno", None)
                    if ln is not None:
                        safe.add(ln)
    return safe


def _at_module_scope(target: ast.AST, tree: ast.AST) -> bool:
    """True if ``target`` is not nested inside any function/loop body."""
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef,
                             ast.For, ast.AsyncFor, ast.While)):
            for sub in ast.walk(node):
                if sub is target:
                    return False
    return True


def _iter_py_files(path: Path) -> list[Path]:
    if path.is_file() and path.suffix == ".py":
        return [path]
    if path.is_dir():
        return sorted(p for p in path.rglob("*.py")
                      if "__pycache__" not in p.parts)
    return []


# --------------------------------------------------------------------------- #
# C05 / C06 — calibration & discrimination evidence in a results dir/columns
# --------------------------------------------------------------------------- #

_CALIB_TOKENS = ("calibration_curve", "calibrationdisplay", "brier",
                 "brier_score", "ece", "expected_calibration",
                 "calibration_slope", "calibration_intercept", "calib")
_DISCRIM_TOKENS = ("auc", "auroc", "auprc", "c_index", "cindex", "harrell",
                   "concordance")
_CI_TOKENS = ("ci_low", "ci_high", "_lower", "_upper", "_low", "_high",
              "ci95", "95ci", "sd", "std", "bootstrap", "n_bootstrap",
              "resample")


def _gather_tokens(results_dir: Optional[str | Path],
                   columns: Optional[Iterable[str]]) -> tuple[list[str], list[str]]:
    """Return (lowercased filenames, lowercased column/text tokens)."""
    file_tokens: list[str] = []
    text_tokens: list[str] = list(c.lower() for c in (columns or []))
    if results_dir:
        rd = Path(results_dir)
        if rd.is_dir():
            for p in rd.rglob("*"):
                if p.is_file():
                    file_tokens.append(p.name.lower())
                    if p.suffix.lower() in (".csv", ".parquet"):
                        text_tokens.extend(_table_columns(p))
    return file_tokens, text_tokens


def _table_columns(path: Path) -> list[str]:
    try:
        if path.suffix.lower() == ".parquet":
            cols = pd.read_parquet(path, engine="pyarrow").columns
        else:
            cols = pd.read_csv(path, nrows=0).columns
        return [str(c).lower() for c in cols]
    except Exception:
        return []


def detect_c05_calibration(results_dir: Optional[str | Path] = None,
                           columns: Optional[Iterable[str]] = None) -> Check:
    files, text = _gather_tokens(results_dir, columns)
    blob = " ".join(files + text)
    has_curve = ("calibration_curve" in blob or "calibrationdisplay" in blob
                 or any("calib" in f for f in files))
    has_score = any(t in blob for t in ("brier", "ece", "expected_calibration"))
    if not (files or text):
        return Check("C05", Verdict.NA,
                     message="No results dir / columns supplied to scan.")
    if has_curve and has_score:
        return Check("C05", Verdict.PASS,
                     message="Calibration curve and Brier/ECE artifacts found.")
    if has_curve or has_score:
        return Check("C05", Verdict.PARTIAL,
                     message="Only one of {calibration curve, Brier/ECE} found.")
    return Check("C05", Verdict.FAIL,
                 message="No calibration artifacts (curve/Brier/ECE) found.")


def detect_c06_discrimination(results_dir: Optional[str | Path] = None,
                              columns: Optional[Iterable[str]] = None) -> Check:
    files, text = _gather_tokens(results_dir, columns)
    blob = " ".join(files + text)
    has_discrim = any(t in blob for t in _DISCRIM_TOKENS)
    has_ci = any(t in blob for t in _CI_TOKENS)
    if not (files or text):
        return Check("C06", Verdict.NA,
                     message="No results dir / columns supplied to scan.")
    if has_discrim and has_ci:
        return Check("C06", Verdict.PASS,
                     message="Discrimination metric with uncertainty/CI found.")
    if has_discrim:
        return Check("C06", Verdict.FAIL,
                     message="Discrimination metric found but no CI/uncertainty.")
    return Check("C06", Verdict.FAIL,
                 message="No discrimination metric (AUC/C-index) found.")


# --------------------------------------------------------------------------- #
# C07 — events-per-variable
# --------------------------------------------------------------------------- #

def detect_c07_epv(n_events: Optional[int] = None,
                   n_features: Optional[int] = None,
                   *, features: Optional[pd.DataFrame] = None,
                   events_col: Optional[str] = None,
                   feature_cols: Optional[Sequence[str]] = None) -> Check:
    """Events-per-variable = n_events / n_features.

    Either supply explicit ``n_events`` and ``n_features``, or a table +
    ``events_col`` (positive class counted as the events) — in which case
    ``n_features`` is inferred from ``feature_cols`` or the numeric columns
    minus the events column.
    """
    if n_events is None and features is not None and events_col:
        if events_col not in features.columns:
            return Check("C07", Verdict.NA,
                         message=f"events-col '{events_col}' not in table.")
        col = features[events_col]
        coerced = pd.Series(pd.to_numeric(col, errors="coerce"))
        n_events = int((col == col.dropna().max()).sum()) if col.dropna().nunique() > 1 \
            else int(coerced.fillna(0).astype(bool).sum())
    if n_features is None:
        if feature_cols is not None:
            n_features = len(feature_cols)
        elif features is not None:
            excl = {events_col} if events_col else set()
            n_features = sum(
                1 for c in features.columns
                if c not in excl and pd.api.types.is_numeric_dtype(features[c])
            )
    if not n_events or not n_features:
        return Check("C07", Verdict.NA,
                     message="Insufficient inputs to compute EPV "
                             "(need n_events and n_features).")
    epv = n_events / n_features
    msg = f"EPV = {n_events}/{n_features} = {epv:.2f}"
    if epv >= 10:
        return Check("C07", Verdict.PASS, message=msg + " (>= 10).")
    if epv >= 5:
        return Check("C07", Verdict.PARTIAL, message=msg + " (5-10).")
    return Check("C07", Verdict.FAIL, message=msg + " (< 5).")


# --------------------------------------------------------------------------- #
# C08 / C09 — stability via the existing ICC filter
# --------------------------------------------------------------------------- #

def detect_c08_c09_stability(features: pd.DataFrame,
                             replicate_col: str,
                             patient_id_col: str,
                             feature_cols: Optional[Sequence[str]] = None,
                             *, threshold: float = 0.75) -> Check:
    """Split a replicate-keyed matrix into two measurement frames (the first
    two replicate levels per patient) and report % features ICC > 0.75 / 0.90.

    Graded verdict = fraction of features with ICC >= ``threshold`` (0.75).
    """
    from qradiomics.analytics.robustness import icc_filter

    if replicate_col not in features.columns:
        return Check("C08", Verdict.NA,
                     message=f"replicate-col '{replicate_col}' not in table.")
    levels = list(pd.unique(features[replicate_col].dropna()))
    if len(levels) < 2:
        return Check("C08", Verdict.NA,
                     message="Need >= 2 replicate levels for ICC.")
    a = features[features[replicate_col] == levels[0]]
    b = features[features[replicate_col] == levels[1]]
    if feature_cols is None:
        excl = {replicate_col, patient_id_col}
        feature_cols = [c for c in features.columns
                        if c not in excl
                        and pd.api.types.is_numeric_dtype(features[c])]
    if not feature_cols:
        return Check("C08", Verdict.NA, message="No numeric feature columns.")
    try:
        _, icc_tbl = icc_filter(a, b, feature_cols, patient_id_col,
                                threshold=threshold)
    except ImportError as e:
        return Check("C08", Verdict.NA, message=str(e))
    valid = icc_tbl["icc"].dropna()
    if valid.empty:
        return Check("C08", Verdict.NA, message="ICC undefined for all features.")
    frac75 = float((valid >= 0.75).mean())
    frac90 = float((valid >= 0.90).mean())
    return Check(
        "C08", Verdict.GRADED, fraction=frac75,
        message=f"{frac75 * 100:.0f}% features ICC > 0.75, "
                f"{frac90 * 100:.0f}% > 0.90 (n={len(valid)}).",
    )


# --------------------------------------------------------------------------- #
# C13 — harmonization adequacy (pre/post ComBat batch association)
# --------------------------------------------------------------------------- #

def detect_c13_harmonization(features: pd.DataFrame,
                             batch_col: str,
                             feature_cols: Optional[Sequence[str]] = None,
                             *, alpha: float = 0.05) -> Check:
    """Run Kruskal-Wallis per feature across batches before and after a
    ComBat call; report the fraction of features with batch association
    p < alpha pre vs post.

    PASS if the post-ComBat significant fraction collapses toward ~alpha
    (<= 2x alpha) from a materially larger pre-ComBat fraction.
    """
    from scipy.stats import kruskal

    from qradiomics.analytics.harmonization import combat_harmonize

    if batch_col not in features.columns:
        return Check("C13", Verdict.NA,
                     message=f"batch-col '{batch_col}' not in table.")
    batches = list(pd.unique(features[batch_col].dropna()))
    if len(batches) < 2:
        return Check("C13", Verdict.NA,
                     message="Single batch — harmonization not applicable.")
    if feature_cols is None:
        excl = {batch_col}
        feature_cols = [c for c in features.columns
                        if c not in excl
                        and pd.api.types.is_numeric_dtype(features[c])]
    if not feature_cols:
        return Check("C13", Verdict.NA, message="No numeric feature columns.")

    def _sig_fraction(df: pd.DataFrame) -> float:
        sig = 0
        tested = 0
        for f in feature_cols:
            groups = [pd.to_numeric(df.loc[df[batch_col] == b, f],
                                    errors="coerce").dropna().to_numpy()
                      for b in batches]
            groups = [g for g in groups if len(g) > 1 and g.std() > 0]
            if len(groups) < 2:
                continue
            tested += 1
            try:
                _, p = kruskal(*groups)
                if p < alpha:
                    sig += 1
            except ValueError:
                continue
        return sig / tested if tested else float("nan")

    pre = _sig_fraction(features)
    try:
        harmonized = combat_harmonize(features, list(feature_cols), batch_col)
    except Exception as e:  # pragma: no cover - defensive
        return Check("C13", Verdict.NA, message=f"ComBat failed: {e}")
    post = _sig_fraction(harmonized)
    if pd.isna(pre) or pd.isna(post):
        return Check("C13", Verdict.NA,
                     message="Batch-association test undefined.")
    msg = (f"batch-associated features: {pre * 100:.0f}% pre-ComBat -> "
           f"{post * 100:.0f}% post-ComBat (alpha={alpha}).")
    if post <= 2 * alpha and post < pre:
        return Check("C13", Verdict.PASS, message=msg + " Collapses toward ~5%.")
    if post < pre:
        return Check("C13", Verdict.PARTIAL,
                     message=msg + " Reduced but not to ~5%.")
    return Check("C13", Verdict.FAIL, message=msg + " No collapse.")


# --------------------------------------------------------------------------- #
# C17 / C18 — IBSI-compliant naming & feature documentation
# --------------------------------------------------------------------------- #

# pyradiomics: <imageType>_<featureClass>_<featureName>
# e.g. original_glcm_Contrast, wavelet-LLH_firstorder_Mean, log-sigma-3-0-mm-3D_...
_PYRAD_IMAGETYPES = (
    "original", "wavelet", "log", "logarithm", "exponential", "gradient",
    "square", "squareroot", "lbp", "lbp-2d", "lbp-3d",
)
_PYRAD_CLASSES = (
    "firstorder", "shape", "shape2d", "glcm", "glrlm", "glszm", "gldm",
    "ngtdm",
)
_PYRAD_RE = re.compile(
    r"^(?P<img>[A-Za-z0-9\-\.]+?)_(?P<cls>firstorder|shape2?d?|glcm|glrlm|glszm|"
    r"gldm|ngtdm)_(?P<name>[A-Za-z0-9]+)$"
)
# IBSI code style, e.g. morph.vol, cm.contrast, stat.mean
_IBSI_RE = re.compile(r"^[a-z0-9]+(\.[a-z0-9_]+)+$")


def _is_ibsi_name(col: str) -> bool:
    c = col.strip()
    m = _PYRAD_RE.match(c)
    if m:
        img = m.group("img").split("-")[0].lower()
        return img in _PYRAD_IMAGETYPES or img.startswith("wavelet") \
            or img.startswith("log")
    return bool(_IBSI_RE.match(c.lower()))


def detect_c17_naming(columns: Iterable[str],
                      *, ignore: Sequence[str] = ()) -> Check:
    """Fraction of feature columns matching the pyradiomics/IBSI name pattern.

    GRADED by the compliant fraction over candidate feature columns
    (columns in ``ignore`` — ids, clinical, outcome — are excluded).
    """
    ignore_low = {i.lower() for i in ignore}
    candidates = [c for c in columns
                  if str(c).lower() not in ignore_low]
    if not candidates:
        return Check("C17", Verdict.NA, message="No feature columns to check.")
    compliant = [c for c in candidates if _is_ibsi_name(str(c))]
    frac = len(compliant) / len(candidates)
    return Check(
        "C17", Verdict.GRADED, fraction=frac,
        message=f"{len(compliant)}/{len(candidates)} "
                f"({frac * 100:.0f}%) columns match IBSI/pyradiomics naming.",
    )


def detect_c18_documentation(columns: Iterable[str],
                             *, ignore: Sequence[str] = ()) -> Check:
    """Provenance coverage proxy: a feature is 'documented' when its name
    carries class+filter provenance (the IBSI/pyradiomics pattern). This is
    the column-only proxy; a real feature dictionary would supersede it."""
    naming = detect_c17_naming(columns, ignore=ignore)
    if naming.verdict is Verdict.NA:
        return Check("C18", Verdict.NA, message=naming.message)
    frac = naming.fraction or 0.0
    return Check(
        "C18", Verdict.GRADED, fraction=frac,
        message=f"Provenance proxy from feature names: {frac * 100:.0f}% "
                f"columns carry class/filter provenance. A standalone "
                f"feature dictionary would give exact coverage.",
    )


# --------------------------------------------------------------------------- #
# C24 — reproducible seed / split (repo grep)
# --------------------------------------------------------------------------- #

_SEED_RE = re.compile(r"\b(random_state|seed|random_seed)\s*=\s*\d+")
_SEED_CALL_RE = re.compile(r"(np\.random\.seed|torch\.manual_seed|"
                           r"random\.seed)\s*\(")
_SPLIT_FILE_RE = re.compile(r"(split|fold|partition).*\.(json|npy|parquet|csv)",
                            re.IGNORECASE)


def detect_c24_seed_split(pipeline_path: str | Path,
                          results_dir: Optional[str | Path] = None) -> Check:
    path = Path(pipeline_path)
    files = _iter_py_files(path)
    if not files:
        return Check("C24", Verdict.NA, message=f"No Python source at {path}.")
    seed_hits: list[str] = []
    for f in files:
        src = f.read_text(errors="replace")
        for i, line in enumerate(src.splitlines(), 1):
            if _SEED_RE.search(line) or _SEED_CALL_RE.search(line):
                seed_hits.append(f"{f.name}:{i}: {line.strip()[:80]}")
    # serialized split artifacts
    split_hits: list[str] = []
    search_roots = [path if path.is_dir() else path.parent]
    if results_dir:
        search_roots.append(Path(results_dir))
    for root in search_roots:
        if root.is_dir():
            for p in root.rglob("*"):
                if p.is_file() and _SPLIT_FILE_RE.search(p.name):
                    split_hits.append(str(p))
    has_seed = bool(seed_hits)
    has_split = bool(split_hits)
    ev = seed_hits[:5] + split_hits[:5]
    if has_seed and has_split:
        return Check("C24", Verdict.PASS,
                     message="Fixed seed AND serialized split indices found.",
                     evidence=ev)
    if has_seed:
        return Check("C24", Verdict.PARTIAL,
                     message="Fixed seed found but no serialized split file.",
                     evidence=ev)
    return Check("C24", Verdict.FAIL,
                 message="No fixed seed or serialized split found.")


# --------------------------------------------------------------------------- #
# Evidence-only / NA detectors for data we may not have
# --------------------------------------------------------------------------- #

def detect_c02_external(pipeline_path: Optional[str | Path] = None,
                        results_dir: Optional[str | Path] = None) -> Check:
    """Heuristic evidence scan for an independent-cohort evaluation block.

    Does not fabricate a numeric grade: reports PARTIAL when only
    internal-CV evidence is found, PASS when explicit external/holdout
    cohort tokens appear, NA when nothing is scannable.
    """
    tokens = ("external", "holdout", "hold_out", "validation_set", "test_cohort",
              "independent", "site", "institution", "lvhs", "external_cohort")
    cv_tokens = ("cross_val", "kfold", "stratifiedkfold", "train_test_split")
    found_ext: list[str] = []
    found_cv = False
    roots = []
    if pipeline_path:
        roots.extend(_iter_py_files(Path(pipeline_path)))
    blob_files = []
    if results_dir and Path(results_dir).is_dir():
        blob_files = [p for p in Path(results_dir).rglob("*") if p.is_file()]
    if not roots and not blob_files:
        return Check("C02", Verdict.NA,
                     message="No pipeline/results supplied to scan for "
                             "external validation.")
    for f in roots:
        low = f.read_text(errors="replace").lower()
        for t in tokens:
            if t in low:
                found_ext.append(f"{f.name}: token '{t}'")
        if any(c in low for c in cv_tokens):
            found_cv = True
    for p in blob_files:
        low = p.name.lower()
        if any(t in low for t in tokens):
            found_ext.append(f"file {p.name}")
    if found_ext:
        return Check("C02", Verdict.PASS,
                     message="External/holdout-cohort evidence found "
                             "(verify it is a genuinely independent cohort).",
                     evidence=found_ext[:8])
    if found_cv:
        return Check("C02", Verdict.PARTIAL,
                     message="Only internal CV/holdout from the same cohort "
                             "detected; no external-cohort evidence.")
    return Check("C02", Verdict.NA,
                 message="No validation evidence found to grade.")


def detect_c10_c11_perturbation(pipeline_path: Optional[str | Path] = None) -> Check:
    """Evidence scan for a perturbation-robustness harness (C10). Returns NA
    when no harness is present (per rubric: drop from denominator)."""
    if not pipeline_path:
        return Check("C10", Verdict.NA,
                     message="No pipeline supplied; perturbation harness "
                             "not assessable.")
    tokens = ("perturb", "gaussian noise", "rotate", "rotation", "translate",
              "translation", "jitter")
    hits: list[str] = []
    for f in _iter_py_files(Path(pipeline_path)):
        low = f.read_text(errors="replace").lower()
        for t in tokens:
            if t in low:
                hits.append(f"{f.name}: '{t}'")
    if hits:
        return Check("C10", Verdict.PARTIAL,
                     message="Perturbation-related code found but per-feature "
                             "ICC not computed here; run the harness to grade.",
                     evidence=hits[:8])
    return Check("C10", Verdict.NA,
                 message="No perturbation harness detected (NA per rubric).")
