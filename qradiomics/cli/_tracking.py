"""Optional MLflow tracking glue for `qr ml` commands.

Design goals:
- Zero hard dependency on mlflow. The CLI must run unchanged when mlflow is
  not installed or `MLFLOW_TRACKING_URI` is unset.
- One join key across the four-server stack (Prefect / Nextflow / nf-weblog /
  MLflow). We honour `NEXTFLOW_RUN_ID` and `PREFECT_FLOW_RUN_ID` env vars
  when present, exposing them as tags on the run.
- Resolve experiment name from (flag → env → cohort-derived default).

Usage in a CLI command::

    from qradiomics.cli._tracking import mlflow_run, log_params, log_metrics, log_artifact

    with mlflow_run(experiment=cohort, run_name=run_name) as run_id:
        log_params({...})
        ... do the work ...
        log_metrics({...})
        log_artifact(model_path)

When tracking is disabled, every helper is a no-op and `run_id` is None.
"""
from __future__ import annotations

import contextlib
import importlib.util
import os
import warnings
from pathlib import Path
from typing import Any, Dict, Iterator, Optional


_REACHABILITY_CACHE: Dict[str, bool] = {}


def _reachable(uri: str, timeout: float = 1.0) -> bool:
    """Quick HEAD probe against the MLflow REST root. Cached per URI per process."""
    if uri in _REACHABILITY_CACHE:
        return _REACHABILITY_CACHE[uri]
    try:
        import urllib.request

        req = urllib.request.Request(uri, method="HEAD")
        with urllib.request.urlopen(req, timeout=timeout):  # noqa: S310
            _REACHABILITY_CACHE[uri] = True
            return True
    except Exception:
        _REACHABILITY_CACHE[uri] = False
        return False


def _enabled() -> bool:
    """Tracking is enabled when MLFLOW_TRACKING_URI is set, mlflow imports, and
    the server responds within a short timeout. The reachability result is
    cached for the process, so the probe runs at most once per URI."""
    uri = os.environ.get("MLFLOW_TRACKING_URI")
    if not uri:
        return False
    if importlib.util.find_spec("mlflow") is None:
        return False
    # Local-file URIs (file://… or no scheme) don't need a network probe.
    if uri.startswith("file:") or not uri.startswith(("http://", "https://")):
        return True
    return _reachable(uri)


def _warn_once(msg: str) -> None:
    """Emit a single warning per unique message per process."""
    if msg in _warn_once._seen:  # type: ignore[attr-defined]
        return
    _warn_once._seen.add(msg)  # type: ignore[attr-defined]
    warnings.warn(msg, RuntimeWarning, stacklevel=2)


_warn_once._seen = set()  # type: ignore[attr-defined]


def _resolve_experiment(experiment: Optional[str], default: str) -> str:
    return (
        experiment
        or os.environ.get("MLFLOW_EXPERIMENT_NAME")
        or default
    )


def _join_tags() -> Dict[str, str]:
    """Tags that link an MLflow run to its upstream Prefect / Nextflow run."""
    tags = {}
    for env_key, tag_key in (
        ("NEXTFLOW_RUN_ID", "nextflow.run_id"),
        ("PREFECT_FLOW_RUN_ID", "prefect.flow_run_id"),
        ("QR_COHORT", "qr.cohort"),
    ):
        v = os.environ.get(env_key)
        if v:
            tags[tag_key] = v
    return tags


@contextlib.contextmanager
def mlflow_run(
    experiment: Optional[str] = None,
    run_name: Optional[str] = None,
    default_experiment: str = "qr_ml",
    run_id: Optional[str] = None,
) -> Iterator[Optional[str]]:
    """Context manager that starts (or resumes) an MLflow run if enabled.

    Yields the run id (str) when tracking is on, otherwise None. If MLflow is
    configured but the server is unreachable, degrades to no-op with a warning
    so the underlying work still completes.
    """
    if not _enabled():
        yield None
        return

    try:
        import mlflow

        if run_id:
            with mlflow.start_run(run_id=run_id) as run:
                yield run.info.run_id
            return

        mlflow.set_experiment(_resolve_experiment(experiment, default_experiment))
        tags = _join_tags()
        with mlflow.start_run(run_name=run_name, tags=tags or None) as run:
            yield run.info.run_id
    except Exception as e:
        _warn_once(
            f"MLflow tracking disabled: {type(e).__name__}: {e}. "
            f"Set MLFLOW_TRACKING_URI to an unreachable server? Continuing without tracking."
        )
        yield None


def log_params(params: Dict[str, Any]) -> None:
    if not _enabled():
        return
    try:
        import mlflow

        safe = {k: (v if v is not None else "") for k, v in params.items()}
        mlflow.log_params(safe)
    except Exception:
        pass


def log_metrics(metrics: Dict[str, Any]) -> None:
    if not _enabled():
        return
    try:
        import mlflow

        for k, v in metrics.items():
            if v is None:
                continue
            try:
                mlflow.log_metric(k, float(v))
            except (TypeError, ValueError):
                continue
    except Exception:
        pass


def log_artifact(path: str | Path) -> None:
    if not _enabled():
        return
    try:
        import mlflow

        p = Path(path)
        if p.exists():
            mlflow.log_artifact(str(p))
    except Exception:
        pass


def log_dict(obj: Dict[str, Any], filename: str) -> None:
    """Persist a dict as a JSON artifact attached to the active run."""
    if not _enabled():
        return
    try:
        import mlflow

        mlflow.log_dict(obj, filename)
    except Exception:
        pass


def active_run_id() -> Optional[str]:
    if not _enabled():
        return None
    try:
        import mlflow

        run = mlflow.active_run()
        return run.info.run_id if run else None
    except Exception:
        return None
