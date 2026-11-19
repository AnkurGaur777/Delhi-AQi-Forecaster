import logging
from contextlib import contextmanager

from aqi import config

log = logging.getLogger(__name__)


@contextmanager
def run(name: str, params: dict | None = None):
    """Log to MLflow when it is installed; otherwise a no-op so the pipeline still runs."""
    try:
        import mlflow
    except ImportError:
        log.warning("mlflow not installed, skipping experiment tracking")
        yield None
        return
    mlflow.set_tracking_uri(f"sqlite:///{config.ROOT / 'mlflow.db'}")
    mlflow.set_experiment("delhi-aqi-forecaster")
    with mlflow.start_run(run_name=name):
        if params:
            mlflow.log_params({k: str(v) for k, v in params.items()})
        yield mlflow


def log_metrics(mlflow, metrics: dict[str, float]) -> None:
    if mlflow is None:
        return
    clean = {k: float(v) for k, v in metrics.items() if v == v}
    mlflow.log_metrics(clean)
