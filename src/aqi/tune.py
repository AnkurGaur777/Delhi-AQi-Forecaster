import json
import logging

import numpy as np
import optuna
import pandas as pd

from aqi import clean, config, features, tracking, validate
from aqi.evaluate import TRAIN_STRIDE_HOURS, horizon_tables
from aqi.models import boosting

log = logging.getLogger("aqi.tune")
TUNE_HORIZON = 24
N_TRIALS = 15


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    hourly = clean.load_processed()
    folds, _ = validate.walk_forward_folds(hourly.index.max())
    first_test = folds[0].test_start
    table = horizon_tables(hourly, [TUNE_HORIZON])[TUNE_HORIZON]
    cols = features.feature_columns(table)
    usable = table["target"].notna().to_numpy() & (table.index.hour % TRAIN_STRIDE_HOURS == 0)
    valid_start = first_test - pd.DateOffset(months=3)
    train = table[usable & validate.train_mask(table.index, valid_start, TUNE_HORIZON)]
    valid = table[usable & (table.index >= valid_start) & validate.train_mask(table.index, first_test, TUNE_HORIZON)]

    def objective(trial: optuna.Trial) -> float:
        params = {
            "max_depth": trial.suggest_int("max_depth", 3, 8),
            "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.12, log=True),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 20),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.5, 10.0, log=True),
        }
        model = boosting.fit_xgb(train[cols], train["target"].to_numpy(float), params)
        return float(np.mean(np.abs(model.predict(valid[cols]) - valid["target"].to_numpy(float))))

    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=N_TRIALS)
    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    (config.REPORTS_DIR / "best_params.json").write_text(json.dumps(study.best_params, indent=2))
    with tracking.run("optuna_h24", {"trials": N_TRIALS, "horizon": TUNE_HORIZON}) as mlf:
        tracking.log_metrics(mlf, {"best_valid_mae": study.best_value})
    log.info("best valid MAE %.3f with %s", study.best_value, study.best_params)


if __name__ == "__main__":
    main()
