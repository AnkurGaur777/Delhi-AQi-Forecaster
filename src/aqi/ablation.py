import logging

import pandas as pd

from aqi import clean, config, evaluate, features, validate
from aqi.models import boosting

log = logging.getLogger("aqi.ablation")
HORIZON = 24


def variant_mae(hourly, table, folds, params, log_target: bool, drop_future: bool) -> float:
    errors = []
    for fold in folds:
        cols = features.feature_columns(table)
        if drop_future:
            cols = [c for c in cols if c not in features.future_only_columns(table)]
        train = evaluate.training_rows(table, fold, HORIZON)
        test = evaluate.test_rows(table, fold)
        model = boosting.fit_xgb(train[cols], train["target"].to_numpy(float), params, log_target=log_target)
        errors.append(abs(model.predict(test[cols]) - test["target"].to_numpy(float)))
    return float(pd.concat([pd.Series(e) for e in errors]).mean())


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    hourly = clean.load_processed()
    folds, _ = validate.walk_forward_folds(hourly.index.max())
    folds = folds[::2]
    table = evaluate.horizon_tables(hourly, [HORIZON])[HORIZON]
    params = evaluate.load_best_params()
    rows = []
    for name, log_target, drop_future in [
        ("log target, with forecast weather", True, False),
        ("raw target, with forecast weather", False, False),
        ("log target, without forecast weather", True, True),
    ]:
        rows.append({"variant": name, "mae_h24": variant_mae(hourly, table, folds, params, log_target, drop_future)})
        log.info("%s done", name)
    result = pd.DataFrame(rows)
    result["note"] = f"every second walk-forward fold ({len(folds)} of 12)"
    result.to_csv(config.REPORTS_DIR / "ablation_h24.csv", index=False)
    print(result.round(3).to_string())


if __name__ == "__main__":
    main()
