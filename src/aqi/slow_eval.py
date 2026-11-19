import argparse
import logging
import time

import numpy as np
import pandas as pd

from aqi import clean, config, validate
from aqi.models import prophet_model, sarimax

log = logging.getLogger("aqi.slow_eval")
LOCATION = "near_ito"
ORIGIN_STEP_HOURS = 24
ORIGIN_HOUR_UTC = 0


def origin_grid(fold: validate.Fold) -> pd.DatetimeIndex:
    grid = pd.date_range(fold.test_start, fold.test_end - pd.Timedelta(hours=1), freq=f"{ORIGIN_STEP_HOURS}h")
    return grid[grid.hour == ORIGIN_HOUR_UTC]


def run(folds_count: int, with_prophet: bool) -> pd.DataFrame:
    hourly = clean.load_processed()
    part = hourly[hourly["location"] == LOCATION].drop(columns="location")
    folds, _ = validate.walk_forward_folds(hourly.index.max(), n_test_months=12)
    folds = folds[-folds_count:]
    rows = []
    started = time.time()
    use_prophet = with_prophet and prophet_model.prophet_available()
    for fold in folds:
        history = part[part.index < fold.train_end]
        params = sarimax.fit_params(history.dropna(subset=["pm2_5"]).interpolate(limit=3))
        prophet_fit = None
        prophet_path = None
        if use_prophet:
            block = part[(part.index >= fold.test_start) & (part.index < fold.test_end + pd.Timedelta(hours=49))]
            prophet_fit = prophet_model.fit(history.iloc[-24 * 365 * 2 :])
            prophet_path = prophet_model.predict(prophet_fit, block)
        for origin in origin_grid(fold):
            past = part[part.index <= origin]
            future = part[(part.index > origin) & (part.index <= origin + pd.Timedelta(hours=48))]
            if len(future) < 48 or past["pm2_5"].iloc[-24:].isna().any():
                continue
            sar = sarimax.forecast_from_origin(past, future, params)
            for h in config.HORIZONS:
                target_time = origin + pd.Timedelta(hours=h)
                truth = part["pm2_5"].get(target_time, np.nan)
                if np.isnan(truth):
                    continue
                row = {
                    "origin": origin,
                    "horizon": h,
                    "fold": fold.name,
                    "y_true": truth,
                    "persistence": past["pm2_5"].iloc[-1],
                    "seasonal_naive": part["pm2_5"].get(
                        origin + pd.Timedelta(hours=h - 24 * int(np.ceil(h / 24))), np.nan
                    ),
                    "sarimax": sar[h - 1],
                }
                if prophet_path is not None:
                    row["prophet"] = prophet_path.get(target_time, np.nan)
                rows.append(row)
        log.info("slow fold %s done (%.0fs)", fold.name, time.time() - started)
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--folds", type=int, default=12)
    parser.add_argument("--no-prophet", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    preds = run(args.folds, not args.no_prophet)
    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    preds.to_parquet(config.REPORTS_DIR / "slow_models_predictions.parquet")
    models = [m for m in ["persistence", "seasonal_naive", "sarimax", "prophet"] if m in preds]
    long = pd.concat(
        [preds[["horizon", "y_true", m]].rename(columns={m: "y_pred"}).assign(model=m) for m in models]
    ).dropna()
    table = validate.point_metrics(long, ["model", "horizon"])
    base = table[table["model"] == "persistence"].set_index("horizon")["mae"]
    table["skill_vs_persistence"] = 1 - table["mae"] / table["horizon"].map(base)
    table.to_csv(config.REPORTS_DIR / "slow_models_horizon.csv", index=False)
    print(table.round(3).to_string())


if __name__ == "__main__":
    main()
