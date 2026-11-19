import argparse
import json
import logging
import time

import numpy as np
import pandas as pd

from aqi import aqi_index, clean, config, features, tracking, validate
from aqi.models import baselines, boosting

log = logging.getLogger("aqi.evaluate")

TRAIN_STRIDE_HOURS = 3
ORIGIN_STEP_HOURS = 6
MODELS = ["persistence", "seasonal_naive", "xgboost"]


def load_best_params() -> dict:
    path = config.REPORTS_DIR / "best_params.json"
    return json.loads(path.read_text()) if path.exists() else {}


def horizon_tables(
    hourly: pd.DataFrame, horizons: list[int], use_future_weather: bool = True
) -> dict[int, pd.DataFrame]:
    origin_table = features.build_origin_table(hourly)
    tables = {}
    for h in horizons:
        table = features.build_horizon_table(hourly, origin_table, h, use_future_weather)
        numeric = table.columns.difference(["location"])
        table[numeric] = table[numeric].astype("float32")
        tables[h] = table.sort_index(kind="stable")
    return tables


def training_rows(table: pd.DataFrame, fold: validate.Fold, horizon: int) -> pd.DataFrame:
    mask = validate.train_mask(table.index, fold.train_end, horizon)
    mask &= (table.index.hour % TRAIN_STRIDE_HOURS == 0) & table["target"].notna().to_numpy()
    return table[mask]


def test_rows(table: pd.DataFrame, fold: validate.Fold) -> pd.DataFrame:
    mask = validate.test_mask(table.index, fold, ORIGIN_STEP_HOURS)
    mask &= table["target"].notna().to_numpy() & table["pm25_now"].notna().to_numpy()
    return table[mask]


def baseline_columns(hourly: pd.DataFrame, rows: pd.DataFrame, horizon: int) -> pd.DataFrame:
    out = pd.DataFrame(index=rows.index)
    out["location"] = rows["location"].to_numpy()
    out["persistence"] = np.nan
    out["seasonal_naive"] = np.nan
    for slug, part in hourly.groupby("location"):
        pm = part["pm2_5"]
        sel = (out["location"] == slug).to_numpy()
        origins = rows.index[sel]
        out.loc[sel, "persistence"] = baselines.persistence(pm, origins, horizon)
        out.loc[sel, "seasonal_naive"] = baselines.seasonal_naive(pm, origins, horizon)
    return out.drop(columns="location")


def run_walk_forward(hourly, tables, folds, with_intervals=True, params=None, label="walk_forward") -> pd.DataFrame:
    results = []
    started = time.time()
    for fold in folds:
        for h, table in tables.items():
            cols = features.feature_columns(table)
            train = training_rows(table, fold, h)
            test = test_rows(table, fold)
            if len(train) < 1000 or test.empty:
                continue
            y_train = train["target"].to_numpy(dtype=float)
            model = boosting.fit_xgb(train[cols], y_train, params)
            frame = pd.DataFrame(
                {
                    "location": test["location"].to_numpy(),
                    "horizon": h,
                    "fold": fold.name,
                    "y_true": test["target"].to_numpy(dtype=float),
                    "xgboost": model.predict(test[cols]),
                },
                index=test.index,
            )
            if with_intervals:
                lo = boosting.fit_quantile(train[cols], y_train, 0.1)
                hi = boosting.fit_quantile(train[cols], y_train, 0.9)
                frame["xgb_lo"] = lo.predict(test[cols])
                frame["xgb_hi"] = hi.predict(test[cols])
            frame = frame.join(baseline_columns(hourly, test, h))
            results.append(frame)
        log.info("%s fold %s done (%.0fs)", label, fold.name, time.time() - started)
    return pd.concat(results).rename_axis("origin").reset_index()


def to_long(preds: pd.DataFrame, models: list[str] = MODELS) -> pd.DataFrame:
    parts = []
    for m in models:
        part = preds[["origin", "location", "horizon", "fold", "y_true", m]].rename(columns={m: "y_pred"})
        part["model"] = m
        parts.append(part.dropna(subset=["y_pred", "y_true"]))
    return pd.concat(parts)


def horizon_summary(preds: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    long = to_long(preds)
    long["season"] = validate.season_of(pd.DatetimeIndex(long["origin"]))
    overall = validate.point_metrics(long, ["model", "horizon"])
    base = overall[overall["model"] == "persistence"].set_index("horizon")["mae"]
    overall["skill_vs_persistence"] = 1 - overall["mae"] / overall["horizon"].map(base)
    by_season = validate.point_metrics(long, ["model", "horizon", "season"])
    base_s = by_season[by_season["model"] == "persistence"].set_index(["horizon", "season"])["mae"]
    keys = list(zip(by_season["horizon"], by_season["season"]))
    by_season["skill_vs_persistence"] = 1 - by_season["mae"].to_numpy() / np.array([base_s.loc[k] for k in keys])
    return overall, by_season


def observed_windows(hourly: pd.DataFrame) -> dict[str, pd.Series]:
    return {slug: part["pm2_5"] for slug, part in hourly.groupby("location")}


def category_summary(preds: pd.DataFrame, hourly: pd.DataFrame, models=MODELS) -> pd.DataFrame:
    pm_by_loc = observed_windows(hourly)
    anchors = sorted(preds["horizon"].unique())
    rows = []
    for model in models:
        wide = preds.pivot_table(index=["location", "origin"], columns="horizon", values=model)
        wide = wide.dropna()
        true_codes = {h: [] for h in anchors}
        pred_codes = {h: [] for h in anchors}
        for slug, part in wide.groupby(level="location"):
            pm = pm_by_loc[slug]
            origins = pd.DatetimeIndex(part.index.get_level_values("origin"))
            window = np.stack(
                [pm.reindex(origins - pd.Timedelta(hours=k)).to_numpy() for k in range(23, -1, -1)], axis=1
            )
            keep = ~np.isnan(window).any(axis=1)
            values = part.to_numpy()
            true_means = {h: validate.true_24h_mean_at(pm, origins, h) for h in anchors}
            for i in np.flatnonzero(keep):
                pred_means = validate.predicted_24h_means(window[i], anchors, values[i])
                for j, h in enumerate(anchors):
                    true_codes[h].append(aqi_index.category_code(true_means[h][i]))
                    pred_codes[h].append(aqi_index.category_code(pred_means[j]))
        for h in anchors:
            rows.append(
                {
                    "model": model,
                    "horizon": h,
                    **validate.category_metrics(np.array(true_codes[h]), np.array(pred_codes[h])),
                }
            )
    return pd.DataFrame(rows)


def interval_scales(preds: pd.DataFrame, target: float = 0.8) -> dict[int, float]:
    """Per-horizon factor that widens the quantile interval until it reaches the target coverage."""
    scales = {}
    grid = np.arange(1.0, 3.01, 0.05)
    for h, part in preds.dropna(subset=["xgb_lo"]).groupby("horizon"):
        y, p = part["y_true"].to_numpy(), part["xgboost"].to_numpy()
        lo, hi = np.minimum(part["xgb_lo"].to_numpy(), p), np.maximum(part["xgb_hi"].to_numpy(), p)
        chosen = grid[-1]
        for k in grid:
            if np.mean((y >= p - (p - lo) * k) & (y <= p + (hi - p) * k)) >= target:
                chosen = k
                break
        scales[int(h)] = float(round(chosen, 2))
    return scales


def apply_scales(preds: pd.DataFrame, scales: dict[int, float]) -> pd.DataFrame:
    out = preds.copy()
    k = out["horizon"].map(scales).fillna(1.0)
    p = out["xgboost"]
    lo, hi = np.minimum(out["xgb_lo"], p), np.maximum(out["xgb_hi"], p)
    out["xgb_lo_cal"] = np.clip(p - (p - lo) * k, 0, None)
    out["xgb_hi_cal"] = p + (hi - p) * k
    return out


def interval_summary(preds: pd.DataFrame) -> pd.DataFrame:
    if "xgb_lo" not in preds:
        return pd.DataFrame()
    rows = []
    for h, part in preds.dropna(subset=["xgb_lo"]).groupby("horizon"):
        rows.append(
            {
                "horizon": h,
                "variant": "raw quantile",
                **validate.interval_metrics(part["y_true"], part["xgb_lo"], part["xgb_hi"]),
            }
        )
        if "xgb_lo_cal" in part:
            rows.append(
                {
                    "horizon": h,
                    "variant": "recalibrated",
                    **validate.interval_metrics(part["y_true"], part["xgb_lo_cal"], part["xgb_hi_cal"]),
                }
            )
    return pd.DataFrame(rows)


def save_reports(preds, hourly, prefix: str) -> dict[str, pd.DataFrame]:
    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    overall, by_season = horizon_summary(preds)
    cats = category_summary(preds, hourly)
    intervals = interval_summary(preds)
    preds.to_parquet(config.REPORTS_DIR / f"{prefix}_predictions.parquet")
    for name, table in {"horizon": overall, "season": by_season, "category": cats, "interval": intervals}.items():
        if not table.empty:
            table.to_csv(config.REPORTS_DIR / f"{prefix}_{name}.csv", index=False)
    return {"horizon": overall, "season": by_season, "category": cats, "interval": intervals}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--folds", type=int, default=12)
    parser.add_argument("--horizons", type=int, nargs="*", default=config.HORIZONS)
    parser.add_argument("--no-intervals", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    hourly = clean.load_processed()
    data_end = hourly.index.max()
    folds, holdout = validate.walk_forward_folds(data_end, n_test_months=args.folds)
    tables = horizon_tables(hourly, args.horizons)
    params = load_best_params()
    with tracking.run("walk_forward", {"folds": args.folds, "horizons": args.horizons, "xgb_params": params}) as mlf:
        preds = run_walk_forward(hourly, tables, folds, not args.no_intervals, params)
        results = save_reports(preds, hourly, "walk_forward")
        for _, row in results["horizon"].iterrows():
            tracking.log_metrics(mlf, {f"mae_{row['model']}_h{int(row['horizon'])}": row["mae"]})
        print(results["horizon"].round(3).to_string())


if __name__ == "__main__":
    main()
